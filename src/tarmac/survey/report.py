from __future__ import annotations

import html
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import umap
from sklearn.cluster import KMeans

from tarmac.report.umap_html import absolute_file_url, dialog_css, dialog_html, image_dialog_js
from tarmac.survey.telemetry import ROUTE_NOTICE

SEED = 42

QUALITY_COLORS = {
    1: "#1a9850",
    2: "#91cf60",
    3: "#fee08b",
    4: "#fc8d59",
    5: "#d73027",
}

QUALITY_COLORSCALE = [
    [0.00, QUALITY_COLORS[1]],
    [0.25, QUALITY_COLORS[2]],
    [0.50, QUALITY_COLORS[3]],
    [0.75, QUALITY_COLORS[4]],
    [1.00, QUALITY_COLORS[5]],
]

CLUSTER_COLORS = [
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#be123c",
    "#4d7c0f",
    "#7c3aed",
    "#ca8a04",
    "#0f766e",
    "#64748b",
]


def build_reports(out_dir: Path) -> dict[str, Path]:
    out_dir = out_dir.expanduser().resolve()
    samples = pd.read_parquet(out_dir / "samples.parquet")
    problems_path = out_dir / "problems_confirmed.parquet"
    if not problems_path.exists():
        problems_path = out_dir / "problems.parquet"
    problems = pd.read_parquet(problems_path)
    telemetry = pd.read_parquet(out_dir / "telemetry.parquet")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    summary["problem_data_source"] = str(problems_path)
    summary["problems_found"] = int(len(problems))

    map_path = out_dir / "map.html"
    table_path = out_dir / "problems_table.html"
    cluster_path = out_dir / "cluster_scatter.html"
    speed_path = out_dir / "speed_chart.html"
    index_path = out_dir / "index.html"
    map_path.write_text(_map_html(samples, problems, telemetry, summary, out_dir=out_dir), encoding="utf-8")
    table_path.write_text(_table_html(problems, summary, out_dir=out_dir), encoding="utf-8")
    cluster_path.write_text(_cluster_scatter_page(samples, summary, out_dir=out_dir), encoding="utf-8")
    speed_path.write_text(_speed_chart_page(samples, summary), encoding="utf-8")
    index_path.write_text(_index_html(summary, samples, problems, out_dir=out_dir), encoding="utf-8")
    return {
        "map_html": map_path,
        "problems_table_html": table_path,
        "cluster_scatter_html": cluster_path,
        "speed_chart_html": speed_path,
        "index_html": index_path,
    }


def _map_html(
    samples: pd.DataFrame,
    problems: pd.DataFrame,
    telemetry: pd.DataFrame,
    summary: dict[str, Any],
    *,
    out_dir: Path,
) -> str:
    route_points = [
        {
            "lat": float(row.lat),
            "lon": float(row.lon),
            "quality": _maybe_int(getattr(row, "quality_grade", None)),
            "t": float(row.t),
            "speed_kmh": _maybe_float(getattr(row, "speed_kmh", None)),
        }
        for row in samples.itertuples()
        if _valid_lat_lon(getattr(row, "lat", None), getattr(row, "lon", None))
    ]
    telemetry_points = [
        [float(row.lat), float(row.lon)]
        for row in telemetry.itertuples()
        if _valid_lat_lon(getattr(row, "lat", None), getattr(row, "lon", None))
    ]
    start = summary.get("start_location", {})
    start_point = _start_point(start, route_points)
    problem_points = []
    for row in problems.itertuples():
        payload = _problem_payload(row, out_dir=out_dir)
        payload["dialog"] = _dialog_payload(payload)
        problem_points.append(payload)
    speed_warning = _speed_warning(summary)
    route_notice = _route_notice(summary)
    data = {
        "route": route_points,
        "telemetryRoute": telemetry_points[:: max(1, len(telemetry_points) // 2000)] if telemetry_points else [],
        "problems": problem_points,
        "start": start_point,
        "qualityColors": QUALITY_COLORS,
        "notice": route_notice,
        "speedWarning": speed_warning,
        "hasGeo": bool(route_points or telemetry_points or start_point),
        "speedLabel": str(summary.get("speed_label", "")),
    }
    payload = json.dumps(data, ensure_ascii=False)
    title = html.escape(f"Tarmac survey map - {summary.get('run_name', 'survey')}")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
  <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; }}
    .banner {{ padding: 10px 14px; background: #fff3cd; border-bottom: 1px solid #e7d38a; font-weight: 700; }}
    #map {{ height: calc(100vh - 42px); width: 100vw; }}
    .legend {{ background: white; padding: 10px 12px; border: 1px solid #cfd6dd; border-radius: 6px; line-height: 1.4; box-shadow: 0 1px 5px rgba(0,0,0,0.2); }}
    .legend .swatch {{ display: inline-block; width: 16px; height: 10px; margin-right: 6px; border: 1px solid rgba(0,0,0,0.18); }}
    .popup-thumb {{ width: 220px; max-height: 150px; object-fit: cover; display: block; margin: 8px 0; border: 1px solid #d6dde3; }}
    .popup-title {{ font-weight: 700; margin-bottom: 4px; }}
    .popup-file {{ font-weight: 700; overflow-wrap: anywhere; }}
  </style>
  {dialog_css()}
</head>
<body>
  <div class="banner">{html.escape(route_notice)}{f" {html.escape(speed_warning)}" if speed_warning else ""}</div>
  <div id="map"></div>
  {dialog_html(include_style=False)}
  <script>
    const data = {payload};
    const map = L.map('map');
    L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png', {{
      maxZoom: 19,
      attribution: '&copy; OpenStreetMap contributors'
    }}).addTo(map);

    const baseRoute = L.polyline(data.telemetryRoute, {{ color: '#6b7280', weight: 3, opacity: 0.35 }});
    const qualitySegments = L.layerGroup();
    for (let i = 1; i < data.route.length; i++) {{
      const a = data.route[i - 1];
      const b = data.route[i];
      const q = b.quality || a.quality || 0;
      const color = data.qualityColors[q] || '#6b7280';
      L.polyline([[a.lat, a.lon], [b.lat, b.lon]], {{ color, weight: 6, opacity: 0.92 }}).addTo(qualitySegments);
    }}
    baseRoute.addTo(map);
    qualitySegments.addTo(map);

    if (data.start) {{
      const startMarker = L.marker([data.start.lat, data.start.lon], {{
        title: 'START'
      }}).bindPopup(`<b>START</b><br>${{data.start.lat.toFixed(6)}}, ${{data.start.lon.toFixed(6)}}<br>${{data.start.source}}`);
      startMarker.addTo(map);
    }}

    const problemLayer = L.layerGroup();
    for (const p of data.problems) {{
      if (!Number.isFinite(p.lat) || !Number.isFinite(p.lon)) continue;
      const marker = L.circleMarker([p.lat, p.lon], {{
        radius: 7,
        color: '#9f1239',
        weight: 2,
        fillColor: '#e11d48',
        fillOpacity: 0.85
      }});
      const image = p.thumbnail_file_url ? tarmacPopupImageLink(
        p,
        `<img class="popup-thumb" src="${{tarmacEscapeHtml(p.thumbnail_file_url)}}" alt="problem thumbnail">`
      ) : '';
      marker.bindPopup(`
        <div class="popup-title">${{p.timestamp_label}} · quality ${{p.quality_grade ?? 'n/a'}}</div>
        <div><b>Speed ${{data.speedLabel}}:</b> ${{p.speed_kmh.toFixed(1)}} km/h</div>
        <div><b>Location:</b> ${{p.lat.toFixed(6)}}, ${{p.lon.toFixed(6)}}</div>
        <div><b>Issues:</b> ${{p.issues.join(', ') || 'none'}}</div>
        ${{p.crack_confirmed ? `<div><b>Crack:</b> area ${{Number(p.crack_area_pct || 0).toFixed(3)}}%, length ${{Number(p.crack_length_px || 0).toFixed(0)}} px</div>` : ''}}
        <div><b>Image:</b> ${{tarmacPopupImageLink(p, `<span class="popup-file">${{tarmacEscapeHtml(p.filename)}}</span>`)}}</div>
        ${{image}}
      `);
      marker.addTo(problemLayer);
    }}
    problemLayer.addTo(map);

    const overlays = {{
      'Telemetry route': baseRoute,
      'Quality-colored samples': qualitySegments,
      'Problem markers': problemLayer
    }};
    L.control.layers(null, overlays, {{ collapsed: false }}).addTo(map);

    const bounds = [];
    for (const p of data.telemetryRoute) bounds.push(p);
    for (const p of data.route) bounds.push([p.lat, p.lon]);
    if (bounds.length) {{
      map.fitBounds(bounds, {{ padding: [30, 30] }});
    }} else if (data.start) {{
      map.setView([data.start.lat, data.start.lon], 16);
    }} else {{
      map.setView([0, 0], 2);
    }}

    const legend = L.control({{ position: 'bottomright' }});
    legend.onAdd = function() {{
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML = '<b>Quality grade</b><br>' +
        [1,2,3,4,5].map(q => `<span class="swatch" style="background:${{data.qualityColors[q]}}"></span>${{q}}`).join('<br>') +
        '<hr><b>GPS source</b><br>' + tarmacEscapeHtml(data.notice);
      return div;
    }};
    legend.addTo(map);

    function tarmacEscapeHtml(value) {{
      return String(value ?? '').replace(/[&<>"']/g, function(char) {{
        return {{ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }}[char];
      }});
    }}
    function tarmacEscapeAttr(value) {{
      return tarmacEscapeHtml(value).replace(/`/g, '&#96;');
    }}
    function tarmacPopupImageLink(p, innerHtml) {{
      const fileUrl = tarmacEscapeHtml(p.original_file_url || p.image_file_url || p.image_path || '');
      const filename = tarmacEscapeHtml(p.filename || p.image_path || '');
      const quality = tarmacEscapeHtml(p.quality_grade ?? '');
      const surface = tarmacEscapeHtml(p.surface_type || '');
      const confidence = tarmacEscapeHtml(p.confidence ?? '');
      const path = tarmacEscapeHtml(p.original_abs_path || p.image_abs_path || p.image_path || '');
      const dialog = tarmacEscapeAttr(JSON.stringify(p.dialog || {{}}));
      return `<a class="tarmac-image-link" href="${{fileUrl}}" data-src="${{fileUrl}}" data-path="${{path}}" data-filename="${{filename}}" data-quality="${{quality}}" data-surface="${{surface}}" data-confidence="${{confidence}}" data-dialog="${{dialog}}">${{innerHtml}}</a>`;
    }}
  </script>
  <script>{image_dialog_js(plot_id=None)}</script>
</body>
</html>
"""


def _table_html(problems: pd.DataFrame, summary: dict[str, Any], *, out_dir: Path) -> str:
    title = html.escape(f"Tarmac survey problem table - {summary.get('run_name', 'survey')}")
    speed_warning = _speed_warning(summary)
    route_notice = _route_notice(summary)
    table = _problem_table_markup(problems, summary, out_dir=out_dir, table_id="problem-table")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #17202a; }}
    header {{ padding: 20px 24px; background: #17202a; color: white; }}
    main {{ padding: 20px 24px; }}
    .banner {{ padding: 10px 12px; background: #fff3cd; border: 1px solid #e7d38a; margin-bottom: 16px; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d8dee5; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e4e9ee; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f5; cursor: pointer; user-select: none; }}
    tr.tarmac-dialog-row {{ cursor: zoom-in; }}
    tr.tarmac-dialog-row:hover {{ background: #f8fafc; }}
    img {{ width: 120px; max-height: 90px; object-fit: cover; border: 1px solid #d8dee5; display: block; }}
    a {{ color: #0f4c81; }}
    .file-link {{ font-weight: 700; overflow-wrap: anywhere; }}
  </style>
  {dialog_css()}
</head>
<body>
  <header><h1>Problem table</h1></header>
  <main>
    <div class="banner">{html.escape(route_notice)}{f" {html.escape(speed_warning)}" if speed_warning else ""}</div>
    {table}
  </main>
  {dialog_html(include_style=False)}
  <script>
    document.querySelectorAll('th').forEach((th, col) => {{
      th.addEventListener('click', () => {{
        const table = th.closest('table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const asc = th.dataset.asc !== 'true';
        rows.sort((a, b) => {{
          const av = a.children[col]?.dataset.sort ?? a.children[col]?.textContent ?? '';
          const bv = b.children[col]?.dataset.sort ?? b.children[col]?.textContent ?? '';
          const an = Number(av);
          const bn = Number(bv);
          const cmp = Number.isFinite(an) && Number.isFinite(bn) ? an - bn : av.localeCompare(bv);
          return asc ? cmp : -cmp;
        }});
        th.dataset.asc = String(asc);
        rows.forEach(row => tbody.appendChild(row));
      }});
    }});
  </script>
  <script>{image_dialog_js(plot_id=None)}</script>
</body>
</html>
"""


def _problem_table_markup(
    problems: pd.DataFrame,
    summary: dict[str, Any],
    *,
    out_dir: Path,
    table_id: str,
) -> str:
    rows: list[str] = []
    for row in problems.itertuples():
        payload = _problem_payload(row, out_dir=out_dir)
        dialog_payload = _dialog_payload(payload)
        dialog_attr = html.escape(json.dumps(dialog_payload, separators=(",", ":"), allow_nan=False), quote=True)
        thumb = ""
        filename_link = _image_link(
            file_url=payload["original_file_url"],
            file_path=payload["original_abs_path"],
            filename=payload["filename"],
            inner=html.escape(payload["filename"]),
            quality=str(payload["quality_grade"] or ""),
            surface=payload["surface_type"],
            confidence=str(payload.get("confidence") or ""),
            css_class="file-link",
            dialog_payload=dialog_payload,
        )
        if payload["thumbnail_file_url"]:
            thumb = _image_link(
                file_url=payload["original_file_url"],
                file_path=payload["original_abs_path"],
                filename=payload["filename"],
                inner=(
                    f'<img src="{html.escape(payload["thumbnail_file_url"], quote=True)}" '
                    'alt="problem thumbnail">'
                ),
                quality=str(payload["quality_grade"] or ""),
                surface=payload["surface_type"],
                confidence=str(payload.get("confidence") or ""),
                dialog_payload=dialog_payload,
            )
        rows.append(
            f'<tr class="tarmac-dialog-row" data-dialog="{dialog_attr}">'
            f"<td data-sort='{float(payload['t']):.3f}'>{html.escape(payload['timestamp_label'])}</td>"
            f"<td data-sort='{float(payload['speed_kmh']):.3f}'>{float(payload['speed_kmh']):.1f}</td>"
            f"<td>{html.escape(_format_lat_lon(payload['lat'], payload['lon']))}</td>"
            f"<td>{html.escape(', '.join(payload['issues']) or 'none')}</td>"
            f"<td data-sort='{float(payload['crack_area_pct'] or 0.0):.6f}'>{float(payload['crack_area_pct'] or 0.0):.3f}%</td>"
            f"<td data-sort='{int(payload['crack_length_px'] or 0)}'>{int(payload['crack_length_px'] or 0)}</td>"
            f"<td data-sort='{payload['quality_grade'] if payload['quality_grade'] is not None else 99}'>{payload['quality_grade'] if payload['quality_grade'] is not None else 'n/a'}</td>"
            f"<td>{html.escape(str(payload['surface_type']))}</td>"
            f"<td>{thumb}<br>{filename_link}</td>"
            "</tr>"
        )
    body = "\n".join(rows) if rows else "<tr><td colspan='9'>No problem frames detected.</td></tr>"
    speed_label = html.escape(str(summary.get("speed_label", "")))
    return f"""<table id="{html.escape(table_id, quote=True)}">
      <thead>
        <tr>
          <th>timestamp</th>
          <th>speed {speed_label} km/h</th>
          <th>lat,lon</th>
          <th>issue(s)</th>
          <th>crack area</th>
          <th>crack length px</th>
          <th>quality</th>
          <th>surface_type</th>
          <th>thumbnail+link</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>"""


def _index_html(summary: dict[str, Any], samples: pd.DataFrame, problems: pd.DataFrame, *, out_dir: Path) -> str:
    stats = [
        ("Samples analyzed", summary.get("samples_analyzed", 0)),
        ("Frames in scatter", len(samples)),
        ("Problems saved", summary.get("problems_found", 0)),
        (f"Mean speed {summary.get('speed_label', '')} km/h", f"{float(summary.get('mean_speed_kmh', 0.0)):.1f}"),
        ("Confirmed cracks", summary.get("confirmed_crack_count", summary.get("crack_count_after_confirmation", 0))),
        ("GPS source", summary.get("gps_source", {}).get("type", "unknown")),
        ("Telemetry", summary.get("telemetry_parse", {}).get("status", "unknown")),
    ]
    stats_html = "\n".join(
        f'<div class="stat"><span>{html.escape(label)}</span><b>{html.escape(str(value))}</b></div>'
        for label, value in stats
    )
    run_name = html.escape(str(summary.get("run_name", "survey")))
    speed_warning = _speed_warning(summary)
    warning_html = f'<div class="banner">{html.escape(speed_warning)}</div>' if speed_warning else ""
    route_notice = html.escape(_route_notice(summary))
    scatter = _cluster_scatter_div(samples, out_dir=out_dir, include_plotlyjs=True, div_id="cluster-scatter")
    speed_chart = _speed_chart_div(samples, include_plotlyjs=False, div_id="speed-chart")
    table = _problem_table_markup(problems, summary, out_dir=out_dir, table_id="problem-table-index")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tarmac survey - {run_name}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #17202a; }}
    header {{ padding: 24px 32px; background: #17202a; color: white; }}
    header h1 {{ margin: 0 0 6px; font-size: 28px; }}
    header div {{ overflow-wrap: anywhere; }}
    main {{ max-width: 1280px; margin: 0 auto; padding: 24px; }}
    h2 {{ margin: 28px 0 12px; }}
    .banner {{ padding: 10px 12px; background: #fff3cd; border: 1px solid #e7d38a; margin: 16px 0; font-weight: 700; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 12px; margin: 18px 0; }}
    .stat {{ background: white; border: 1px solid #d8dee5; border-radius: 8px; padding: 12px; }}
    .stat span {{ display: block; color: #4b5563; font-size: 13px; margin-bottom: 5px; }}
    .stat b {{ display: block; font-size: 21px; }}
    .panel {{ background: white; border: 1px solid #d8dee5; border-radius: 8px; padding: 12px; margin: 12px 0 20px; overflow-x: auto; }}
    .links {{ display: flex; flex-wrap: wrap; gap: 12px; margin: 16px 0; }}
    a {{ color: #0f4c81; font-weight: 700; }}
    table {{ width: 100%; border-collapse: collapse; background: white; border: 1px solid #d8dee5; }}
    th, td {{ padding: 9px 10px; border-bottom: 1px solid #e4e9ee; text-align: left; vertical-align: top; }}
    th {{ background: #eef2f5; cursor: pointer; user-select: none; }}
    tr.tarmac-dialog-row {{ cursor: zoom-in; }}
    tr.tarmac-dialog-row:hover {{ background: #f8fafc; }}
    td img {{ width: 120px; max-height: 90px; object-fit: cover; border: 1px solid #d8dee5; display: block; }}
    .file-link {{ font-weight: 700; overflow-wrap: anywhere; }}
  </style>
  {dialog_css()}
</head>
<body>
  <header>
    <h1>Tarmac road survey</h1>
    <div>{html.escape(str(summary.get("input_path", run_name)))}</div>
  </header>
  <main>
    <div class="banner">{route_notice}</div>
    {warning_html}
    <section class="stats">{stats_html}</section>
    <nav class="links">
      <a href="map.html">Open map</a>
      <a href="cluster_scatter.html">Open cluster scatter</a>
      <a href="speed_chart.html">Open speed chart</a>
      <a href="problems_table.html">Open problem table</a>
      <a href="summary.json">Open summary JSON</a>
    </nav>
    <h2>All-frame cluster scatter</h2>
    <div class="panel">{scatter}</div>
    <h2>Speed over time</h2>
    <div class="panel">{speed_chart}</div>
    <h2>Problems</h2>
    <div class="panel">{table}</div>
  </main>
  {dialog_html(include_style=False)}
  <script>
    document.querySelectorAll('th').forEach((th, col) => {{
      th.addEventListener('click', () => {{
        const table = th.closest('table');
        const tbody = table.querySelector('tbody');
        const rows = Array.from(tbody.querySelectorAll('tr'));
        const asc = th.dataset.asc !== 'true';
        rows.sort((a, b) => {{
          const av = a.children[col]?.dataset.sort ?? a.children[col]?.textContent ?? '';
          const bv = b.children[col]?.dataset.sort ?? b.children[col]?.textContent ?? '';
          const an = Number(av);
          const bn = Number(bv);
          const cmp = Number.isFinite(an) && Number.isFinite(bn) ? an - bn : av.localeCompare(bv);
          return asc ? cmp : -cmp;
        }});
        th.dataset.asc = String(asc);
        rows.forEach(row => tbody.appendChild(row));
      }});
    }});
  </script>
  <script>{image_dialog_js(plot_id="cluster-scatter")}</script>
</body>
</html>
"""


def _cluster_scatter_page(samples: pd.DataFrame, summary: dict[str, Any], *, out_dir: Path) -> str:
    run_name = html.escape(str(summary.get("run_name", "survey")))
    scatter = _cluster_scatter_div(samples, out_dir=out_dir, include_plotlyjs=True, div_id="cluster-scatter")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tarmac survey cluster scatter - {run_name}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #17202a; }}
    main {{ padding: 20px 24px; }}
    .panel {{ background: white; border: 1px solid #d8dee5; border-radius: 8px; padding: 12px; }}
  </style>
  {dialog_css()}
</head>
<body>
  <main>
    <h1>All-frame cluster scatter</h1>
    <div class="panel">{scatter}</div>
  </main>
  {dialog_html(include_style=False)}
  <script>{image_dialog_js(plot_id="cluster-scatter")}</script>
</body>
</html>
"""


def _speed_chart_page(samples: pd.DataFrame, summary: dict[str, Any]) -> str:
    run_name = html.escape(str(summary.get("run_name", "survey")))
    speed_chart = _speed_chart_div(samples, include_plotlyjs=True, div_id="speed-chart")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tarmac survey speed chart - {run_name}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #17202a; }}
    main {{ padding: 20px 24px; }}
    .panel {{ background: white; border: 1px solid #d8dee5; border-radius: 8px; padding: 12px; }}
  </style>
</head>
<body>
  <main>
    <h1>Speed over time</h1>
    <div class="panel">{speed_chart}</div>
  </main>
</body>
</html>
"""


def _cluster_scatter_div(samples: pd.DataFrame, *, out_dir: Path, include_plotlyjs: bool, div_id: str) -> str:
    if samples.empty:
        return "<p>No sampled frames are available for the cluster scatter.</p>"
    try:
        projection, labels = _project_and_cluster(samples)
    except ValueError as exc:
        return f"<p>Cluster scatter unavailable: {html.escape(str(exc))}</p>"

    df = samples.reset_index(drop=True)
    customdata = _survey_customdata(df, out_dir=out_dir)
    hover_text = _survey_hover_text(df, labels)
    fig = go.Figure()
    cluster_ids = sorted(int(value) for value in np.unique(labels))
    for trace_index, cluster_id in enumerate(cluster_ids):
        mask = labels == cluster_id
        indexes = np.flatnonzero(mask)
        fig.add_trace(
            go.Scattergl(
                x=projection[mask, 0],
                y=projection[mask, 1],
                mode="markers",
                marker={
                    "size": 9,
                    "color": CLUSTER_COLORS[trace_index % len(CLUSTER_COLORS)],
                    "line": {"width": 0.5, "color": "#111827"},
                    "opacity": 0.82,
                },
                text=[hover_text[i] for i in indexes],
                customdata=[customdata[i] for i in indexes],
                hovertemplate="%{text}<extra></extra>",
                name=f"cluster {cluster_id}",
                visible=True,
            )
        )
    quality_colors = [
        QUALITY_COLORS.get(_maybe_int(getattr(row, "quality_grade", None)) or 0, "#9aa0a6")
        for row in df.itertuples()
    ]
    fig.add_trace(
        go.Scattergl(
            x=projection[:, 0],
            y=projection[:, 1],
            mode="markers",
            marker={
                "size": 9,
                "color": quality_colors,
                "line": {"width": 0.5, "color": "#111827"},
                "opacity": 0.82,
            },
            text=hover_text,
            customdata=customdata,
            hovertemplate="%{text}<extra></extra>",
            name="quality",
            visible=False,
        )
    )
    cluster_visible = [True] * len(cluster_ids) + [False]
    quality_visible = [False] * len(cluster_ids) + [True]
    fig.update_layout(
        title=f"All sampled frames ({len(df):,}) projected from DINOv3 embeddings",
        template="plotly_white",
        height=650,
        margin={"l": 42, "r": 24, "t": 64, "b": 42},
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "left", "x": 0},
        updatemenus=[
            {
                "type": "buttons",
                "direction": "right",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.18,
                "buttons": [
                    {
                        "label": "Color: cluster",
                        "method": "update",
                        "args": [{"visible": cluster_visible}, {"showlegend": True}],
                    },
                    {
                        "label": "Color: quality",
                        "method": "update",
                        "args": [{"visible": quality_visible}, {"showlegend": False}],
                    },
                ],
            }
        ],
    )
    fig.update_xaxes(title="UMAP 1", zeroline=False)
    fig.update_yaxes(title="UMAP 2", zeroline=False)
    return pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=False, div_id=div_id)


def _speed_chart_div(samples: pd.DataFrame, *, include_plotlyjs: bool, div_id: str) -> str:
    if samples.empty or "speed_kmh" not in samples:
        return "<p>No speed samples are available.</p>"
    df = samples.sort_values("t").reset_index(drop=True)
    x_minutes = df["t"].astype(float) / 60.0
    speed = df["speed_kmh"].astype(float)
    mean_speed = float(speed.mean()) if len(speed) else 0.0
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_minutes,
            y=speed,
            mode="lines+markers",
            line={"color": "#2563eb", "width": 2},
            marker={"size": 5, "color": "#0f766e"},
            text=[
                f"{_timestamp_label(float(row.t))}<br>speed={float(row.speed_kmh):.1f} km/h<br>quality={_maybe_int(getattr(row, 'quality_grade', None)) or 'n/a'}"
                for row in df.itertuples()
            ],
            hovertemplate="%{text}<extra></extra>",
            name="speed",
        )
    )
    fig.add_hline(
        y=mean_speed,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"mean {mean_speed:.1f} km/h",
        annotation_position="top left",
    )
    fig.update_layout(
        title="Driver speed over sampled frames",
        template="plotly_white",
        height=360,
        margin={"l": 46, "r": 24, "t": 55, "b": 46},
        showlegend=False,
    )
    fig.update_xaxes(title="Time (min)")
    fig.update_yaxes(title="Speed (km/h)", rangemode="tozero")
    return pio.to_html(fig, include_plotlyjs=include_plotlyjs, full_html=False, div_id=div_id)


def _project_and_cluster(samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    matrix = _embedding_matrix(samples)
    if matrix is None:
        raise ValueError("samples.parquet does not contain a valid embedding for every sampled frame")
    n_samples = matrix.shape[0]
    if n_samples == 1:
        return np.zeros((1, 2), dtype=np.float32), np.zeros(1, dtype=np.int32)
    if n_samples == 2:
        projection = np.array([[-0.5, 0.0], [0.5, 0.0]], dtype=np.float32)
    else:
        n_neighbors = min(15, max(2, n_samples - 1))
        reducer = umap.UMAP(
            n_components=2,
            n_neighbors=n_neighbors,
            min_dist=0.08,
            metric="cosine",
            random_state=SEED,
        )
        projection = reducer.fit_transform(matrix).astype(np.float32)
    cluster_count = _cluster_count(n_samples)
    if cluster_count <= 1:
        labels = np.zeros(n_samples, dtype=np.int32)
    else:
        labels = KMeans(n_clusters=cluster_count, random_state=SEED, n_init=10).fit_predict(
            _normalize_rows(matrix)
        )
    return projection, labels.astype(np.int32)


def _embedding_matrix(samples: pd.DataFrame) -> np.ndarray | None:
    if "embedding" not in samples:
        return None
    arrays: list[np.ndarray] = []
    expected_dim: int | None = None
    for value in samples["embedding"].tolist():
        array = np.asarray(value, dtype=np.float32).reshape(-1)
        if array.size == 0:
            return None
        expected_dim = expected_dim or int(array.size)
        if int(array.size) != expected_dim:
            return None
        arrays.append(array)
    if len(arrays) != len(samples):
        return None
    return np.vstack(arrays).astype(np.float32)


def _cluster_count(n_samples: int) -> int:
    if n_samples <= 1:
        return 1
    if n_samples <= 3:
        return min(2, n_samples)
    return min(n_samples, max(2, min(12, int(round(math.sqrt(n_samples / 2.0))))))


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (matrix / norms).astype(np.float32)


def _survey_customdata(samples: pd.DataFrame, *, out_dir: Path) -> list[list[Any]]:
    return [_dialog_customdata(_frame_payload(row, out_dir=out_dir)) for row in samples.itertuples()]


def _survey_hover_text(samples: pd.DataFrame, labels: np.ndarray) -> list[str]:
    text: list[str] = []
    for index, row in enumerate(samples.itertuples()):
        issues = ", ".join(_json_list(getattr(row, "issues", "[]"))) or "none"
        text.append(
            f"{html.escape(_timestamp_label(float(getattr(row, 't', 0.0))))}"
            f"<br>cluster={int(labels[index])}"
            f"<br>quality={_maybe_int(getattr(row, 'quality_grade', None)) or 'n/a'}"
            f"<br>speed={float(getattr(row, 'speed_kmh', 0.0) or 0.0):.1f} km/h"
            f"<br>issues={html.escape(issues)}"
        )
    return text


def _problem_payload(row: Any, *, out_dir: Path) -> dict[str, Any]:
    return _frame_payload(row, out_dir=out_dir)


def _frame_payload(row: Any, *, out_dir: Path) -> dict[str, Any]:
    issues = _json_list(getattr(row, "issues", "[]"))
    problem_path = str(getattr(row, "problem_image", "") or "")
    frame_path = str(getattr(row, "frame_image", "") or getattr(row, "frame_thumbnail", "") or "")
    thumbnail_path = str(getattr(row, "thumbnail_image", "") or frame_path or "")
    original_path = problem_path or frame_path
    marked_path = str(getattr(row, "crack_overlay_image", "") or getattr(row, "crackseg_overlay_path", "") or "")
    original_abs = _resolve_survey_path(out_dir, original_path)
    thumbnail_abs = _resolve_survey_path(out_dir, thumbnail_path) if thumbnail_path else None
    marked_abs = _resolve_survey_path(out_dir, marked_path) if marked_path else None
    filename = original_abs.name if original_abs is not None else Path(original_path).name
    speed_kmh = _maybe_float(getattr(row, "speed_kmh", None)) or 0.0
    confidence = _maybe_float(getattr(row, "confidence", None))
    crack_area_pct = _maybe_float(getattr(row, "crack_area_pct", None)) or 0.0
    crack_length_px = _maybe_int(getattr(row, "crack_length_px", None)) or 0
    crack_confirmed = bool(getattr(row, "crack_confirmed", False))
    return {
        "t": float(getattr(row, "t", 0.0)),
        "timestamp_label": _timestamp_label(float(getattr(row, "t", 0.0))),
        "speed_kmh": speed_kmh,
        "lat": _json_float_or_none(getattr(row, "lat", None)),
        "lon": _json_float_or_none(getattr(row, "lon", None)),
        "issues": issues,
        "quality_grade": _maybe_int(getattr(row, "quality_grade", None)),
        "surface_type": str(getattr(row, "surface_type", "unknown") or "unknown"),
        "confidence": confidence,
        "image_path": original_path,
        "problem_image": problem_path,
        "frame_image": frame_path,
        "thumbnail_path": thumbnail_path,
        "marked_path": marked_path,
        "image_abs_path": str(original_abs) if original_abs is not None else original_path,
        "original_abs_path": str(original_abs) if original_abs is not None else original_path,
        "thumbnail_abs_path": str(thumbnail_abs) if thumbnail_abs is not None else thumbnail_path,
        "marked_abs_path": str(marked_abs) if marked_abs is not None else marked_path,
        "image_file_url": absolute_file_url(str(original_abs)) if original_abs is not None else "",
        "original_file_url": absolute_file_url(str(original_abs)) if original_abs is not None else "",
        "thumbnail_file_url": absolute_file_url(str(thumbnail_abs)) if thumbnail_abs is not None else "",
        "marked_file_url": absolute_file_url(str(marked_abs)) if marked_abs is not None else "",
        "filename": filename,
        "crack_detected": crack_confirmed,
        "crack_confirmed": crack_confirmed,
        "crack_area_pct": crack_area_pct,
        "crack_length_px": crack_length_px,
        "crack_max_component_length_px": _maybe_int(getattr(row, "crack_max_component_length_px", None)),
        "crack_confirmation_reason": str(getattr(row, "crack_confirmation_reason", "") or ""),
        "crack_segmenter": str(getattr(row, "crack_segmenter", "") or ""),
    }


def _dialog_payload(payload: dict[str, Any]) -> dict[str, Any]:
    lat = payload.get("lat")
    lon = payload.get("lon")
    return {
        "kind": "survey",
        "src": payload.get("original_file_url", ""),
        "fileUrl": payload.get("original_file_url", ""),
        "path": payload.get("original_abs_path", ""),
        "filename": payload.get("filename", ""),
        "markedSrc": payload.get("marked_file_url", ""),
        "markedPath": payload.get("marked_abs_path", ""),
        "quality": str(payload.get("quality_grade") if payload.get("quality_grade") is not None else ""),
        "surface": payload.get("surface_type", ""),
        "confidence": _format_optional_float(payload.get("confidence"), digits=3),
        "timestamp": payload.get("timestamp_label", ""),
        "speedKmh": f"{float(payload.get('speed_kmh') or 0.0):.1f}",
        "lat": _format_optional_float(lat, digits=6),
        "lon": _format_optional_float(lon, digits=6),
        "issues": ", ".join(payload.get("issues", [])) or "none",
        "crackAreaPct": f"{float(payload.get('crack_area_pct') or 0.0):.3f}",
        "crackLengthPx": str(int(payload.get("crack_length_px") or 0)),
        "crackDetected": bool(payload.get("crack_confirmed", False)),
        "crackSegmenter": payload.get("crack_segmenter", ""),
        "noCracksNote": "no cracks detected",
    }


def _dialog_customdata(payload: dict[str, Any]) -> list[Any]:
    dialog = _dialog_payload(payload)
    return [
        "survey",
        dialog["path"],
        dialog["src"],
        dialog["surface"],
        dialog["quality"],
        dialog["confidence"],
        dialog["filename"],
        dialog["timestamp"],
        dialog["fileUrl"],
        dialog["markedSrc"],
        dialog["speedKmh"],
        dialog["lat"],
        dialog["lon"],
        dialog["issues"],
        dialog["crackAreaPct"],
        dialog["crackLengthPx"],
        dialog["crackDetected"],
        dialog["crackSegmenter"],
        dialog["noCracksNote"],
    ]


def _format_optional_float(value: Any, *, digits: int) -> str:
    number = _maybe_float(value)
    if number is None:
        return ""
    return f"{number:.{digits}f}"


def _resolve_survey_path(out_dir: Path, value: str) -> Path | None:
    if not value:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = out_dir / path
    return path.resolve()


def _start_point(start: Any, route_points: list[dict[str, Any]]) -> dict[str, Any] | None:
    start = start if isinstance(start, dict) else {}
    lat = _maybe_float(start.get("lat"))
    lon = _maybe_float(start.get("lon"))
    if lat is None or lon is None:
        if route_points:
            lat = _maybe_float(route_points[0].get("lat"))
            lon = _maybe_float(route_points[0].get("lon"))
        if lat is None or lon is None:
            return None
    return {
        "lat": lat,
        "lon": lon,
        "alt_m": start.get("alt_m"),
        "source": start.get("source", "unknown"),
    }


def _route_notice(summary: dict[str, Any]) -> str:
    notice = summary.get("route_notice")
    if notice:
        return str(notice)
    source_type = str(summary.get("gps_source", {}).get("type", ""))
    if source_type == "sidecar":
        return "GPS route from sidecar track."
    if source_type == "embedded_video":
        return "GPS route from embedded timed metadata."
    if source_type == "none":
        return "No GPS track was found; map route is omitted."
    return ROUTE_NOTICE


def _valid_lat_lon(lat: Any, lon: Any) -> bool:
    return _maybe_float(lat) is not None and _maybe_float(lon) is not None


def _json_float_or_none(value: Any) -> float | None:
    number = _maybe_float(value)
    return float(number) if number is not None else None


def _format_lat_lon(lat: Any, lon: Any) -> str:
    lat_value = _maybe_float(lat)
    lon_value = _maybe_float(lon)
    if lat_value is None or lon_value is None:
        return "n/a"
    return f"{lat_value:.6f}, {lon_value:.6f}"


def _image_link(
    *,
    file_url: str,
    file_path: str,
    filename: str,
    inner: str,
    quality: str = "",
    surface: str = "",
    confidence: str = "",
    css_class: str = "",
    dialog_payload: dict[str, Any] | None = None,
) -> str:
    classes = " ".join(part for part in ["tarmac-image-link", css_class] if part)
    attrs = {
        "class": classes,
        "href": file_url,
        "data-src": file_url,
        "data-path": file_path,
        "data-filename": filename,
        "data-quality": quality,
        "data-surface": surface,
        "data-confidence": confidence,
    }
    if dialog_payload is not None:
        attrs["data-dialog"] = json.dumps(dialog_payload, separators=(",", ":"), allow_nan=False)
    attr_text = " ".join(f'{key}="{html.escape(str(value), quote=True)}"' for key, value in attrs.items())
    return f"<a {attr_text}>{inner}</a>"


def _speed_warning(summary: dict[str, Any]) -> str | None:
    warning = summary.get("speed_warning")
    if warning:
        return str(warning)
    if str(summary.get("gps_source", {}).get("type", "")) != "imu_deadreckon":
        return None
    try:
        mean_speed = float(summary.get("mean_speed_kmh", 0.0))
    except (TypeError, ValueError):
        return None
    if int(summary.get("samples_analyzed", 0) or 0) > 1 and mean_speed < 5.0:
        return (
            "Mean IMU-estimated speed is below 5 km/h for this moving survey; "
            "treat speed and distance as unreliable."
        )
    return None


def _timestamp_label(seconds: float) -> str:
    minutes = int(seconds // 60)
    remainder = seconds - minutes * 60
    return f"{minutes:02d}:{remainder:05.2f}"


def _json_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item) for item in value]
    if not value:
        return []
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [str(value)]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def _maybe_int(value: Any) -> int | None:
    if value is None or pd.isna(value):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _maybe_float(value: Any) -> float | None:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
