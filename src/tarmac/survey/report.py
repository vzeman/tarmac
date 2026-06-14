from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tarmac.survey.telemetry import ROUTE_NOTICE

QUALITY_COLORS = {
    1: "#1a9850",
    2: "#91cf60",
    3: "#fee08b",
    4: "#fc8d59",
    5: "#d73027",
}


def build_reports(out_dir: Path) -> dict[str, Path]:
    out_dir = out_dir.expanduser().resolve()
    samples = pd.read_parquet(out_dir / "samples.parquet")
    problems = pd.read_parquet(out_dir / "problems.parquet")
    telemetry = pd.read_parquet(out_dir / "telemetry.parquet")
    summary = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))

    map_path = out_dir / "map.html"
    table_path = out_dir / "problems_table.html"
    index_path = out_dir / "index.html"
    map_path.write_text(_map_html(samples, problems, telemetry, summary), encoding="utf-8")
    table_path.write_text(_table_html(problems, summary), encoding="utf-8")
    index_path.write_text(_index_html(summary), encoding="utf-8")
    return {"map_html": map_path, "problems_table_html": table_path, "index_html": index_path}


def _map_html(
    samples: pd.DataFrame,
    problems: pd.DataFrame,
    telemetry: pd.DataFrame,
    summary: dict[str, Any],
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
        if pd.notna(getattr(row, "lat", None)) and pd.notna(getattr(row, "lon", None))
    ]
    telemetry_points = [
        [float(row.lat), float(row.lon)]
        for row in telemetry.itertuples()
        if pd.notna(getattr(row, "lat", None)) and pd.notna(getattr(row, "lon", None))
    ]
    start = summary.get("start_location", {})
    start_point = {
        "lat": float(start.get("lat", route_points[0]["lat"] if route_points else 0.0)),
        "lon": float(start.get("lon", route_points[0]["lon"] if route_points else 0.0)),
        "alt_m": start.get("alt_m"),
        "source": start.get("source", "unknown"),
    }
    problem_points = [_problem_payload(row) for row in problems.itertuples()]
    data = {
        "route": route_points,
        "telemetryRoute": telemetry_points[:: max(1, len(telemetry_points) // 2000)] if telemetry_points else [],
        "problems": problem_points,
        "start": start_point,
        "qualityColors": QUALITY_COLORS,
        "notice": "Route is IMU-estimated (approximate, drifts) — no continuous GPS in source.",
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
  </style>
</head>
<body>
  <div class="banner">Route is IMU-estimated (approximate, drifts) — no continuous GPS in source.</div>
  <div id="map"></div>
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

    const startMarker = L.marker([data.start.lat, data.start.lon], {{
      title: 'START'
    }}).bindPopup(`<b>START</b><br>${{data.start.lat.toFixed(6)}}, ${{data.start.lon.toFixed(6)}}<br>${{data.start.source}}`);
    startMarker.addTo(map);

    const problemLayer = L.layerGroup();
    for (const p of data.problems) {{
      const marker = L.circleMarker([p.lat, p.lon], {{
        radius: 7,
        color: '#9f1239',
        weight: 2,
        fillColor: '#e11d48',
        fillOpacity: 0.85
      }});
      const image = p.thumbnail_path ? `<a href="${{p.image_path}}"><img class="popup-thumb" src="${{p.thumbnail_path}}" alt="problem thumbnail"></a>` : '';
      marker.bindPopup(`
        <div class="popup-title">${{p.timestamp_label}} · quality ${{p.quality_grade ?? 'n/a'}}</div>
        <div><b>Speed:</b> ${{p.speed_kmh.toFixed(1)}} km/h</div>
        <div><b>Location:</b> ${{p.lat.toFixed(6)}}, ${{p.lon.toFixed(6)}}</div>
        <div><b>Issues:</b> ${{p.issues.join(', ') || 'none'}}</div>
        ${{image}}
        <a href="${{p.image_path}}">Open full image</a>
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
    }} else {{
      map.setView([data.start.lat, data.start.lon], 16);
    }}

    const legend = L.control({{ position: 'bottomright' }});
    legend.onAdd = function() {{
      const div = L.DomUtil.create('div', 'legend');
      div.innerHTML = '<b>Quality grade</b><br>' +
        [1,2,3,4,5].map(q => `<span class="swatch" style="background:${{data.qualityColors[q]}}"></span>${{q}}`).join('<br>') +
        '<hr><b>Route caveat</b><br>IMU-estimated, approximate, drifting';
      return div;
    }};
    legend.addTo(map);
  </script>
</body>
</html>
"""


def _table_html(problems: pd.DataFrame, summary: dict[str, Any]) -> str:
    rows: list[str] = []
    for row in problems.itertuples():
        payload = _problem_payload(row)
        thumb = ""
        if payload["thumbnail_path"]:
            thumb = (
                f'<a href="{html.escape(payload["image_path"])}">'
                f'<img src="{html.escape(payload["thumbnail_path"])}" alt="problem thumbnail"></a>'
            )
        rows.append(
            "<tr>"
            f"<td data-sort='{float(payload['t']):.3f}'>{html.escape(payload['timestamp_label'])}</td>"
            f"<td data-sort='{float(payload['speed_kmh']):.3f}'>{float(payload['speed_kmh']):.1f}</td>"
            f"<td>{payload['lat']:.6f}, {payload['lon']:.6f}</td>"
            f"<td>{html.escape(', '.join(payload['issues']) or 'none')}</td>"
            f"<td data-sort='{payload['quality_grade'] if payload['quality_grade'] is not None else 99}'>{payload['quality_grade'] if payload['quality_grade'] is not None else 'n/a'}</td>"
            f"<td>{html.escape(str(payload['surface_type']))}</td>"
            f"<td>{thumb}<br><a href='{html.escape(payload['image_path'])}'>full image</a></td>"
            "</tr>"
        )
    body = "\n".join(rows) if rows else "<tr><td colspan='7'>No problem frames detected.</td></tr>"
    title = html.escape(f"Tarmac survey problem table - {summary.get('run_name', 'survey')}")
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
    img {{ width: 120px; max-height: 90px; object-fit: cover; border: 1px solid #d8dee5; display: block; }}
    a {{ color: #0f4c81; }}
  </style>
</head>
<body>
  <header><h1>Problem table</h1></header>
  <main>
    <div class="banner">Route is IMU-estimated (approximate, drifts) — no continuous GPS in source.</div>
    <table id="problem-table">
      <thead>
        <tr>
          <th>timestamp</th>
          <th>speed_kmh</th>
          <th>lat,lon</th>
          <th>issue(s)</th>
          <th>quality</th>
          <th>surface_type</th>
          <th>thumbnail+link</th>
        </tr>
      </thead>
      <tbody>{body}</tbody>
    </table>
  </main>
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
</body>
</html>
"""


def _index_html(summary: dict[str, Any]) -> str:
    stats = [
        ("Samples analyzed", summary.get("samples_analyzed", 0)),
        ("Problems saved", summary.get("problems_found", 0)),
        ("Mean speed km/h", f"{float(summary.get('mean_speed_kmh', 0.0)):.1f}"),
        ("Telemetry", summary.get("telemetry_parse", {}).get("status", "unknown")),
    ]
    stats_html = "\n".join(
        f"<li><b>{html.escape(label)}:</b> {html.escape(str(value))}</li>" for label, value in stats
    )
    run_name = html.escape(str(summary.get("run_name", "survey")))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tarmac survey - {run_name}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f7f9; color: #17202a; }}
    main {{ max-width: 780px; margin: 0 auto; padding: 28px 20px; }}
    .banner {{ padding: 10px 12px; background: #fff3cd; border: 1px solid #e7d38a; margin: 16px 0; font-weight: 700; }}
    a {{ color: #0f4c81; font-weight: 700; }}
  </style>
</head>
<body>
  <main>
    <h1>Tarmac road survey</h1>
    <div class="banner">Route is IMU-estimated (approximate, drifts) — no continuous GPS in source.</div>
    <ul>{stats_html}</ul>
    <p><a href="map.html">Open map</a></p>
    <p><a href="problems_table.html">Open problem table</a></p>
    <p><a href="summary.json">Open summary JSON</a></p>
  </main>
</body>
</html>
"""


def _problem_payload(row: Any) -> dict[str, Any]:
    issues = _json_list(getattr(row, "issues", "[]"))
    image_path = str(getattr(row, "problem_image", "") or "")
    thumbnail_path = str(getattr(row, "thumbnail_image", "") or "")
    return {
        "t": float(getattr(row, "t", 0.0)),
        "timestamp_label": _timestamp_label(float(getattr(row, "t", 0.0))),
        "speed_kmh": float(getattr(row, "speed_kmh", 0.0) or 0.0),
        "lat": float(getattr(row, "lat", 0.0) or 0.0),
        "lon": float(getattr(row, "lon", 0.0) or 0.0),
        "issues": issues,
        "quality_grade": _maybe_int(getattr(row, "quality_grade", None)),
        "surface_type": str(getattr(row, "surface_type", "unknown") or "unknown"),
        "image_path": image_path,
        "thumbnail_path": thumbnail_path,
    }


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
