from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from tarmac.report.umap_html import absolute_file_url, dialog_css, dialog_html, image_dialog_js
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
    index_path = out_dir / "index.html"
    map_path.write_text(_map_html(samples, problems, telemetry, summary, out_dir=out_dir), encoding="utf-8")
    table_path.write_text(_table_html(problems, summary, out_dir=out_dir), encoding="utf-8")
    index_path.write_text(_index_html(summary), encoding="utf-8")
    return {"map_html": map_path, "problems_table_html": table_path, "index_html": index_path}


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
    problem_points = [_problem_payload(row, out_dir=out_dir) for row in problems.itertuples()]
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
    function tarmacPopupImageLink(p, innerHtml) {{
      const fileUrl = tarmacEscapeHtml(p.image_file_url || p.image_path || '');
      const filename = tarmacEscapeHtml(p.filename || p.image_path || '');
      const quality = tarmacEscapeHtml(p.quality_grade ?? '');
      const surface = tarmacEscapeHtml(p.surface_type || '');
      const confidence = tarmacEscapeHtml(p.confidence ?? '');
      const path = tarmacEscapeHtml(p.image_abs_path || p.image_path || '');
      return `<a class="tarmac-image-link" href="${{fileUrl}}" data-src="${{fileUrl}}" data-path="${{path}}" data-filename="${{filename}}" data-quality="${{quality}}" data-surface="${{surface}}" data-confidence="${{confidence}}">${{innerHtml}}</a>`;
    }}
  </script>
  <script>{image_dialog_js(plot_id=None)}</script>
</body>
</html>
"""


def _table_html(problems: pd.DataFrame, summary: dict[str, Any], *, out_dir: Path) -> str:
    rows: list[str] = []
    for row in problems.itertuples():
        payload = _problem_payload(row, out_dir=out_dir)
        thumb = ""
        filename_link = _image_link(
            file_url=payload["image_file_url"],
            file_path=payload["image_abs_path"],
            filename=payload["filename"],
            inner=html.escape(payload["filename"]),
            quality=str(payload["quality_grade"] or ""),
            surface=payload["surface_type"],
            confidence=str(payload.get("confidence") or ""),
            css_class="file-link",
        )
        if payload["thumbnail_file_url"]:
            thumb = (
                _image_link(
                    file_url=payload["image_file_url"],
                    file_path=payload["image_abs_path"],
                    filename=payload["filename"],
                    inner=(
                        f'<img src="{html.escape(payload["thumbnail_file_url"], quote=True)}" '
                        'alt="problem thumbnail">'
                    ),
                    quality=str(payload["quality_grade"] or ""),
                    surface=payload["surface_type"],
                    confidence=str(payload.get("confidence") or ""),
                )
            )
        rows.append(
            "<tr>"
            f"<td data-sort='{float(payload['t']):.3f}'>{html.escape(payload['timestamp_label'])}</td>"
            f"<td data-sort='{float(payload['speed_kmh']):.3f}'>{float(payload['speed_kmh']):.1f}</td>"
            f"<td>{html.escape(_format_lat_lon(payload['lat'], payload['lon']))}</td>"
            f"<td>{html.escape(', '.join(payload['issues']) or 'none')}</td>"
            f"<td data-sort='{payload['quality_grade'] if payload['quality_grade'] is not None else 99}'>{payload['quality_grade'] if payload['quality_grade'] is not None else 'n/a'}</td>"
            f"<td>{html.escape(str(payload['surface_type']))}</td>"
            f"<td>{thumb}<br>{filename_link}</td>"
            "</tr>"
        )
    body = "\n".join(rows) if rows else "<tr><td colspan='7'>No problem frames detected.</td></tr>"
    title = html.escape(f"Tarmac survey problem table - {summary.get('run_name', 'survey')}")
    speed_warning = _speed_warning(summary)
    route_notice = _route_notice(summary)
    speed_label = html.escape(str(summary.get("speed_label", "")))
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
    .file-link {{ font-weight: 700; overflow-wrap: anywhere; }}
  </style>
  {dialog_css()}
</head>
<body>
  <header><h1>Problem table</h1></header>
  <main>
    <div class="banner">{html.escape(route_notice)}{f" {html.escape(speed_warning)}" if speed_warning else ""}</div>
    <table id="problem-table">
      <thead>
        <tr>
          <th>timestamp</th>
          <th>speed {speed_label} km/h</th>
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


def _index_html(summary: dict[str, Any]) -> str:
    stats = [
        ("Samples analyzed", summary.get("samples_analyzed", 0)),
        ("Problems saved", summary.get("problems_found", 0)),
        (f"Mean speed {summary.get('speed_label', '')} km/h", f"{float(summary.get('mean_speed_kmh', 0.0)):.1f}"),
        ("Confirmed cracks", summary.get("confirmed_crack_count", summary.get("crack_count_after_confirmation", 0))),
        ("GPS source", summary.get("gps_source", {}).get("type", "unknown")),
        ("Telemetry", summary.get("telemetry_parse", {}).get("status", "unknown")),
    ]
    stats_html = "\n".join(
        f"<li><b>{html.escape(label)}:</b> {html.escape(str(value))}</li>" for label, value in stats
    )
    run_name = html.escape(str(summary.get("run_name", "survey")))
    speed_warning = _speed_warning(summary)
    warning_html = f'<div class="banner">{html.escape(speed_warning)}</div>' if speed_warning else ""
    route_notice = html.escape(_route_notice(summary))
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
    <div class="banner">{route_notice}</div>
    {warning_html}
    <ul>{stats_html}</ul>
    <p><a href="map.html">Open map</a></p>
    <p><a href="problems_table.html">Open problem table</a></p>
    <p><a href="summary.json">Open summary JSON</a></p>
  </main>
</body>
</html>
"""


def _problem_payload(row: Any, *, out_dir: Path) -> dict[str, Any]:
    issues = _json_list(getattr(row, "issues", "[]"))
    image_path = str(getattr(row, "problem_image", "") or "")
    thumbnail_path = str(getattr(row, "thumbnail_image", "") or "")
    image_abs = _resolve_survey_path(out_dir, image_path)
    thumbnail_abs = _resolve_survey_path(out_dir, thumbnail_path) if thumbnail_path else None
    filename = image_abs.name if image_abs is not None else Path(image_path).name
    return {
        "t": float(getattr(row, "t", 0.0)),
        "timestamp_label": _timestamp_label(float(getattr(row, "t", 0.0))),
        "speed_kmh": float(getattr(row, "speed_kmh", 0.0) or 0.0),
        "lat": _json_float_or_none(getattr(row, "lat", None)),
        "lon": _json_float_or_none(getattr(row, "lon", None)),
        "issues": issues,
        "quality_grade": _maybe_int(getattr(row, "quality_grade", None)),
        "surface_type": str(getattr(row, "surface_type", "unknown") or "unknown"),
        "confidence": _maybe_float(getattr(row, "confidence", None)),
        "image_path": image_path,
        "thumbnail_path": thumbnail_path,
        "image_abs_path": str(image_abs) if image_abs is not None else image_path,
        "thumbnail_abs_path": str(thumbnail_abs) if thumbnail_abs is not None else thumbnail_path,
        "image_file_url": absolute_file_url(str(image_abs)) if image_abs is not None else "",
        "thumbnail_file_url": absolute_file_url(str(thumbnail_abs)) if thumbnail_abs is not None else "",
        "filename": filename,
        "crack_confirmed": bool(getattr(row, "crack_confirmed", False)),
        "crack_area_pct": _maybe_float(getattr(row, "crack_area_pct", None)),
        "crack_max_component_length_px": _maybe_int(getattr(row, "crack_max_component_length_px", None)),
        "crack_confirmation_reason": str(getattr(row, "crack_confirmation_reason", "") or ""),
    }


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
