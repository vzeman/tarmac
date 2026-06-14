from __future__ import annotations

import json
import html as html_lib
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import umap

from tarmac.defect import DEFECT_LABELS
from tarmac.inference.analyze import image_to_base64, load_active_artifacts

QUALITY_COLORS = {
    1: "#1a9850",
    2: "#91cf60",
    3: "#fee08b",
    4: "#fc8d59",
    5: "#d73027",
}


def build_html_report(run_dir: Path, output: Path | None = None) -> Path:
    run_dir = run_dir.expanduser().resolve()
    results_path = run_dir / "results.parquet"
    summary_path = run_dir / "summary.json"
    if not results_path.exists():
        raise FileNotFoundError(f"Missing analysis results: {results_path}")
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing analysis summary: {summary_path}")

    df = pd.read_parquet(results_path)
    summary = json.loads(summary_path.read_text())
    artifacts = load_active_artifacts()
    ref_df = pd.read_parquet(artifacts.embeddings_path)
    ref_df = ref_df[ref_df["kind"] == "full"].reset_index(drop=True)
    ref_embeddings = np.vstack(ref_df["embedding"].to_numpy()).astype("float32")
    run_embeddings = np.vstack(df["embedding"].to_numpy()).astype("float32")
    reducer_path = Path("models/umap_reducer.pkl")
    reducer = load_or_fit_umap(reducer_path, ref_embeddings)
    ref_xy = reducer.embedding_
    run_xy = reducer.transform(run_embeddings)

    timeline = quality_timeline(df)
    condition_panel = condition_assessment_panel(run_dir)
    crack_panel = cracked_sections_panel(run_dir, df)
    structural_panel = structural_defects_panel(df)
    crack_geometry = crack_geometry_panel(run_dir, df)
    scatter = umap_scatter(ref_xy, ref_df, run_xy, df)
    gps = gps_scatter(df) if {"latitude", "longitude"}.issubset(df.columns) else ""
    gallery = gallery_html(run_dir, df)
    stats = headline_stats(summary)
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Tarmac Report - {run_dir.name}</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; background: #f6f7f9; }}
    header {{ padding: 28px 32px; background: #17202a; color: white; }}
    main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
    h1 {{ margin: 0 0 6px; font-size: 30px; }}
    h2 {{ margin-top: 32px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 12px; margin: 20px 0; }}
    .stat {{ background: white; border: 1px solid #dde2e7; border-radius: 8px; padding: 14px; }}
    .stat b {{ display: block; font-size: 24px; }}
    .panel {{ background: white; border: 1px solid #dde2e7; border-radius: 8px; padding: 12px; margin: 16px 0; }}
    .gallery {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(170px, 1fr)); gap: 14px; }}
    .thumb {{ background: white; border: 1px solid #dde2e7; border-radius: 8px; overflow: hidden; }}
    .thumb img {{ width: 100%; max-height: 200px; object-fit: cover; display: block; }}
    .thumb div {{ padding: 9px; font-size: 13px; }}
    .badge {{ display: inline-block; color: #111; border-radius: 999px; padding: 2px 8px; font-weight: 700; }}
    .priority-badge {{ display: inline-block; border-radius: 999px; padding: 3px 9px; font-weight: 700; font-size: 12px; color: #111; }}
    .priority-none {{ background: #d7f0df; }}
    .priority-monitor {{ background: #d9e8ff; }}
    .priority-plan-repair {{ background: #ffe2aa; }}
    .priority-urgent {{ background: #ffb8b8; }}
    .crack-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); gap: 14px; }}
    .crack-card {{ background: white; border: 1px solid #dde2e7; border-radius: 8px; overflow: hidden; }}
    .crack-image {{ position: relative; line-height: 0; }}
    .crack-image img {{ width: 100%; display: block; }}
    .crack-tile {{ position: absolute; box-sizing: border-box; border: 1px solid rgba(255,255,255,0.75); }}
    .crack-tile.hot {{ background: rgba(210, 36, 36, 0.45); border-color: rgba(210, 36, 36, 0.9); }}
    .crack-meta {{ padding: 9px; font-size: 13px; line-height: 1.35; }}
    .measure-table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    .measure-table th, .measure-table td {{ border-bottom: 1px solid #dde2e7; padding: 8px; text-align: left; }}
    .measure-table th {{ background: #f1f4f7; }}
    .defect-tag {{ display: inline-block; margin: 2px 4px 2px 0; padding: 2px 8px; border-radius: 999px; background: #e8eef5; font-weight: 700; font-size: 12px; }}
  </style>
</head>
<body>
  <header>
    <h1>Tarmac Analysis Report</h1>
    <div>{summary.get("input_path", run_dir.name)}</div>
  </header>
  <main>
    {stats}
    {condition_panel}
    <h2>Quality Timeline</h2>
    <div class="panel">{timeline}</div>
    {crack_panel}
    {structural_panel}
    {crack_geometry}
    <h2>Embedding Map</h2>
    <div class="panel">{scatter}</div>
    {gps}
    <h2>Per-frame Gallery</h2>
    <div class="gallery">{gallery}</div>
  </main>
</body>
</html>
"""
    output = output or (run_dir / "report.html")
    output.write_text(html)
    return output


def load_or_fit_umap(path: Path, reference_embeddings: np.ndarray) -> Any:
    if path.exists():
        return joblib.load(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    reducer = umap.UMAP(n_neighbors=20, min_dist=0.05, metric="cosine", random_state=42)
    reducer.fit(reference_embeddings)
    joblib.dump(reducer, path)
    return reducer


def headline_stats(summary: dict[str, Any]) -> str:
    distribution = summary.get("quality_distribution", {})
    dist_text = ", ".join(f"{k}: {v}" for k, v in distribution.items()) or "n/a"
    return f"""
    <section class="stats">
      <div class="stat"><span>Frames analyzed</span><b>{summary.get("frames_analyzed", 0)}</b></div>
      <div class="stat"><span>Dominant surface</span><b>{summary.get("dominant_surface_type", "unknown")}</b></div>
      <div class="stat"><span>Mean confidence</span><b>{float(summary.get("mean_confidence", 0.0)):.3f}</b></div>
      <div class="stat"><span>Quality distribution</span><b>{dist_text}</b></div>
    </section>
"""


def condition_assessment_panel(run_dir: Path) -> str:
    assessment_path = run_dir / "assessment.parquet"
    if not assessment_path.exists():
        return ""
    assessment = pd.read_parquet(assessment_path)
    if assessment.empty:
        return ""
    counts = assessment["repair_priority"].fillna("none").astype(str).value_counts().to_dict()
    count_text = ", ".join(
        f"{priority}: {int(counts.get(priority, 0))}"
        for priority in ["none", "monitor", "plan_repair", "urgent"]
    )
    mean_grade = float(assessment["overall_condition_grade"].astype(float).mean())
    rows: list[str] = []
    for row in assessment.itertuples():
        priority = str(getattr(row, "repair_priority", "none"))
        priority_class = "priority-" + priority.replace("_", "-")
        defects = str(getattr(row, "key_defects", "") or "none")
        rationale = html_lib.escape(str(getattr(row, "rationale", "")))
        rows.append(
            "<tr>"
            f"<td>{int(getattr(row, 'frame_index', 0))}</td>"
            f"<td>{html_lib.escape(str(getattr(row, 'filename', '')))}</td>"
            f"<td>{int(getattr(row, 'overall_condition_grade', 0))}</td>"
            f"<td>{html_lib.escape(str(getattr(row, 'pci_proxy_descriptor', '')))}</td>"
            f'<td><span class="priority-badge {priority_class}">{html_lib.escape(priority)}</span></td>'
            f"<td>{html_lib.escape(defects)}</td>"
            f"<td>{rationale}</td>"
            "</tr>"
        )
    table = (
        '<table class="measure-table"><thead><tr><th>Frame</th><th>File</th>'
        "<th>Condition grade</th><th>Descriptor</th><th>Repair priority</th>"
        "<th>Key defects</th><th>Rationale</th></tr></thead><tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )
    return f"""
    <h2 id="condition-assessment">Condition assessment</h2>
    <div class="panel">
      <p><b>Mean proxy condition grade:</b> {mean_grade:.2f} · <b>Repair priority counts:</b> {html_lib.escape(count_text)}</p>
      <p>This is a PCI-like visual proxy, not an official ASTM D6433 PCI. Binder content, density/air voids, and water-damage progression are not measured.</p>
      {table}
    </div>
"""


def quality_timeline(df: pd.DataFrame) -> str:
    qualities = df["predicted_quality"].fillna(0).astype(int)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"].fillna(df["frame_index"]) if "timestamp" in df else df["frame_index"],
            y=qualities,
            mode="lines+markers",
            marker={
                "size": 10,
                "color": [QUALITY_COLORS.get(int(q), "#9aa0a6") for q in qualities],
            },
            text=df["filename"],
            hovertemplate="%{text}<br>quality=%{y}<extra></extra>",
        )
    )
    fig.update_yaxes(title="Predicted quality", range=[5.3, 0.7], dtick=1)
    fig.update_xaxes(title="Frame / timestamp")
    fig.update_layout(margin={"l": 48, "r": 20, "t": 20, "b": 45}, height=380)
    return pio.to_html(fig, include_plotlyjs=True, full_html=False)


def cracked_sections_panel(run_dir: Path, df: pd.DataFrame) -> str:
    if "crack_ratio" not in df.columns:
        return """
    <h2 id="cracked-sections">Cracked sections</h2>
    <div class="panel">Crack detection head not found for this run.</div>
"""
    return f"""
    <h2 id="cracked-sections">Cracked sections</h2>
    <div class="panel">{crack_timeline(df)}</div>
    <div class="crack-grid">{crack_overlay_gallery(run_dir, df)}</div>
"""


def crack_timeline(df: pd.DataFrame) -> str:
    ratios = df["crack_ratio"].fillna(0.0).astype(float)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=df["timestamp"].fillna(df["frame_index"]) if "timestamp" in df else df["frame_index"],
            y=ratios,
            mode="lines+markers",
            marker={"size": 10, "color": "#d22424"},
            text=df["filename"],
            hovertemplate="%{text}<br>crack_ratio=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_yaxes(title="Crack ratio", range=[-0.03, 1.03])
    fig.update_xaxes(title="Frame / timestamp")
    fig.update_layout(margin={"l": 48, "r": 20, "t": 20, "b": 45}, height=320)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def structural_defects_panel(df: pd.DataFrame) -> str:
    if "structural_defects" not in df.columns:
        return ""
    rows: list[str] = []
    for row in df.itertuples():
        defects = _structural_defect_list(getattr(row, "structural_defects", "[]"))
        tags = (
            " ".join(f'<span class="defect-tag">{defect}</span>' for defect in defects)
            if defects
            else "none"
        )
        ratios = []
        for label in DEFECT_LABELS:
            value = getattr(row, f"defect_{label}_ratio", None)
            if value is not None and not pd.isna(value):
                ratios.append(f"{label}: {float(value):.2f}")
        rows.append(
            f"<tr><td>{row.frame_index}</td><td>{row.filename}</td><td>{tags}</td>"
            f"<td>{', '.join(ratios) or 'n/a'}</td></tr>"
        )
    table = (
        '<table class="measure-table"><thead><tr><th>Frame</th><th>File</th>'
        '<th>Detected defect types</th><th>Tile ratios</th></tr></thead><tbody>'
        + "\n".join(rows)
        + "</tbody></table>"
    )
    return f"""
    <h2 id="structural-defects">Structural defects</h2>
    <div class="panel">{table}</div>
"""


def _structural_defect_list(value: Any) -> list[str]:
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            parsed = []
    elif isinstance(value, list):
        parsed = value
    else:
        parsed = []
    return [str(item) for item in parsed]


def crack_overlay_gallery(run_dir: Path, df: pd.DataFrame) -> str:
    cells: list[str] = []
    for row in df.itertuples():
        thumb = run_dir / row.thumbnail_path
        if not thumb.exists():
            continue
        tiles = json.loads(row.tile_details) if isinstance(row.tile_details, str) else []
        overlays = "\n".join(_tile_overlay(tile) for tile in tiles)
        src = image_to_base64(thumb)
        ratio = float(getattr(row, "crack_ratio", 0.0) or 0.0)
        cells.append(
            f"""<article class="crack-card">
  <div class="crack-image"><img src="data:image/jpeg;base64,{src}" alt="{row.filename}">{overlays}</div>
  <div class="crack-meta"><b>{ratio:.2f}</b> crack ratio<br>{row.filename}</div>
</article>"""
        )
    return "\n".join(cells)


def _tile_overlay(tile: dict[str, Any]) -> str:
    if tile.get("tile_box"):
        left_px, top_px, right_px, lower_px = [float(x) for x in tile["tile_box"]]
        width_pct = max(0.0, right_px - left_px)
        height_pct = max(0.0, lower_px - top_px)
        source_w = max(float(tile.get("image_width", right_px)), 1.0)
        source_h = max(float(tile.get("image_height", lower_px)), 1.0)
        left = left_px / source_w * 100.0
        top = top_px / source_h * 100.0
        width = width_pct / source_w * 100.0
        height = height_pct / source_h * 100.0
    else:
        width = 100.0 / 3.0
        height = 25.0
        top_base = 50.0
        if tile.get("region") == "full":
            height = 100.0 / 3.0
            top_base = 0.0
        tile_name = str(tile.get("tile", "tile_0"))
        try:
            index = int(tile_name.split("_")[-1])
        except ValueError:
            index = 0
        row = index // 3
        col = index % 3
        left = col * (100.0 / 3.0)
        top = top_base + row * height
    hot = bool(tile.get("tile_crack", False))
    prob = float(tile.get("tile_crack_prob", 0.0) or 0.0)
    cls = "crack-tile hot" if hot else "crack-tile"
    tile_name = str(tile.get("tile", "tile_0"))
    return (
        f'<div class="{cls}" title="{tile_name}: crack probability {prob:.3f}" '
        f'style="left:{left:.4f}%;top:{top:.4f}%;width:{width:.4f}%;height:{height:.4f}%;"></div>'
    )


def crack_geometry_panel(run_dir: Path, df: pd.DataFrame) -> str:
    if "crack_area_pct" not in df.columns:
        return ""
    rows = []
    cards = []
    for row in df.itertuples():
        overlay_rel = getattr(row, "crackseg_overlay_path", None)
        overlay = run_dir / overlay_rel if isinstance(overlay_rel, str) else None
        area_pct = float(getattr(row, "crack_area_pct", 0.0) or 0.0)
        length_px = int(getattr(row, "crack_length_px", 0) or 0)
        mean_width_px = float(getattr(row, "crack_mean_width_px", 0.0) or 0.0)
        rows.append(
            f"<tr><td>{row.filename}</td><td>{area_pct:.4f}%</td>"
            f"<td>{length_px}</td><td>{mean_width_px:.2f}</td></tr>"
        )
        if overlay is not None and overlay.exists():
            src = image_to_base64(overlay)
            cards.append(
                f"""<article class="crack-card">
  <img src="data:image/png;base64,{src}" alt="{row.filename} crack geometry">
  <div class="crack-meta"><b>{area_pct:.4f}%</b> area<br>{length_px} px length<br>{row.filename}</div>
</article>"""
            )
    table = (
        '<table class="measure-table"><thead><tr><th>Frame</th><th>Area</th>'
        '<th>Length px</th><th>Mean width px</th></tr></thead><tbody>'
        + "\n".join(rows)
        + "</tbody></table>"
    )
    return f"""
    <h2 id="crack-geometry">Crack geometry</h2>
    <div class="panel">{table}</div>
    <div class="crack-grid">{''.join(cards)}</div>
"""


def umap_scatter(
    ref_xy: np.ndarray,
    ref_df: pd.DataFrame,
    run_xy: np.ndarray,
    run_df: pd.DataFrame,
) -> str:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=ref_xy[:, 0],
            y=ref_xy[:, 1],
            mode="markers",
            marker={"size": 4, "color": "rgba(120,120,120,0.25)"},
            text=ref_df["surface_type"].astype(str) + " q=" + ref_df["quality"].astype(str),
            hovertemplate="reference<br>%{text}<extra></extra>",
            name="reference",
        )
    )
    qualities = run_df["predicted_quality"].fillna(0).astype(int)
    fig.add_trace(
        go.Scatter(
            x=run_xy[:, 0],
            y=run_xy[:, 1],
            mode="markers",
            marker={
                "size": 12,
                "line": {"width": 1, "color": "#111"},
                "color": [QUALITY_COLORS.get(int(q), "#9aa0a6") for q in qualities],
            },
            text=[
                f"{row.filename}<br>quality={row.predicted_quality}<br>type={row.surface_type}<br>confidence={row.confidence:.3f}"
                for row in run_df.itertuples()
            ],
            hovertemplate="%{text}<extra></extra>",
            name="analyzed",
        )
    )
    fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=520)
    return pio.to_html(fig, include_plotlyjs=False, full_html=False)


def gps_scatter(df: pd.DataFrame) -> str:
    gps_df = df.dropna(subset=["latitude", "longitude"])
    if gps_df.empty:
        return ""
    qualities = gps_df["predicted_quality"].fillna(0).astype(int)
    fig = go.Figure(
        go.Scatter(
            x=gps_df["longitude"],
            y=gps_df["latitude"],
            mode="markers",
            marker={
                "size": 11,
                "color": [QUALITY_COLORS.get(int(q), "#9aa0a6") for q in qualities],
            },
            text=gps_df["filename"],
            hovertemplate="%{text}<br>quality=%{marker.color}<extra></extra>",
        )
    )
    fig.update_xaxes(title="Longitude")
    fig.update_yaxes(title="Latitude")
    fig.update_layout(margin={"l": 48, "r": 20, "t": 20, "b": 45}, height=360)
    return f'<h2>GPS Scatter</h2><div class="panel">{pio.to_html(fig, include_plotlyjs=False, full_html=False)}</div>'


def gallery_html(run_dir: Path, df: pd.DataFrame) -> str:
    cells: list[str] = []
    for row in df.itertuples():
        thumb = run_dir / row.thumbnail_path
        if not thumb.exists():
            continue
        q = int(row.predicted_quality) if not pd.isna(row.predicted_quality) else 0
        color = QUALITY_COLORS.get(q, "#d7dce1")
        src = image_to_base64(thumb)
        cells.append(
            f"""<article class="thumb">
  <img src="data:image/jpeg;base64,{src}" alt="{row.filename}">
  <div><span class="badge" style="background:{color}">Q{q or "n/a"}</span> {row.surface_type}<br>{row.filename}</div>
</article>"""
        )
    return "\n".join(cells)
