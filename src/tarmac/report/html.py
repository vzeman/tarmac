from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import umap

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
  </style>
</head>
<body>
  <header>
    <h1>Tarmac Analysis Report</h1>
    <div>{summary.get("input_path", run_dir.name)}</div>
  </header>
  <main>
    {stats}
    <h2>Quality Timeline</h2>
    <div class="panel">{timeline}</div>
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
