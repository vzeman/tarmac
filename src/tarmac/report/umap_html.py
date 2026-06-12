from __future__ import annotations

import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

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


def reference_scatter_html(
    df: pd.DataFrame,
    projection: np.ndarray,
    path: Path,
    *,
    title: str = "Frozen-backbone UMAP projection",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    custom = reference_customdata(df)
    hovertemplate = (
        "path=%{customdata[1]}<br>type=%{customdata[3]}<br>"
        "quality=%{customdata[4]}<br>split=%{customdata[7]}<extra></extra>"
    )
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Surface type", "Quality"))
    surface_codes, surface_uniques = pd.factorize(df["surface_type"])
    fig.add_trace(
        go.Scattergl(
            x=projection[:, 0],
            y=projection[:, 1],
            mode="markers",
            marker={"color": surface_codes, "colorscale": "Turbo", "size": 5, "opacity": 0.75},
            customdata=custom,
            hovertemplate=hovertemplate,
            name="surface_type",
            text=[surface_uniques[i] for i in surface_codes],
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scattergl(
            x=projection[:, 0],
            y=projection[:, 1],
            mode="markers",
            marker={
                "color": df["quality"].astype(int),
                "colorscale": QUALITY_COLORSCALE,
                "cmin": 1,
                "cmax": 5,
                "size": 5,
                "opacity": 0.75,
            },
            customdata=custom,
            hovertemplate=hovertemplate,
            name="quality",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(title=title, template="plotly_white", height=720)
    write_clickable_html(fig, path)


def visualize_scatter_html(
    ref_xy: np.ndarray,
    ref_df: pd.DataFrame,
    item_xy: np.ndarray,
    item_df: pd.DataFrame,
    path: Path,
    *,
    title: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=ref_xy[:, 0],
            y=ref_xy[:, 1],
            mode="markers",
            marker={"size": 4, "color": "rgba(120,120,120,0.25)"},
            customdata=reference_customdata(ref_df),
            text=ref_df["surface_type"].astype(str) + " q=" + ref_df["quality"].astype(str),
            hovertemplate="reference<br>%{text}<extra></extra>",
            name="reference",
        )
    )
    qualities = item_df["predicted_quality"].astype(int)
    fig.add_trace(
        go.Scatter(
            x=item_xy[:, 0],
            y=item_xy[:, 1],
            mode="markers",
            marker={
                "size": 12,
                "line": {"width": 1, "color": "#111"},
                "color": qualities,
                "colorscale": QUALITY_COLORSCALE,
                "cmin": 1,
                "cmax": 5,
                "colorbar": {"title": "quality"},
            },
            customdata=folder_customdata(item_df),
            text=[
                f"{row.filename}<br>quality={row.predicted_quality}<br>type={row.surface_type}<br>confidence={row.confidence:.3f}"
                for row in item_df.itertuples()
            ],
            hovertemplate="%{text}<extra></extra>",
            name="folder images",
        )
    )
    fig.update_layout(
        title=title,
        template="plotly_white",
        height=720,
        margin={"l": 40, "r": 24, "t": 60, "b": 45},
    )
    write_clickable_html(fig, path)


def embeddable_scatter_html(
    ref_xy: np.ndarray,
    ref_df: pd.DataFrame,
    item_xy: np.ndarray,
    item_df: pd.DataFrame,
) -> str:
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=ref_xy[:, 0],
            y=ref_xy[:, 1],
            mode="markers",
            marker={"size": 4, "color": "rgba(120,120,120,0.25)"},
            customdata=reference_customdata(ref_df),
            text=ref_df["surface_type"].astype(str) + " q=" + ref_df["quality"].astype(str),
            hovertemplate="reference<br>%{text}<extra></extra>",
            name="reference",
        )
    )
    qualities = item_df["predicted_quality"].fillna(0).astype(int)
    fig.add_trace(
        go.Scatter(
            x=item_xy[:, 0],
            y=item_xy[:, 1],
            mode="markers",
            marker={
                "size": 12,
                "line": {"width": 1, "color": "#111"},
                "color": [QUALITY_COLORS.get(int(q), "#9aa0a6") for q in qualities],
            },
            customdata=folder_customdata(item_df),
            text=[
                f"{row.filename}<br>quality={row.predicted_quality}<br>type={row.surface_type}<br>confidence={row.confidence:.3f}"
                for row in item_df.itertuples()
            ],
            hovertemplate="%{text}<extra></extra>",
            name="analyzed",
        )
    )
    fig.update_layout(margin={"l": 20, "r": 20, "t": 20, "b": 20}, height=520)
    return pio.to_html(
        fig,
        include_plotlyjs=False,
        full_html=False,
        div_id="umap-scatter",
        post_script=click_handler_js(),
    ) + dialog_html()


def reference_customdata(df: pd.DataFrame) -> list[list[Any]]:
    paths = df["image_path"].astype(str).map(_absolute_file_url)
    filenames = df["image_path"].astype(str).map(lambda value: Path(value).name)
    return np.stack(
        [
            np.full(len(df), "reference", dtype=object),
            df["image_path"].astype(str).to_numpy(),
            paths.to_numpy(),
            df["surface_type"].astype(str).to_numpy(),
            df["quality"].astype(str).to_numpy(),
            np.full(len(df), "", dtype=object),
            filenames.to_numpy(),
            df.get("split", pd.Series([""] * len(df))).astype(str).to_numpy(),
            np.full(len(df), "", dtype=object),
        ],
        axis=1,
    ).tolist()


def folder_customdata(df: pd.DataFrame) -> list[list[Any]]:
    return np.stack(
        [
            np.full(len(df), "folder", dtype=object),
            df["source_path"].astype(str).to_numpy(),
            df["thumbnail_data_url"].astype(str).to_numpy(),
            df["surface_type"].astype(str).to_numpy(),
            df["predicted_quality"].astype(str).to_numpy(),
            df["confidence"].map(lambda value: f"{float(value):.3f}").to_numpy(),
            df["filename"].astype(str).to_numpy(),
            np.full(len(df), "", dtype=object),
            df["source_path"].astype(str).map(_absolute_file_url).to_numpy(),
        ],
        axis=1,
    ).tolist()


def write_clickable_html(fig: go.Figure, path: Path) -> None:
    body = pio.to_html(
        fig,
        include_plotlyjs=True,
        full_html=False,
        div_id="umap-scatter",
        post_script=click_handler_js(),
    )
    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(str(fig.layout.title.text or "Tarmac UMAP"))}</title>
  {dialog_css()}
</head>
<body>
  {body}
  {dialog_html(include_style=False)}
</body>
</html>
"""
    path.write_text(html_text, encoding="utf-8")


def dialog_css() -> str:
    return """<style>
body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #17202a; }
.img-dialog-backdrop { position: fixed; inset: 0; display: none; align-items: center; justify-content: center; background: rgba(0,0,0,0.58); z-index: 10000; padding: 24px; }
.img-dialog-backdrop.open { display: flex; }
.img-dialog { width: min(760px, 96vw); max-height: 92vh; overflow: auto; background: #fff; border-radius: 8px; box-shadow: 0 18px 60px rgba(0,0,0,0.35); }
.img-dialog header { display: flex; align-items: center; justify-content: space-between; gap: 12px; padding: 14px 16px; border-bottom: 1px solid #dde2e7; }
.img-dialog h2 { margin: 0; font-size: 18px; line-height: 1.3; overflow-wrap: anywhere; }
.img-dialog button { border: 0; background: transparent; font-size: 28px; line-height: 1; cursor: pointer; color: #34495e; }
.img-dialog-body { padding: 16px; }
.img-dialog-body img { display: block; max-width: 100%; max-height: 512px; margin: 0 auto 14px; object-fit: contain; background: #f2f4f7; }
.img-dialog-meta { display: grid; grid-template-columns: max-content minmax(0, 1fr); gap: 8px 12px; font-size: 14px; }
.img-dialog-meta dt { font-weight: 700; color: #4b5563; }
.img-dialog-meta dd { margin: 0; overflow-wrap: anywhere; }
</style>"""


def dialog_html(*, include_style: bool = True) -> str:
    style = dialog_css() if include_style else ""
    return f"""{style}
<div id="img-dialog" class="img-dialog-backdrop" role="dialog" aria-modal="true" aria-labelledby="img-dialog-title">
  <section class="img-dialog">
    <header>
      <h2 id="img-dialog-title">Image</h2>
      <button id="img-dialog-close" type="button" aria-label="Close">&times;</button>
    </header>
    <div class="img-dialog-body">
      <img id="img-dialog-image" alt="">
      <dl class="img-dialog-meta">
        <dt>Filename</dt><dd id="img-dialog-filename"></dd>
        <dt>Quality</dt><dd id="img-dialog-quality"></dd>
        <dt>Surface type</dt><dd id="img-dialog-surface"></dd>
        <dt>Confidence</dt><dd id="img-dialog-confidence"></dd>
        <dt>Path</dt><dd id="img-dialog-path"></dd>
      </dl>
    </div>
  </section>
</div>"""


def click_handler_js() -> str:
    return r"""
function tarmacDialogText(id, value) {
  var element = document.getElementById(id);
  if (element) element.textContent = value || "";
}
function tarmacOpenImageDialog(data) {
  if (!data) return;
  var dialog = document.getElementById("img-dialog");
  var image = document.getElementById("img-dialog-image");
  if (!dialog || !image) return;
  var kind = data[0] || "";
  var path = data[1] || "";
  var src = data[2] || "";
  var surface = data[3] || "";
  var quality = data[4] || "";
  var confidence = data[5] || "";
  var filename = data[6] || path;
  var displayPath = data[8] || src || path;
  image.src = src || displayPath;
  image.alt = filename;
  tarmacDialogText("img-dialog-title", filename);
  tarmacDialogText("img-dialog-filename", filename);
  tarmacDialogText("img-dialog-quality", quality ? "Q" + quality : "");
  tarmacDialogText("img-dialog-surface", surface);
  tarmacDialogText("img-dialog-confidence", confidence || (kind === "reference" ? "reference point" : ""));
  tarmacDialogText("img-dialog-path", displayPath || path);
  dialog.classList.add("open");
}
function tarmacCloseImageDialog() {
  var dialog = document.getElementById("img-dialog");
  var image = document.getElementById("img-dialog-image");
  if (dialog) dialog.classList.remove("open");
  if (image) image.removeAttribute("src");
}
document.addEventListener("click", function(event) {
  if (event.target && event.target.id === "img-dialog") tarmacCloseImageDialog();
  if (event.target && event.target.id === "img-dialog-close") tarmacCloseImageDialog();
});
document.addEventListener("keydown", function(event) {
  if (event.key === "Escape") tarmacCloseImageDialog();
});
var plot = document.getElementById("{plot_id}");
if (plot && plot.on) {
  plot.on("plotly_click", function(eventData) {
    if (eventData && eventData.points && eventData.points.length) {
      tarmacOpenImageDialog(eventData.points[0].customdata);
    }
  });
}
"""


def _absolute_file_url(value: str) -> str:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path.cwd() / path
    return path.resolve().as_uri()
