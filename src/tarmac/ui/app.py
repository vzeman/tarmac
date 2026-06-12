from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from tarmac.inference.analyze import analyze_path
from tarmac.report.html import QUALITY_COLORS, build_html_report, load_or_fit_umap
from tarmac.inference.analyze import load_active_artifacts


def timeline_figure(df: pd.DataFrame) -> go.Figure:
    qualities = df["predicted_quality"].fillna(0).astype(int)
    fig = go.Figure(
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
    fig.update_layout(height=360, margin={"l": 40, "r": 20, "t": 20, "b": 40})
    return fig


def umap_figure(df: pd.DataFrame) -> go.Figure:
    artifacts = load_active_artifacts()
    ref_df = pd.read_parquet(artifacts.embeddings_path)
    ref_df = ref_df[ref_df["kind"] == "full"].reset_index(drop=True)
    ref_embeddings = np.vstack(ref_df["embedding"].to_numpy()).astype("float32")
    run_embeddings = np.vstack(df["embedding"].to_numpy()).astype("float32")
    reducer = load_or_fit_umap(Path("models/umap_reducer.pkl"), ref_embeddings)
    ref_xy = reducer.embedding_
    run_xy = reducer.transform(run_embeddings)
    qualities = df["predicted_quality"].fillna(0).astype(int)
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=ref_xy[:, 0],
            y=ref_xy[:, 1],
            mode="markers",
            marker={"size": 4, "color": "rgba(120,120,120,0.22)"},
            name="reference",
            hoverinfo="skip",
        )
    )
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
            text=df["filename"],
            hovertemplate="%{text}<extra></extra>",
            name="analyzed",
        )
    )
    fig.update_layout(height=500, margin={"l": 20, "r": 20, "t": 20, "b": 20})
    return fig


st.set_page_config(page_title="Tarmac", layout="wide")
st.title("Tarmac")

source_mode = st.radio("Input", ["Path", "Upload"], horizontal=True)
fps = st.number_input("Video FPS", min_value=0.1, max_value=30.0, value=2.0, step=0.5)
auto_threshold = st.toggle("Auto non-road threshold", value=True)
threshold = st.slider("Non-road threshold", min_value=0.0, max_value=1.0, value=0.45, step=0.01)

input_path: Path | None = None
temp_file: tempfile.NamedTemporaryFile | None = None
if source_mode == "Path":
    path_text = st.text_input("Photo, directory, or video path")
    if path_text:
        input_path = Path(path_text).expanduser()
else:
    upload = st.file_uploader("Upload photo or video", type=["jpg", "jpeg", "png", "webp", "mp4", "mov", "m4v"])
    if upload is not None:
        suffix = Path(upload.name).suffix
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        temp_file.write(upload.getbuffer())
        temp_file.close()
        input_path = Path(temp_file.name)

if st.button("Analyze", type="primary", disabled=input_path is None):
    assert input_path is not None
    progress = st.progress(0, text="Preparing analysis")
    try:
        out_dir = Path("runs") / input_path.stem
        progress.progress(15, text="Embedding frames and tiles")
        summary = analyze_path(
            input_path=input_path,
            out_dir=out_dir,
            fps=float(fps),
            non_road_threshold=None if auto_threshold else float(threshold),
            device="cpu",
        )
        progress.progress(75, text="Loading results")
        df = pd.read_parquet(summary["results_parquet"])
        st.session_state["run_dir"] = summary["out_dir"]
        st.session_state["results"] = df
        progress.progress(100, text="Done")
    except Exception as exc:
        st.exception(exc)

df = st.session_state.get("results")
run_dir = st.session_state.get("run_dir")
if df is not None and run_dir is not None:
    st.subheader("Results")
    st.dataframe(
        df.drop(columns=["embedding", "tile_details"], errors="ignore"),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Quality Timeline")
    st.plotly_chart(timeline_figure(df), use_container_width=True)

    st.subheader("UMAP Scatter")
    try:
        st.plotly_chart(umap_figure(df), use_container_width=True)
    except Exception as exc:
        st.warning(f"UMAP scatter unavailable: {exc}")

    report_path = build_html_report(Path(run_dir))
    st.download_button(
        "Download HTML report",
        data=report_path.read_bytes(),
        file_name=report_path.name,
        mime="text/html",
    )
