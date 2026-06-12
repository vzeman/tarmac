# Tarmac

Road surface quality analysis using unified image manifests, vision embeddings, clustering, inference, HTML reports, and a Streamlit UI.

## Quickstart

```bash
uv sync
uv run tarmac download streetsurfacevis
uv run tarmac prepare
uv run python scripts/smoke_phase1.py
```

The Phase 1 workflow downloads StreetSurfaceVis v1.0 1024px images plus labels into `data/raw/streetsurfacevis/`, then writes a unified manifest to `data/processed/manifest.parquet`.

Useful commands:

```bash
uv run tarmac --help
uv run tarmac download --help
uv run tarmac prepare
```

## Analyze Photos Or Video

Phase 4 uses the active model defined in `models/active_model.json` and the matching reference artifacts under `data/processed/` and `models/`.

```bash
# Analyze one photo.
UV_CACHE_DIR=.uv-cache uv run tarmac analyze path/to/photo.jpg

# Analyze a directory of images.
UV_CACHE_DIR=.uv-cache uv run tarmac analyze path/to/images --out runs/my-road

# Analyze a video at 2 fps, or override frame extraction rate.
UV_CACHE_DIR=.uv-cache uv run tarmac analyze path/to/video.mp4 --fps 2 --out runs/my-video
```

Video analysis requires `ffmpeg` on `PATH`. On macOS:

```bash
brew install ffmpeg
```

Each run writes `results.parquet`, `tiles.parquet`, `summary.json`, and thumbnail images into the run directory. The analyzer embeds the full frame plus six lower-half road tiles, predicts per-tile quality with cosine kNN, marks low-confidence tiles as non-road, and aggregates road tiles to a per-frame result. By default, the non-road threshold is calibrated from validation road tiles and capped at 0.45; pass `--non-road-threshold` to override it.

## HTML Report

```bash
UV_CACHE_DIR=.uv-cache uv run tarmac report runs/my-video
```

The report is written to `runs/my-video/report.html`. It includes headline stats, a quality timeline, UMAP embedding scatter, an embedded thumbnail gallery, and a GPS scatter when EXIF GPS is available. The first report generation fits and persists `models/umap_reducer.pkl`; later reports reuse it and transform new runs into the same reference space.

## Streamlit UI

```bash
UV_CACHE_DIR=.uv-cache uv run tarmac ui
```

Or run Streamlit directly:

```bash
UV_CACHE_DIR=.uv-cache uv run streamlit run src/tarmac/ui/app.py
```

The UI supports a local path or uploaded image/video, runs the same analysis pipeline, shows the results table plus Plotly charts, and provides a download button for the generated HTML report.

## Phase 4/5 Smoke Test

```bash
UV_CACHE_DIR=.uv-cache uv run python scripts/smoke_phase45.py
```

The smoke test copies 12 labeled dataset images to `/tmp/tarmac_test_imgs`, creates `/tmp/tarmac_test.mp4` from 30 dataset images, runs analysis/report generation, verifies the HTML report, and checks that Streamlit serves HTTP 200 within 30 seconds.
