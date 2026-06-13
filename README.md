# Tarmac — Road Surface Quality Analysis

Tarmac analyzes photos and videos of road surfaces (asphalt/tarmac, concrete, paving stones, sett, unpaved) and grades their **quality** on a 1–5 scale (1 = excellent, 5 = very bad). It works by turning every image — and every road tile within it — into an embedding vector, then classifying it against a reference set of tens of thousands of labelled surfaces using cosine similarity. The result lets you map how road quality changes along a street and see which sections are good and which are failing.

## What the quality space looks like

Each dot below is one road image, projected from a 768-dimensional embedding down to 2-D with UMAP and coloured by its quality grade (purple = excellent → yellow = very bad). A good model should arrange images so that similar-quality surfaces sit together.

| Off-the-shelf backbone (frozen) | After supervised-contrastive fine-tuning |
| --- | --- |
| ![Frozen embeddings — quality grades are mixed](reports/umap_quality.png) | ![Fine-tuned embeddings — quality grades separate cleanly](reports/umap_quality_best.png) |
| Quality grades are smeared across the space — the raw backbone tells *what* a surface is, but not *how good* it is. | After fine-tuning, quality forms clear regions: very-bad/unpaved (yellow) breaks off into its own island, excellent asphalt (purple) clusters on the right. New images land in the right neighbourhood. |

This separation is what makes cosine-similarity classification reliable: a new photo is embedded, and its nearest neighbours in this space vote on its quality and surface type.

## How it works

```
photo / video frame
   │
   ├─ split into road tiles (3×2 grid over the lower half of the frame)
   │
   ▼
DINOv3 ViT-B/16 backbone  ──►  768-d embedding per tile  (L2-normalised)
   │                                   │
   │                                   ▼
   │                          FAISS cosine index of ~9k labelled reference tiles
   │                                   │
   ▼                                   ▼
per-tile prediction  ◄──  k-NN vote (quality 1–5, surface type, confidence)
   │
   ▼
per-frame result = median quality of road tiles, majority surface type
   │
   ▼
report:  quality-along-the-street timeline · UMAP scatter · GPS map · gallery
```

1. **Tiling.** Full frames contain sky, cars and buildings, so each frame is cut into a 3×2 grid of tiles over the lower (road) half. Quality is judged per tile and aggregated, and tiles that don't look like any road surface are dropped as *non-road*.
2. **Embedding.** A [DINOv3 ViT-B/16](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m) vision transformer turns each tile into a 768-dimensional vector. DINOv3 is a strong texture/structure encoder out of the box; we fine-tune the last 4 transformer blocks.
3. **Fine-tuning.** A **supervised-contrastive loss** pulls together tiles that share the same (surface type, quality) label and pushes apart those that differ. This is what turns the left scatter above into the right one. Training runs on Apple-Silicon GPU (MPS), checkpoints every epoch, and stops early on the best validation score.
4. **Classification.** New tiles are matched against the labelled reference set with **cosine k-NN** (FAISS inner-product index over normalised vectors). The reference set's k-means clusters also let you find "similar surfaces" by nearest centroid.
5. **Reporting.** Per-frame results become an HTML report: quality over time/distance, a UMAP scatter of the analysed frames inside the reference space, an EXIF-GPS map when available, and a thumbnail gallery.

Why not reinforcement learning or a world model? The task is *representation learning + metric classification*, not sequential decision-making — there is no agent or reward. A vision transformer embedding space, fine-tuned contrastively, is the direct fit. See [`PLAN.md`](PLAN.md) for the full rationale and roadmap.

## Results

Cosine k-NN (k=10) classification on the held-out validation + test split of [StreetSurfaceVis](https://zenodo.org/records/11449977) (9,122 images). "Off-by-one" is the share of quality predictions within one grade of the truth.

| Model | Surface type acc / F1 | Quality acc / F1 | Quality MAE | Off-by-one | Silhouette (type+quality) |
| --- | --- | --- | --- | --- | --- |
| DINOv2 frozen | 0.798 / 0.674 | 0.470 / 0.465 | 0.62 | 0.911 | −0.028 |
| DINOv3 frozen | 0.813 / 0.680 | 0.490 / 0.485 | 0.58 | 0.939 | −0.024 |
| DINOv2 fine-tuned | 0.954 / **0.895** | 0.671 / 0.647 | 0.33 | 0.996 | **0.069** |
| **DINOv3 fine-tuned** (active) | 0.954 / 0.873 | 0.666 / **0.664** | 0.34 | **0.999** | 0.020 |

Fine-tuning is the decisive step: surface-type accuracy rises from ~0.80 to **0.954** and quality macro-F1 from ~0.47 to **0.66**, with quality errors almost always just one adjacent grade (off-by-one ≈ **99.9%**). The active model — fine-tuned DINOv3 — is recorded in `models/active_model.json`. Full breakdown in [`reports/PHASE3_FINETUNE.md`](reports/PHASE3_FINETUNE.md).

## Quickstart

```bash
uv sync                                  # create env + install deps
uv run tarmac download streetsurfacevis  # fetch the reference dataset (~2.4 GB)
uv run tarmac prepare                     # build the unified manifest
```

Datasets and model checkpoints are **not** committed to git (see `.gitignore`); the commands above reproduce them locally.

### Analyze photos or video

```bash
# A single photo, a directory of images, or a video.
uv run tarmac analyze path/to/photo.jpg
uv run tarmac analyze path/to/images --out runs/my-road
uv run tarmac analyze path/to/video.mp4 --fps 2 --out runs/my-video
```

Video requires `ffmpeg` (`brew install ffmpeg`). Each run writes `results.parquet`, `tiles.parquet`, `summary.json` and thumbnails into the run directory.

### Generate an HTML report

```bash
uv run tarmac report runs/my-video      # -> runs/my-video/report.html
```

Headline stats, a quality timeline, a UMAP scatter of the run inside the reference space, a GPS map (when EXIF GPS exists), and a thumbnail gallery.

### Visualize a folder of images in the vector space

```bash
uv run tarmac visualize path/to/folder  # -> reports/visualize_<folder>.html
```

Plots every image in the folder against the gray reference cloud, coloured by predicted quality. **Click any dot to open a dialog showing that image**, its filename and predicted grade. Self-contained HTML — open it straight from disk.

### Interactive UI

```bash
uv run tarmac ui                         # Streamlit app
```

Upload a photo/video or point at a local path, run the pipeline, and browse the table, charts and downloadable report in the browser.

## Datasets

| Dataset | Size | Labels | Role |
| --- | --- | --- | --- |
| [StreetSurfaceVis](https://zenodo.org/records/11449977) | 9,122 | surface type × quality (excellent→very bad) | Primary — trained & evaluated here |
| [RSCD](https://thu-rsxd.com/rscd/) | 1M | material × unevenness × friction | Scale-up (downloader included) |
| [RTK](https://data.mendeley.com/datasets/fxy5khmhpb/1) | 77,547 | asphalt/paved/unpaved + defects | Scale-up (downloader included) |
| [CQU-BPDD](https://github.com/DearCaat/CQU-BPDD), [CRACK500](https://github.com/fyangneil/pavement-crack-detection) | 60k / 500 | crack & distress types, pixel masks | Planned: crack detection (Phase 7) |

## Project layout

```
src/tarmac/
  datasets/   downloaders + unification to one manifest
  embedding/  DINOv3 embedder, road tiling
  train/      supervised-contrastive fine-tuning
  cluster/    k-means / HDBSCAN, cosine assignment
  eval/       accuracy, F1, silhouette, UMAP scatter
  inference/  photo/video analysis, folder visualization
  report/     HTML report + click-to-view UMAP scatter
  ui/         Streamlit app
reports/      committed metrics + visualizations
PLAN.md       full architecture, decisions and phase plan
```

## Roadmap

Built so far: data pipeline, embeddings, contrastive fine-tuning, clustering, evaluation, inference, reports, folder visualization and UI. Next (see `PLAN.md`, Phase 7): explicit **crack and defect detection** — a multi-label defect head, defect-aware embeddings, and crack heatmaps on DINOv3 dense patch tokens.
