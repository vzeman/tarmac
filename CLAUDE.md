# Tarmac

Road surface quality analysis using DINOv2/DINOv3 ViT backbone fine-tuning, kNN-based quality clustering, crack detection, structural defect classification, and GPS-annotated video surveys.

```bash
uv run tarmac --help        # list all commands
uv run tarmac label-ui      # start labeling UI at http://127.0.0.1:8765
```

## Documentation

| Doc | What's in it |
|-----|-------------|
| [docs/cli.md](docs/cli.md) | Every CLI command with options, grouped by purpose |
| [docs/datasets.md](docs/datasets.md) | Manifests, crack datasets, data paths |
| [docs/models.md](docs/models.md) | Model checkpoints, FAISS indices, what each .pt file is |
| [docs/training.md](docs/training.md) | Training pipeline, SupCon, kNN F1, domain adaptation, UMAP |
| [docs/labeling-ui.md](docs/labeling-ui.md) | Labeling UI modes, keyboard shortcuts, scatter panel, presets |
| [docs/custom-objects.md](docs/custom-objects.md) | Video frames → dataset, bbox annotation, export COCO/YOLO, train new heads |

## Key paths

```
src/tarmac/
  cli.py        — all CLI commands (typer app)
  cluster/      — k-means clustering on road quality embeddings
  crack/        — crack classifier head, segmentation head, evaluation
  datasets/     — per-dataset download/parse + manifest builders
  defect/       — multi-label structural defect classifier
  embedding/    — DINOv2/DINOv3 backbone embedder, tiling, FAISS index
  inference/    — analyze / assess pipelines for photos, directories, video
  labeling/     — FastAPI server + self-contained HTML labeling UI
  survey/       — GPS/IMU video survey pipeline, strip canvas viewer
  train/        — SupCon fine-tuning, domain adaptation, crack SupCon

data/raw/         — downloaded datasets (one subdirectory per dataset)
data/processed/   — manifests, embeddings, cluster assignments, corrections
models/           — backbone + head checkpoints, FAISS indices
```

## Current status

- `models/crack_finetuned_backbone.pt` — training complete (epoch 7/10, val kNN F1 = 0.9793)
- `data/processed/label_scatter_2d.parquet` — scatter build running (~275k images, ~2-3h)
- Labeling UI ready: `uv run tarmac label-ui`
