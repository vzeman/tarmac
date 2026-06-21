# CLI Reference

All commands run via `uv run tarmac <command>`. Use `uv run tarmac --help` to list all.

## Data preparation

```bash
uv run tarmac prepare              # build manifest.parquet from data/raw/
uv run tarmac prepare-cracks       # build crack_manifest.parquet
uv run tarmac prepare-defects      # build defect_manifest.parquet
```

## Embedding

```bash
uv run tarmac embed                          # embed manifest.parquet → embeddings.parquet + FAISS
uv run tarmac embed --suffix finetuned       # embed with fine-tuned backbone, suffix artifacts
uv run tarmac embed-defects                  # embed defect_manifest → defect_embeddings.parquet
```

## Training

```bash
# Road quality backbone (SupCon on surface/quality clusters)
uv run tarmac train [--epochs N] [--unfrozen-blocks 4]

# Crack backbone fine-tuning (SupCon on binary crack/no-crack)
uv run tarmac train-crack-backbone \
  [--initial-checkpoint models/finetuned_backbone.pt] \
  [--epochs 10] [--patience 4]

# Crack classifier head (frozen backbone → 2-class MLP)
uv run tarmac train-crack

# Dense segmentation head (pixel-precise crack masks)
uv run tarmac train-seg-head

# Multi-label defect head
uv run tarmac train-defect

# Domain adaptation (SSL on recorded frames + pseudo-labeling + gated re-fine-tune)
uv run tarmac domain-adapt --video path/to/video.mp4 [--video path2.mp4]
```

All training runs on MPS (Apple Silicon). Never falls back silently to CPU — fix or fail.
Checkpoints saved every epoch with optimizer state for resume support.

## Evaluation

```bash
uv run tarmac evaluate                  # embedding + cluster quality metrics
uv run tarmac evaluate --suffix finetuned
uv run tarmac evaluate-crack            # crack head: precision/recall/F1 on val+test
uv run tarmac evaluate-seg-head         # segmentation head: IoU/Dice
uv run tarmac evaluate-defect           # defect head: macro-AP
```

## Inference

```bash
uv run tarmac analyze path/to/image.jpg   # single image
uv run tarmac analyze path/to/dir/        # directory
uv run tarmac analyze path/to/video.mp4   # video

uv run tarmac assess path/to/input        # analyze + PCI-proxy condition report

uv run tarmac crack-measure path/to/img   # crack area/length/width measurements

uv run tarmac visualize path/to/dir/      # project into UMAP space → HTML
uv run tarmac report runs/my-run/         # build HTML report from analyze output
```

## Survey (GPS video)

```bash
uv run tarmac survey video.mp4 [--fps 1.0] [--gps-source auto]
uv run tarmac survey-confirm runs/my-run/    # re-check saved frames with dense crack confirmation
uv run tarmac strip-view runs/my-run/        # build continuous tiled push-broom strip viewer
```

## Labeling UI

```bash
uv run tarmac label-ui                   # start FastAPI server + open browser at http://127.0.0.1:8765
uv run tarmac build-label-scatter        # compute UMAP 2D for scatter panel (all manifests, ~2-3h)
```

See [labeling-ui.md](labeling-ui.md) for full UI documentation.
