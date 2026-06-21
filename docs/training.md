# Training

## ML concepts

**SupCon (Supervised Contrastive Learning)** — fine-tunes the last N ViT blocks. Pulls same-class embeddings together, pushes different-class apart in 768D CLS-token space. Used for both road quality (surface/quality labels) and crack (binary) fine-tuning.

**kNN F1 validation** — evaluates backbone quality without training a classifier head: embed val set with frozen backbone, run k-nearest-neighbours against train embeddings, measure binary F1 on the resulting labels. Used as the training checkpoint criterion for crack fine-tuning.

**WeightedRandomSampler** — balances crack/no-crack class imbalance during crack backbone training (crack images are a minority in most datasets).

**Domain adaptation** — three-stage pipeline: (1) SimSiam SSL on unlabeled road frames from recorded video, (2) kNN pseudo-labeling of adapted embeddings against labeled set, (3) gated re-fine-tune (only accepted if val quality macro-F1 improves).

**UMAP scatter** — 768D CLS embeddings projected to 2D via UMAP for visual exploration. Three datasets merged (crack_manifest + defect_manifest + road manifest) into `label_scatter_2d.parquet`. Used in the labeling UI scatter panel.

## Crack backbone pipeline (full sequence)

```bash
# 1. Download and build crack manifest
uv run tarmac prepare-cracks
# → data/processed/crack_manifest.parquet (~170k images)

# 2. Fine-tune backbone on crack data (SupCon, binary crack/no-crack)
uv run tarmac train-crack-backbone \
  --initial-checkpoint models/finetuned_backbone.pt \
  --epochs 10 --patience 4
# → models/crack_finetuned_backbone.pt  (best epoch by kNN F1)
# → models/crack_finetuned_backbone.json (training history)

# 3. Train binary classifier head on frozen backbone
uv run tarmac train-crack
# → models/crack_head.pt

# 4. Evaluate
uv run tarmac evaluate-crack

# 5. Build labeling scatter (optional, needed for scatter panel in label-ui)
uv run tarmac build-label-scatter
# → data/processed/label_scatter_2d.parquet (~275k points, ~2-3h)
```

## Current best results

`crack_finetuned_backbone.pt` — epoch 7/10, val kNN F1 = **0.9793**.

Full training history in `models/crack_finetuned_backbone.json`.

## Notes

- All training runs on MPS (Apple Silicon). Never falls back silently to CPU.
- Checkpoints saved every epoch with optimizer state for resume support (`models/checkpoints/`).
- Patience-based early stopping: training halts after N epochs without improvement on val kNN F1.
- `train-crack-backbone` accepts `--initial-checkpoint` to warm-start from a road-quality backbone.
