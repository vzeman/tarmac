# Models & Checkpoints

All checkpoints live in `models/`. `data/` and `models/` are gitignored.

## Backbone checkpoints

| File | Description |
|------|-------------|
| `crack_finetuned_backbone.pt` | **Primary crack backbone.** DINOv2-base, SupCon fine-tuned on crack_manifest. Best epoch 7/10, val kNN F1 = **0.9793**. Last 4 ViT blocks unfrozen. |
| `crack_finetuned_backbone.json` | Training history, config, per-epoch val F1. |
| `finetuned_backbone.pt` | Road quality backbone. SupCon on manifest.parquet (surface/quality labels). |
| `domain_adapt_finetuned.pt` | Domain-adapted backbone. SimSiam SSL on recorded road frames + pseudo-label re-fine-tune. |
| `finetuned_dinov3.pt` | DINOv3 ViT-B/16 fine-tuned backbone (earlier run). |
| `finetuned_dinov3_backbone_only.pt` | DINOv3 backbone-only weights extracted from above. |

## Head checkpoints

| File | Description |
|------|-------------|
| `crack_head.pt` | Binary crack classifier (frozen backbone → 2-class MLP). |
| `crack_seg_head.pt` | Dense patch-token segmentation head for pixel-precise crack masks. |
| `defect_head.pt` | Multi-label structural defect classifier (crack/spalling/efflorescence/exposed_rebar/corrosion). |

## FAISS indices

| File | Description |
|------|-------------|
| `faiss_full.index` | Base DINOv2 embeddings. |
| `faiss_full_dinov3_frozen.index` | DINOv3 frozen. |
| `faiss_full_dinov3_finetuned.index` | DINOv3 fine-tuned. |
| `faiss_full_dinov2_finetuned.index` | DINOv2 fine-tuned. |
| `faiss_full_dinov3_domain_adapt.index` | Domain-adapted backbone. |

## Checkpoints directory

`models/checkpoints/` — per-epoch checkpoints from active training runs (kept for resume support).
