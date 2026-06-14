# Domain Adaptation On Recorded Road Frames

## Data

- Frames extracted: 2376
- Road tiles extracted: 14256
- Pseudo-labeled tiles selected: 0 of 14256 scored
- Augmented manifest rows: 9122 (9122 StreetSurfaceVis + 0 pseudo)

## Self-Supervised Adaptation

- Method: SimSiam negative-cosine self-distillation with two strong augmentations of each road tile.
- Epochs: 3
- Tiles used for SSL: 6000
- Conservative settings: backbone LR 1e-5, last 2 DINOv3 blocks unfrozen, MPS/eager attention, no CPU fallback.
- Checkpoint: `models/checkpoints/domain_adapt/domain_adapted.pt`

## Pseudo-Labeling

- Reference: current active fine-tuned model `models/finetuned_dinov3.pt` against StreetSurfaceVis cosine kNN.
- Thresholds: mean neighbor cosine >= 0.80, surface margin >= 0.10, quality margin >= 0.08.
- Balancing: max_total=3000, max_per_surface_quality=150.

### Pseudo-Label Counts

| Group | Counts |
|---|---|
| surface_type | `{}` |
| quality | `{}` |
| surface_type+quality | `{}` |

## Held-Out StreetSurfaceVis Metrics

| Metric | Current active | Candidate | Delta |
|---|---:|---:|---:|
| surface_type val+test accuracy | 0.9536 | 0.9562 | +0.0026 |
| surface_type val+test macro-F1 | 0.8732 | 0.8817 | +0.0085 |
| quality val+test accuracy | 0.6662 | 0.6830 | +0.0168 |
| quality val+test macro-F1 | 0.6637 | 0.6625 | -0.0012 |
| quality val+test MAE | 0.3351 | 0.3196 | -0.0155 |
| quality val+test off-by-one | 0.9987 | 0.9974 | -0.0013 |
| silhouette surface_type | 0.3575 | 0.4429 | +0.0854 |
| silhouette surface_type+quality | 0.0198 | 0.0334 | +0.0136 |

## Acceptance Gate

- Required: quality macro-F1 must improve above 0.6637.
- Required: type accuracy must stay within 0.0050 of 0.9536.
- Quality macro-F1 pass: no (0.6625, delta -0.0012).
- Type accuracy pass: yes (0.9562, delta +0.0026).
- Gate verdict: **REJECTED**.

## Active Model

- Before: `models/finetuned_dinov3.pt`
- After: `models/finetuned_dinov3.pt`
- Candidate suffix/artifacts: `dinov3_domain_adapt`
- Candidate best SupCon epoch: 1
- Candidate clustering k: 5
- The active model was not replaced because the strict gate did not pass.
