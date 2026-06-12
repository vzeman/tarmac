# Phase 3 Fine-Tuning Comparison

## Training Summary

Run: `dinov3_supcon`; backbone: `facebook/dinov3-vitb16-pretrain-lvd1689m`; device: `mps`; attention: `eager`.
Trained 5 epochs with early-stopping patience 3; best epoch 2 with val quality macro-F1 0.6408.
MPS NaN root cause/fix: previous training code caught MPS failures/non-finite losses and silently moved the run to CPU. The trainer now requires MPS, uses float32 eager attention with AMP/autocast disabled, runs a two-batch finite MPS sanity train step before epochs, writes NaN layer diagnostics, and aborts instead of falling back.

| Epoch | Loss | Val quality macro-F1 | Device |
|---:|---:|---:|---|
| 1 | 1.5950 | 0.6251 | mps |
| 2 | 1.4374 | 0.6408 | mps |
| 3 | 1.3847 | 0.6382 | mps |
| 4 | 1.3407 | 0.6198 | mps |
| 5 | 1.3282 | 0.6330 | mps |

## Four-Way Metrics

| Run | type val acc | type val macro-F1 | type test acc | type test macro-F1 | type val+test acc | type val+test macro-F1 | quality val acc | quality val macro-F1 | quality val MAE | quality val off-by-one | quality test acc | quality test macro-F1 | quality test MAE | quality test off-by-one | quality val+test acc | quality val+test macro-F1 | quality val+test MAE | quality val+test off-by-one | silhouette(type) | silhouette(type+quality) | purity(type) | purity(quality) | purity(type+quality) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Frozen DINOv2 | 0.7758 | 0.6713 | 0.8196 | 0.6734 | 0.7977 | 0.6740 | 0.4691 | 0.4602 | 0.6186 | 0.9124 | 0.4716 | 0.4695 | 0.6289 | 0.9098 | 0.4704 | 0.4648 | 0.6237 | 0.9111 | 0.0343 | -0.0283 | 0.4978 | 0.3967 | 0.2347 |
| Fine-tuned DINOv2 | 0.9536 | 0.9278 | 0.9536 | 0.8527 | 0.9536 | 0.8951 | 0.6546 | 0.6300 | 0.3505 | 0.9948 | 0.6881 | 0.6597 | 0.3144 | 0.9974 | 0.6714 | 0.6466 | 0.3325 | 0.9961 | 0.6073 | 0.0694 | 0.9471 | 0.4631 | 0.4440 |
| Frozen DINOv3 | 0.7964 | 0.6810 | 0.8299 | 0.6708 | 0.8131 | 0.6799 | 0.4794 | 0.4816 | 0.5928 | 0.9304 | 0.5000 | 0.4862 | 0.5567 | 0.9485 | 0.4897 | 0.4850 | 0.5747 | 0.9394 | 0.0412 | -0.0238 | 0.5685 | 0.4104 | 0.2636 |
| Fine-tuned DINOv3 | 0.9459 | 0.8871 | 0.9613 | 0.8499 | 0.9536 | 0.8732 | 0.6675 | 0.6497 | 0.3351 | 0.9974 | 0.6649 | 0.6665 | 0.3351 | 1.0000 | 0.6662 | 0.6637 | 0.3351 | 0.9987 | 0.3575 | 0.0198 | 0.9549 | 0.4611 | 0.4440 |

## Winner

Winner: **Fine-tuned DINOv3** by val+test quality macro-F1 (0.6637); silhouette(type+quality) tiebreaker value 0.0198.
Active model written to `models/active_model.json`: `{"backbone": "dinov3", "checkpoint": "models/finetuned_dinov3.pt"}`.

Winner UMAP artifacts: `reports/umap_scatter_best.html`, `reports/umap_quality_best.png`.

## Caveats

- Fine-tuned DINOv3 improves quality macro-F1 but has lower silhouette(type+quality) than fine-tuned DINOv2; winner selection follows the requested primary metric.
- DINOv3 best epoch by validation quality macro-F1 was epoch 2, not the final epoch; per-epoch checkpoints are under `models/checkpoints/dinov3_supcon/`, with `best.pt` also kept there.
- `models/finetuned_dinov3_backbone_only.pt` is an inference-convenience export used during evaluation; the active checkpoint remains `models/finetuned_dinov3.pt`.
