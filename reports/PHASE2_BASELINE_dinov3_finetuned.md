# Phase 2 Frozen-Backbone Baseline

Backbone used: `dinov3` (`facebook/dinov3-vitb16-pretrain-lvd1689m`) on `mps`.
Chosen k-means k: `5`.

## kNN Metrics

| Target | Split | Accuracy | Macro-F1 | MAE | Off-by-one accuracy |
|---|---:|---:|---:|---:|---:|
| surface_type | val | 0.9459 | 0.8871 | - | - |
| surface_type | test | 0.9613 | 0.8499 | - | - |
| surface_type | val_test | 0.9536 | 0.8732 | - | - |
| quality | val | 0.6675 | 0.6497 | 0.3351 | 0.9974 |
| quality | test | 0.6649 | 0.6665 | 0.3351 | 1.0000 |
| quality | val_test | 0.6662 | 0.6637 | 0.3351 | 0.9987 |

## Embedding And Cluster Metrics

| Metric | Value |
|---|---:|
| Silhouette vs surface_type | 0.3575 |
| Silhouette vs surface_type + quality | 0.0198 |
| k-means cluster purity: surface_type | 0.9549 |
| k-means cluster purity: quality | 0.4611 |
| k-means cluster purity: surface_type + quality | 0.4440 |

Artifacts: `reports/phase2_metrics.json`, `reports/umap_scatter.html`, `reports/umap_quality.png`.
