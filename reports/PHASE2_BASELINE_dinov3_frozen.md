# Phase 2 Frozen-Backbone Baseline

Backbone used: `dinov3` (`facebook/dinov3-vitb16-pretrain-lvd1689m`) on `mps`.
Chosen k-means k: `5`.

## kNN Metrics

| Target | Split | Accuracy | Macro-F1 | MAE | Off-by-one accuracy |
|---|---:|---:|---:|---:|---:|
| surface_type | val | 0.7964 | 0.6810 | - | - |
| surface_type | test | 0.8299 | 0.6708 | - | - |
| surface_type | val_test | 0.8131 | 0.6799 | - | - |
| quality | val | 0.4794 | 0.4816 | 0.5928 | 0.9304 |
| quality | test | 0.5000 | 0.4862 | 0.5567 | 0.9485 |
| quality | val_test | 0.4897 | 0.4850 | 0.5747 | 0.9394 |

## Embedding And Cluster Metrics

| Metric | Value |
|---|---:|
| Silhouette vs surface_type | 0.0412 |
| Silhouette vs surface_type + quality | -0.0238 |
| k-means cluster purity: surface_type | 0.5685 |
| k-means cluster purity: quality | 0.4104 |
| k-means cluster purity: surface_type + quality | 0.2636 |

Artifacts: `reports/phase2_metrics.json`, `reports/umap_scatter.html`, `reports/umap_quality.png`.
