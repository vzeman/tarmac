# Phase 2 Frozen-Backbone Baseline

Backbone used: `dinov2` (`facebook/dinov2-base`) on `mps`.
Chosen k-means k: `5`.
Requested backbone `facebook/dinov3-vitb16-pretrain-lvd1689m` was unavailable, so the run used the fallback backbone.

## kNN Metrics

| Target | Split | Accuracy | Macro-F1 | MAE | Off-by-one accuracy |
|---|---:|---:|---:|---:|---:|
| surface_type | val | 0.7758 | 0.6713 | - | - |
| surface_type | test | 0.8196 | 0.6734 | - | - |
| surface_type | val_test | 0.7977 | 0.6740 | - | - |
| quality | val | 0.4691 | 0.4602 | 0.6186 | 0.9124 |
| quality | test | 0.4716 | 0.4695 | 0.6289 | 0.9098 |
| quality | val_test | 0.4704 | 0.4648 | 0.6237 | 0.9111 |

## Embedding And Cluster Metrics

| Metric | Value |
|---|---:|
| Silhouette vs surface_type | 0.0343 |
| Silhouette vs surface_type + quality | -0.0283 |
| k-means cluster purity: surface_type | 0.4978 |
| k-means cluster purity: quality | 0.3967 |
| k-means cluster purity: surface_type + quality | 0.2347 |

Artifacts: `reports/phase2_metrics.json`, `reports/umap_scatter.html`, `reports/umap_quality.png`.
