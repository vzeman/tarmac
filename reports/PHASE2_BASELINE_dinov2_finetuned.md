# Phase 2 Frozen-Backbone Baseline

Backbone used: `dinov2` (`facebook/dinov2-base`) on `mps`.
Chosen k-means k: `5`.

## kNN Metrics

| Target | Split | Accuracy | Macro-F1 | MAE | Off-by-one accuracy |
|---|---:|---:|---:|---:|---:|
| surface_type | val | 0.9536 | 0.9278 | - | - |
| surface_type | test | 0.9536 | 0.8527 | - | - |
| surface_type | val_test | 0.9536 | 0.8951 | - | - |
| quality | val | 0.6546 | 0.6300 | 0.3505 | 0.9948 |
| quality | test | 0.6881 | 0.6597 | 0.3144 | 0.9974 |
| quality | val_test | 0.6714 | 0.6466 | 0.3325 | 0.9961 |

## Embedding And Cluster Metrics

| Metric | Value |
|---|---:|
| Silhouette vs surface_type | 0.6073 |
| Silhouette vs surface_type + quality | 0.0694 |
| k-means cluster purity: surface_type | 0.9471 |
| k-means cluster purity: quality | 0.4631 |
| k-means cluster purity: surface_type + quality | 0.4440 |

Artifacts: `reports/phase2_metrics.json`, `reports/umap_scatter.html`, `reports/umap_quality.png`.
