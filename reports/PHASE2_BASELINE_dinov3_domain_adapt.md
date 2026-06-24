# Phase 2 Frozen-Backbone Baseline

Backbone used: `dinov3` (`facebook/dinov3-vitb16-pretrain-lvd1689m`) on `mps`.
Chosen k-means k: `5`.

## kNN Metrics

| Target | Split | Accuracy | Macro-F1 | MAE | Off-by-one accuracy |
|---|---:|---:|---:|---:|---:|
| surface_type | val | 0.9485 | 0.8910 | - | - |
| surface_type | test | 0.9639 | 0.8653 | - | - |
| surface_type | val_test | 0.9562 | 0.8817 | - | - |
| quality | val | 0.6856 | 0.6641 | 0.3144 | 1.0000 |
| quality | test | 0.6804 | 0.6563 | 0.3247 | 0.9948 |
| quality | val_test | 0.6830 | 0.6625 | 0.3196 | 0.9974 |

## Embedding And Cluster Metrics

| Metric | Value |
|---|---:|
| Silhouette vs surface_type | 0.4429 |
| Silhouette vs surface_type + quality | 0.0334 |
| k-means cluster purity: surface_type | 0.9569 |
| k-means cluster purity: quality | 0.4606 |
| k-means cluster purity: surface_type + quality | 0.4443 |

Artifacts: `reports/phase2_metrics.json`, `reports/umap_scatter.html`, `reports/umap_quality.png`.
