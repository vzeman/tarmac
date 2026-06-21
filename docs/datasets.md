# Datasets

## Manifests

| File | Rows | Description |
|------|------|-------------|
| `data/processed/manifest.parquet` | 9,122 | Road quality images. Columns: `image_path`, `source_dataset`, `kind` (full/tile), `split`. No crack labels. Used for backbone SupCon fine-tuning and inference. |
| `data/processed/crack_manifest.parquet` | ~170k | Binary crack detection. Columns: `image_path`, `source_dataset`, `tile`, `has_crack` (0/1), `split`. 70/15/15 train/val/test per (source, label). |
| `data/processed/defect_manifest.parquet` | ~98k | Multi-label structural defects. Built from CODEBRIM-style datasets. |
| `data/processed/label_corrections.parquet` | — | Manual label overrides saved by the labeling UI. Columns: `id`, `image_path`, `labels_json` (JSON dict). |
| `data/processed/label_scatter_2d.parquet` | ~275k | 2D UMAP projection for the scatter panel. Built by `build-label-scatter`. |

## Crack datasets

All sourced into `data/raw/` and unified by `prepare-cracks`:

| Dataset | Source | Notes |
|---------|--------|-------|
| `cracks_concrete_pavement` | Kaggle | 15k crack + 15k no-crack |
| `crack500` | SDNET subset | 85 images, crack only |
| `deepcrack` | GitHub | 1,076 images |
| `deepcrack_liu` | GitHub | 527 images |
| `hf_crack` | HuggingFace | 1,145 images |
| `masonry_crack` | GitHub pairs | 100 images |
| `mendeley5y9` | Mendeley | 20k crack + 20k no-crack |
| `rdd2022_*` | RDD2022 (6 countries) | ~25k total, crack-annotated only |
| `runway_roboflow` | Roboflow | 240 images, binary |
| `sdnet2018` | SDNET2018 | ~55k images (bridge/wall/pavement) |
| `khanh11k` | Google Drive (manual) | seg pairs |
| `crack500_seg` | OneDrive (manual) | seg pairs |
| `metu_crack_seg` | Mendeley (manual) | seg pairs |
| `paggnet_crack` | GitHub pairs | seg pairs |

Manual-only datasets (require downloading by hand before `prepare-cracks` can include them):
`khanh11k`, `crack500_seg`, `metu_crack_seg`.
