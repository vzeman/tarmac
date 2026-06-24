# Crack Segmentation and Measurement

Phase 7c added full-frame runway/pavement analysis and pixel-level crack geometry. Track 2 replaces the default pixel mask with a learned frozen-DINOv3 dense-token segmentation head when `models/crack_seg_head.pt` is present.

## Default Segmenter

Default for `tarmac crack-measure` and `tarmac analyze --crack-segmentation`: `dinov3_dense_head`.

The learned model uses the active fine-tuned DINOv3 ViT-B/16 backbone frozen at 512 px input resolution. The 32x32 patch-token grid is decoded by a lightweight convolutional upsampler to a full-resolution crack logit map. The classical Frangi/Sato/black-hat method remains the fallback only when the learned checkpoint is absent.

Chosen threshold: `0.950` (max f0.5 on validation).

## Dense-Head Metrics

| Split | Source | IoU | Dice/F1 | Precision | Recall | Pixel accuracy | Images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val | overall | 0.5958 | 0.7467 | 0.7310 | 0.7631 | 0.9909 | 622 |
| val | crackairport | 0.5164 | 0.6811 | 0.6407 | 0.7268 | 0.9923 | 338 |
| val | crackforest | 0.2254 | 0.3679 | 0.7667 | 0.2420 | 0.9583 | 18 |
| val | deepcrack_liu | 0.7427 | 0.8524 | 0.8347 | 0.8708 | 0.9892 | 79 |
| val | hf_crack | 0.6707 | 0.8029 | 0.7498 | 0.8640 | 0.9919 | 172 |
| val | masonry_crack | 0.4895 | 0.6573 | 0.7865 | 0.5645 | 0.9946 | 15 |
| test | overall | 0.6103 | 0.7580 | 0.7189 | 0.8015 | 0.9919 | 619 |
| test | crackairport | 0.5212 | 0.6853 | 0.6333 | 0.7465 | 0.9921 | 337 |
| test | crackforest | 0.4540 | 0.6244 | 0.6959 | 0.5663 | 0.9895 | 17 |
| test | deepcrack_liu | 0.7405 | 0.8509 | 0.8367 | 0.8655 | 0.9905 | 79 |
| test | hf_crack | 0.6631 | 0.7974 | 0.7453 | 0.8574 | 0.9923 | 171 |
| test | masonry_crack | 0.3931 | 0.5644 | 0.7439 | 0.4546 | 0.9930 | 15 |

## Common Test Comparison

Pixel metrics below use the held-out split from `data/processed/crack_seg_expanded/manifest.jsonl` when present, otherwise the deterministic CrackAirport + CrackForest raw-data split, with masks resized to 512 px.

| Segmenter | IoU | Dice/F1 | Precision | Recall | Images |
| --- | ---: | ---: | ---: | ---: | ---: |
| dinov3_dense_head | 0.6103 | 0.7580 | 0.7189 | 0.8015 | 619 |
| classical | 0.0242 | 0.0473 | 0.0753 | 0.0345 | 619 |

Verdict: the DINOv3 dense head is the selected default because it is trained directly on pixel masks from CrackAirport and CrackForest and produces a full-resolution mask that keeps the existing area, length, and width measurement path intact. The classical method remains the no-checkpoint fallback.

## Example Overlays

License-safe CrackAirport examples, left to right: original, ground-truth mask, learned prediction.

- `reports/examples/08_crack_seg_learned.png`
- `reports/examples/08_crack_seg_learned_2.png`
- `reports/examples/08_crack_seg_learned_3.png`

## Outputs

- `tarmac crack-measure <image|dir> --out DIR`: `<name>_crackseg.png`, `crack_measurements.csv`, `crack_measurements.parquet`.
- `tarmac analyze --region full --crack-segmentation`: `crackseg/frame_*_crackseg.png` plus geometry columns in `results.parquet`.
- `tarmac report`: crack geometry overlay gallery and measurement table.

