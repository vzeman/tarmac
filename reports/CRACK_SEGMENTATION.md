# Crack Segmentation and Measurement

Phase 7c added full-frame runway/pavement analysis and pixel-level crack geometry. Track 2 replaces the default pixel mask with a learned frozen-DINOv3 dense-token segmentation head when `models/crack_seg_head.pt` is present.

## Default Segmenter

Default for `tarmac crack-measure` and `tarmac analyze --crack-segmentation`: `dinov3_dense_head`.

The learned model uses the active fine-tuned DINOv3 ViT-B/16 backbone frozen at 512 px input resolution. The 32x32 patch-token grid is decoded by a lightweight convolutional upsampler to a full-resolution crack logit map. The classical Frangi/Sato/black-hat method remains the fallback only when the learned checkpoint is absent.

Chosen threshold: `0.875` (max Dice on validation).

## Dense-Head Metrics

| Split | Source | IoU | Dice/F1 | Precision | Recall | Pixel accuracy | Images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| val | overall | 0.4984 | 0.6652 | 0.5992 | 0.7476 | 0.9900 | 356 |
| val | crackairport | 0.5277 | 0.6909 | 0.5981 | 0.8176 | 0.9917 | 338 |
| val | crackforest | 0.3481 | 0.5165 | 0.6075 | 0.4492 | 0.9580 | 18 |
| test | overall | 0.5191 | 0.6834 | 0.5855 | 0.8207 | 0.9911 | 354 |
| test | crackairport | 0.5266 | 0.6899 | 0.5897 | 0.8312 | 0.9914 | 337 |
| test | crackforest | 0.4110 | 0.5825 | 0.5175 | 0.6662 | 0.9853 | 17 |

## Common Test Comparison

Pixel metrics below use the common held-out `data/processed/yolo_seg_expanded` test split with masks resized to 512 px.

| Segmenter | IoU | Dice/F1 | Precision | Recall | Images |
| --- | ---: | ---: | ---: | ---: | ---: |
| dinov3_dense_head | 0.5191 | 0.6834 | 0.5855 | 0.8207 | 354 |
| classical | 0.0226 | 0.0441 | 0.0651 | 0.0334 | 354 |

Verdict: the DINOv3 dense head is the selected default because it is trained directly on pixel masks from CrackAirport and CrackForest and produces a full-resolution mask that keeps the existing area, length, and width measurement path intact. YOLO-seg-expanded remains the mobile/export model; the classical method remains the no-checkpoint fallback.

## Example Overlays

License-safe CrackAirport examples, left to right: original, ground-truth mask, learned prediction.

- `reports/examples/08_crack_seg_learned.png`
- `reports/examples/08_crack_seg_learned_2.png`
- `reports/examples/08_crack_seg_learned_3.png`

## Outputs

- `tarmac crack-measure <image|dir> --out DIR`: `<name>_crackseg.png`, `crack_measurements.csv`, `crack_measurements.parquet`.
- `tarmac analyze --region full --crack-segmentation`: `crackseg/frame_*_crackseg.png` plus geometry columns in `results.parquet`.
- `tarmac report`: crack geometry overlay gallery and measurement table.
