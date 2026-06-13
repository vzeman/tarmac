# Crack Segmentation and Measurement

Phase 7c adds full-frame runway/pavement analysis and pixel-level crack geometry.

## Method

The current segmenter is classical hybrid segmentation, not a trained mask model:

1. Run the existing crack classifier head over an overlapping full-frame sliding grid to create a smoothed crack-probability prior.
2. Convert the original image to grayscale and extract dark thin ridges using `skimage` Frangi/Sato vesselness plus morphological black-hat enhancement.
3. Threshold only inside likely crack regions, clean small objects and holes, skeletonize the binary mask, and measure area, length, mean width, max width, and connected components.
4. Render actual crack-mask pixels as a semi-transparent red overlay on the original full image, with component outlines and area/length annotations.

Outputs:

- `tarmac crack-measure <image|dir> --out DIR`: `<name>_crackseg.png`, `crack_measurements.csv`, `crack_measurements.parquet`.
- `tarmac analyze --region full --crack-segmentation`: `crackseg/frame_*_crackseg.png` plus geometry columns in `results.parquet`.
- `tarmac report`: Crack geometry overlay gallery and measurement table.

## Learned Segmenter Status

No learned segmentation head was trained in this pass.

- `data/raw/crack500` and `data/raw/deepcrack` currently contain GitHub code mirrors, not the real image/mask datasets. Local scan found `85` image-like files and `0` mask-like files under `crack500`, and `2` image-like files and `0` mask-like files under `deepcrack`.
- `ROBOFLOW_API_KEY` was present, but Roboflow Universe does not provide a public API for arbitrary Universe dataset search; segmentation dataset discovery remains a manual/UI step before an export can be scripted. References: [Roboflow Universe dataset search docs](https://docs.roboflow.com/universe/find-a-dataset-on-universe), [Roboflow community answer on Universe search API](https://discuss.roboflow.com/t/is-it-possible-to-use-api-to-search-datasets-in-universe-roboflow-com/3625), and [Roboflow export docs](https://docs.roboflow.com/datasets/dataset-versions/exporting-data).
- Therefore `models/crack_seg_head.pt` was not produced, and Dice/IoU are not available.

When real masks are available, the next step is a frozen-DINO dense-token segmentation head with BCE+Dice loss on MPS only, checkpoint/resume, and held-out Dice/IoU reporting here.

## Runway Smoke Results

Command:

```bash
UV_CACHE_DIR=.uv-cache uv run tarmac crack-measure /tmp/tarmac_runway_test --out /tmp/crackseg_runway
```

| image | crack_area_pct | total_length_px |
|---|---:|---:|
| 00_noncracked_7bZbwGcaeRuiVafMZ3tJ_images.jpg | 0.9300 | 264 |
| 01_noncracked_SZZypQlarGgXvhgeuL6z_images_25_.jpg | 1.6843 | 274 |
| 02_cracked_ynkmCZTzQgrdhz0tlw5A_download.jpg | 1.1163 | 194 |
| 03_cracked_p6w6TRpjctUisWyFI21w_download_7_.jpg | 1.7011 | 380 |
| 04_cracked_TZ3vqoJClYFy95tDr6le_images_10_.jpg | 2.2390 | 264 |
| 05_cracked_laJ8ALH5RcO8WfWi0eId_images_14_.jpg | 0.0715 | 20 |
| 06_cracked_KMcFj8cPxO1bKxAwpEXF_images_16_.jpg | 1.1185 | 308 |
| 07_cracked_JYGoySlhQeIOgyXSnBXz_download_3_.jpg | 0.4431 | 131 |
| 08_cracked_IMtjcRNRh4pYqDjDiJcx_download_5_.jpg | 1.7613 | 351 |
| 09_cracked_H6DAgjSbYC68Qv9hHFWp_images_26_.jpg | 1.1726 | 280 |
| 10_cracked_HxCAWVVoeXB0vMRaJLIV_images_28_.jpg | 2.4524 | 510 |
| 11_cracked_qTJwW5FhWiIst0Us5JLY_images_23_.jpg | 0.4678 | 142 |

Filename-cracked group mean: `1.2544%` area, `258.0 px` length.

Filename-noncracked group mean: `1.3071%` area, `269.0 px` length.

Caveat: the two files named `noncracked` visibly contain large transverse crack-like dark defects, so they are not near-empty under pixel measurement. The smoke script prints this inconsistency instead of treating those names as reliable ground truth.

## Full-Frame Verification

`tarmac analyze /tmp/tarmac_runway_test/02_cracked_ynkmCZTzQgrdhz0tlw5A_download.jpg --region full --out /tmp/analyze_full_runway` produced:

- selected region: `full`
- tile rows: `9`
- first tile box: `[0, 0, 83, 68]`
- last tile box: `[165, 135, 248, 203]`
- crackseg overlay: `/tmp/analyze_full_runway/crackseg/frame_000000_02_cracked_ynkmCZTzQgrdhz0tlw5A_download_crackseg.png`

`--region auto` on the same runway image selected `full`.
