# YOLO Mobile Track

Full-training results, generated 2026-06-13 on Apple M3 Max using MPS and seed 42. DINOv3 remains the high-accuracy server-side teacher. These YOLO11 students are trained on labels, with an optional distillation hook, and exported for mobile runtimes rather than converted from DINOv3 weights.

## Final Model Selection

| Track | Candidate | Test metric | Decision |
| --- | --- | --- | --- |
| Crack segmentation | YOLO11n-seg, 200 epochs, imgsz 512 | mask mAP50 `0.1853`, mAP50-95 `0.0389` | selected default |
| Crack segmentation | YOLO11s-seg, 200 epochs, imgsz 512 | mask mAP50 `0.1536`, mAP50-95 `0.0293` | retained as higher-capacity artifact, not default |
| Surface type | YOLO11n-cls, 100 epochs, imgsz 224 | top-1 `0.8436` vs DINOv3 `0.954` | selected |
| Quality | YOLO11n-cls, 100 epochs, imgsz 224 | top-1 `0.5064`, off-by-one `0.9390` vs DINOv3 off-by-one `0.999` | below threshold |
| Quality | YOLO11s-cls, 100 epochs, imgsz 224 | top-1 `0.5026`, off-by-one `0.9459` vs DINOv3 off-by-one `0.999` | selected as better quality model |

| Model | Params | Size MB | Metric | CPU ms | MPS ms | FPS CPU/MPS | Export sizes | Mobile suitability |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| crack_seg | 2,842,803 | 5.7 | mask mAP50 0.185, mAP50-95 0.039 | 18.6 | 5.5 | 53.7/182.9 | best.onnx: 11.0 MB | near-real-time target; validate on-device |
| cls_type | 1,537,509 | 3.1 | top1 0.844 | 3.7 | 1.5 | 272.1/651.5 | best.onnx: 5.9 MB | near-real-time target; validate on-device |
| cls_quality | 5,449,413 | 10.5 | top1 0.503, off-by-one 0.946 | 5.3 | 1.5 | 189.9/658.5 | best.onnx: 20.8 MB | near-real-time target; validate on-device |

## Runway Smoke Inference

`tarmac yolo-detect /tmp/tarmac_runway_test --out runs/yolo_detect_final --device mps` correctly handled 10 of 12 held-out runway sample images: 9 of 10 cracked images were flagged and 1 of 2 non-cracked images was rejected. The misses were one false positive non-cracked runway image and one false negative cracked runway image.

Example overlays:

- `runs/yolo_detect_final/overlays/00_noncracked_7bZbwGcaeRuiVafMZ3tJ_images_yolo_crackseg.png`
- `runs/yolo_detect_final/overlays/02_cracked_ynkmCZTzQgrdhz0tlw5A_download_yolo_crackseg.png`
- `runs/yolo_detect_final/overlays/03_cracked_p6w6TRpjctUisWyFI21w_download_7__yolo_crackseg.png`
- `runs/yolo_detect_final/overlays/10_cracked_HxCAWVVoeXB0vMRaJLIV_images_28__yolo_crackseg.png`

## Export Caveats

- crack_seg: coreml: only 0-dimensional arrays can be converted to Python scalars
- cls_type: coreml: only 0-dimensional arrays can be converted to Python scalars
- cls_quality: coreml: only 0-dimensional arrays can be converted to Python scalars

ONNX exports are available for Android/edge deployment through ONNX Runtime Mobile. TensorFlow/TFLite export was intentionally not attempted on this Mac because the TensorFlow toolchain is heavy and fragile here; ONNX covers the Android path. CoreML export was enabled by installing `coremltools`, but Ultralytics/CoreML conversion failed with the error above for all three models, so no iOS `.mlpackage` is available from this run.

## Caveats

CrackAirport includes many empty-mask/background images and sparse crack labels; the final segmentation mAP is materially better than the 5-epoch smoke run but still modest. The YOLO mobile classifiers are fast and compact, but they remain below the fine-tuned DINOv3 teacher on type accuracy and quality off-by-one.
