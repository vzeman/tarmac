# YOLO Mobile Track

DINOv3 remains the high-accuracy server-side teacher. These YOLO11 students are trained on labels, with an optional distillation hook, and exported for mobile runtimes rather than converted from DINOv3 weights.

| Model | Params | Size MB | Metric | CPU ms | MPS ms | FPS CPU/MPS | Export sizes | Mobile suitability |
| --- | ---: | ---: | --- | ---: | ---: | --- | --- | --- |
| crack_seg | 2,842,803 | 5.7 | mask mAP50 0.041, mAP50-95 0.009 | 21.3 | 3.4 | 46.9/295.1 | best.onnx: 11.0 MB | near-real-time target; validate on-device |
| cls_type | 1,537,509 | 3.0 | top1 0.803 | 4.0 | 1.9 | 248.9/519.4 | best.onnx: 5.9 MB | near-real-time target; validate on-device |
| cls_quality | 1,537,509 | 3.0 | top1 0.479, off-by-one 0.929 | 4.2 | 1.8 | 240.4/559.8 | best.onnx: 5.9 MB | near-real-time target; validate on-device |

## Export Caveats

- crack_seg: coreml: No module named 'coremltools', tflite: No module named 'tensorflow'
- cls_type: coreml: No module named 'coremltools', tflite: No module named 'tensorflow'
- cls_quality: coreml: No module named 'coremltools', tflite: No module named 'tensorflow'
