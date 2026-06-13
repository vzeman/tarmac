from __future__ import annotations

from pathlib import Path


def main() -> None:
    required = [
        Path("data/processed/yolo_crack_seg/data.yaml"),
        Path("data/processed/yolo_cls_type/prep_metadata.json"),
        Path("data/processed/yolo_cls_quality/prep_metadata.json"),
        Path("models/yolo/crack_seg/weights/best.pt"),
        Path("models/yolo/cls_type/weights/best.pt"),
        Path("models/yolo/cls_quality/weights/best.pt"),
        Path("reports/yolo_benchmark.json"),
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise SystemExit("Missing YOLO smoke artifacts:\n" + "\n".join(missing))
    export_root = Path("models/yolo/export")
    if not list(export_root.rglob("*.onnx")):
        raise SystemExit("Missing at least one ONNX export under models/yolo/export.")
    detect_dir = Path("runs/yolo_detect/overlays")
    if not detect_dir.exists() or not list(detect_dir.glob("*_yolo_crackseg.png")):
        raise SystemExit("YOLO detect overlays were not produced.")
    print("YOLO smoke passed.")


if __name__ == "__main__":
    main()
