from __future__ import annotations

import shutil
from pathlib import Path

from tarmac.yolo.common import file_size_mb, write_json


def export_yolo(task: str, weights: Path | None = None, output_dir: Path = Path("models/yolo/export")) -> dict[str, object]:
    if task not in {"seg", "cls_type", "cls_quality"}:
        raise ValueError("task must be one of: seg, cls_type, cls_quality.")
    if weights is None:
        weights = {
            "seg": Path("models/yolo/crack_seg/weights/best.pt"),
            "cls_type": Path("models/yolo/cls_type/weights/best.pt"),
            "cls_quality": Path("models/yolo/cls_quality/weights/best.pt"),
        }[task]
    if not weights.exists():
        raise FileNotFoundError(f"YOLO weights not found: {weights}")
    from ultralytics import YOLO

    output_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    produced: dict[str, dict[str, object]] = {}
    errors: dict[str, str] = {}
    for fmt in ("onnx", "coreml"):
        try:
            exported = Path(model.export(format=fmt, imgsz=512 if task == "seg" else 224, device="cpu"))
            target = output_dir / task / exported.name
            target.parent.mkdir(parents=True, exist_ok=True)
            if exported.resolve() != target.resolve():
                if target.exists():
                    if target.is_dir():
                        shutil.rmtree(target)
                    else:
                        target.unlink()
                if exported.is_dir():
                    shutil.move(str(exported), str(target))
                else:
                    exported.replace(target)
            produced[fmt] = {"path": str(target), "size_mb": file_size_mb(target)}
        except Exception as exc:
            errors[fmt] = str(exc)
    payload = {"task": task, "weights": str(weights), "produced": produced, "errors": errors}
    write_json(output_dir / task / "export_metadata.json", payload)
    return payload
