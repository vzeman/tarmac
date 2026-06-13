from __future__ import annotations

import shutil
from pathlib import Path

from tarmac.yolo.common import SEED, require_mps, seed_everything, write_json


def train_yolo_cls(
    target: str,
    model: str = "yolo11n-cls.pt",
    epochs: int = 50,
    patience: int = 10,
    imgsz: int = 224,
    batch: int = 32,
    device: str = "mps",
    seed: int = SEED,
    distill: bool = False,
) -> dict[str, object]:
    """Train YOLO11 classification students.

    DINOv3 remains the server-side teacher. The optional ``distill`` hook is
    intentionally off by default: when ``models/active_model.json`` exists it can
    be extended to pseudo-label unlabeled road tiles, increasing mobile-student
    coverage without converting DINO weights. YOLO is trained from labels (plus
    optional teacher labels), then exported to mobile runtimes.
    """
    if target not in {"type", "quality"}:
        raise ValueError("target must be 'type' or 'quality'.")
    require_mps(device)
    if distill and not Path("models/active_model.json").exists():
        raise RuntimeError("--distill requested, but models/active_model.json is not available.")
    seed_everything(seed)
    from ultralytics import YOLO

    data_dir = Path(f"data/processed/yolo_cls_{target}").resolve()
    output_dir = Path(f"models/yolo/cls_{target}").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best = output_dir / "weights" / "best.pt"
    save_dir = output_dir
    if epochs <= 0:
        if not best.exists():
            raise FileNotFoundError(f"--epochs 0 requires existing weights at {best}")
    else:
        yolo = YOLO(model)
        results = yolo.train(
            data=str(data_dir),
            task="classify",
            imgsz=imgsz,
            epochs=epochs,
            patience=patience,
            seed=seed,
            device="mps",
            batch=batch,
            project=str(output_dir.parent.resolve()),
            name=output_dir.name,
            exist_ok=True,
            deterministic=True,
        )
        save_dir = Path(getattr(results, "save_dir", output_dir)).resolve()
        if save_dir != output_dir and (save_dir / "weights").exists():
            shutil.copytree(save_dir / "weights", output_dir / "weights", dirs_exist_ok=True)
    metrics = YOLO(str(best)).val(data=str(data_dir), imgsz=imgsz, device="mps", split="test")
    top1 = float(getattr(metrics, "top1", 0.0))
    payload = {
        "target": target,
        "best_weights": str(best),
        "model": model,
        "epochs": epochs,
        "imgsz": imgsz,
        "top1": top1,
        "results_dir": str(save_dir),
    }
    if target == "quality":
        payload["off_by_one"] = float(_quality_off_by_one(best, data_dir, imgsz))
    write_json(output_dir / "metrics.json", payload)
    return payload


def _quality_off_by_one(weights: Path, data_dir: Path, imgsz: int) -> float:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    test_dir = data_dir / "test"
    images = sorted(p for p in test_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not images:
        return 0.0
    ok = 0
    results = model.predict(source=[str(p) for p in images], imgsz=imgsz, device="mps", verbose=False, stream=True)
    for image_path, result in zip(images, results, strict=True):
        true_q = _quality_from_path(image_path)
        if true_q is None:
            raise ValueError(f"Could not resolve quality label for source path: {image_path}")
        pred_name = result.names[int(result.probs.top1)]
        pred_q = int(str(pred_name).lstrip("q"))
        ok += int(abs(pred_q - true_q) <= 1)
    return ok / len(images)


def _quality_from_path(path: Path) -> int | None:
    for part in reversed(path.parts):
        if len(part) >= 2 and part[0] == "q" and part[1:].isdigit():
            return int(part[1:])
    return None
