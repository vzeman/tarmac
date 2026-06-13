from __future__ import annotations

import shutil
from pathlib import Path

from tarmac.yolo.common import SEED, require_mps, seed_everything, write_json


def train_yolo_seg(
    data_yaml: Path = Path("data/processed/yolo_crack_seg/data.yaml"),
    model: str = "yolo11n-seg.pt",
    output_dir: Path = Path("models/yolo/crack_seg"),
    epochs: int = 100,
    patience: int = 20,
    imgsz: int = 512,
    batch: int = 8,
    device: str = "mps",
    seed: int = SEED,
) -> dict[str, object]:
    """Train a mobile YOLO11 segmentation student on real CrackAirport masks."""
    require_mps(device)
    seed_everything(seed)
    from ultralytics import YOLO

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    yolo = YOLO(model)
    results = yolo.train(
        data=str(data_yaml.resolve()),
        task="segment",
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
    best = output_dir / "weights" / "best.pt"
    metrics = YOLO(str(best)).val(data=str(data_yaml.resolve()), imgsz=imgsz, device="mps", split="test")
    payload = {
        "best_weights": str(best),
        "model": model,
        "epochs": epochs,
        "imgsz": imgsz,
        "box_map50": float(getattr(metrics.box, "map50", 0.0)),
        "box_map": float(getattr(metrics.box, "map", 0.0)),
        "mask_map50": float(getattr(metrics.seg, "map50", 0.0)),
        "mask_map": float(getattr(metrics.seg, "map", 0.0)),
        "results_dir": str(save_dir),
    }
    write_json(output_dir / "metrics.json", payload)
    return payload
