from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from tarmac.crack.segment import measure_crack_mask, render_crack_overlay
from tarmac.yolo.common import IMAGE_EXTENSIONS


def yolo_detect(
    path: Path,
    weights: Path = Path("models/yolo/crack_seg/weights/best.pt"),
    out_dir: Path = Path("runs/yolo_detect"),
    imgsz: int = 512,
    conf: float = 0.25,
    device: str = "cpu",
    mm_per_pixel: float | None = None,
) -> dict[str, object]:
    if not weights.exists():
        raise FileNotFoundError(f"YOLO segmentation weights not found: {weights}")
    from ultralytics import YOLO

    image_paths = _collect_images(path)
    if not image_paths:
        raise RuntimeError(f"No images found in {path}")
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(weights))
    rows: list[dict[str, object]] = []
    results = model.predict(source=[str(p) for p in image_paths], imgsz=imgsz, conf=conf, device=device, verbose=False, stream=True)
    for image_path, result in zip(image_paths, results, strict=True):
        with Image.open(image_path) as image:
            mask = _result_mask(result, image.height, image.width)
            measurements = measure_crack_mask(mask, mm_per_pixel=mm_per_pixel)
            overlay_path = overlay_dir / f"{image_path.stem}_yolo_crackseg.png"
            render_crack_overlay(image.convert("RGB"), mask, measurements, overlay_path)
        rows.append(
            {
                "image_path": str(image_path),
                "filename": image_path.name,
                "overlay_path": str(overlay_path),
                "has_crack": bool(measurements["crack_area_px"] > 0),
                **measurements,
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "yolo_detections.csv", index=False)
    df.to_parquet(out_dir / "yolo_detections.parquet", index=False)
    summary = {
        "input": str(path),
        "images": len(rows),
        "cracked": int(df["has_crack"].sum()) if not df.empty else 0,
        "mean_crack_area_pct": float(df["crack_area_pct"].mean()) if not df.empty else 0.0,
        "output_dir": str(out_dir),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def _collect_images(path: Path) -> list[Path]:
    path = path.expanduser().resolve()
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
        return [path]
    return []


def _result_mask(result: object, height: int, width: int) -> np.ndarray:
    masks = getattr(result, "masks", None)
    if masks is None or masks.data is None:
        return np.zeros((height, width), dtype=bool)
    data = masks.data.detach().cpu().numpy()
    if data.size == 0:
        return np.zeros((height, width), dtype=bool)
    merged = np.any(data > 0.5, axis=0).astype(np.uint8)
    from cv2 import INTER_NEAREST, resize

    return resize(merged, (width, height), interpolation=INTER_NEAREST).astype(bool)
