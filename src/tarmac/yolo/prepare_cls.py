from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from PIL import Image
from tqdm import tqdm

from tarmac.embedding.tiling import tile_boxes
from tarmac.yolo.common import write_json


@dataclass(frozen=True)
class ClsPrepResult:
    type_dir: Path
    quality_dir: Path
    tile_count: int
    type_classes: list[str]
    quality_classes: list[str]


def prepare_yolo_cls(
    manifest_path: Path = Path("data/processed/manifest.parquet"),
    type_dir: Path = Path("data/processed/yolo_cls_type"),
    quality_dir: Path = Path("data/processed/yolo_cls_quality"),
    tile_size: int = 224,
) -> ClsPrepResult:
    """Build Ultralytics ImageFolder datasets from lower-half 3x2 road tiles."""
    manifest = pd.read_parquet(manifest_path)
    required = {"image_path", "surface_type", "quality", "split"}
    missing = required - set(manifest.columns)
    if missing:
        raise RuntimeError(f"Manifest is missing required columns: {sorted(missing)}")
    type_classes = sorted(str(value) for value in manifest["surface_type"].dropna().unique())
    quality_classes = [f"q{i}" for i in range(1, 6)]
    tile_count = 0

    for row in tqdm(manifest.to_dict("records"), desc="yolo-cls tiles"):
        image_path = Path(str(row["image_path"]))
        if not image_path.exists():
            continue
        split = str(row["split"])
        surface_type = str(row["surface_type"])
        quality = f"q{int(row['quality'])}"
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            boxes = tile_boxes(rgb.width, rgb.height, tile_cols=3, tile_rows=2, region="lower_half")
            for tile_index, box in enumerate(boxes):
                tile = rgb.crop(box).resize((tile_size, tile_size))
                name = f"{image_path.stem}_tile{tile_index:02d}.jpg"
                type_target = type_dir / split / surface_type / name
                quality_target = quality_dir / split / quality / name
                type_target.parent.mkdir(parents=True, exist_ok=True)
                quality_target.parent.mkdir(parents=True, exist_ok=True)
                tile.save(type_target, quality=92)
                if quality_target.resolve() != type_target.resolve():
                    shutil.copy2(type_target, quality_target)
                tile_count += 1

    payload = {
        "tile_count": tile_count,
        "type_classes": type_classes,
        "quality_classes": quality_classes,
        "source_manifest": str(manifest_path),
        "tiling": "lower_half 3x2",
    }
    write_json(type_dir / "prep_metadata.json", payload)
    write_json(quality_dir / "prep_metadata.json", payload)
    return ClsPrepResult(type_dir, quality_dir, tile_count, type_classes, quality_classes)
