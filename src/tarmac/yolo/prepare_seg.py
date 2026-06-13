from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from tarmac.datasets.crackairport import find_crackairport_pairs
from tarmac.yolo.common import SEED, write_json


@dataclass(frozen=True)
class SegPrepResult:
    output_dir: Path
    data_yaml: Path
    pair_count: int
    label_count: int
    empty_mask_count: int
    split_counts: dict[str, int]


def prepare_yolo_seg(
    raw_dir: Path = Path("data/raw/crackairport"),
    output_dir: Path = Path("data/processed/yolo_crack_seg"),
    seed: int = SEED,
    keep_empty: bool = True,
) -> SegPrepResult:
    """Convert CrackAirport masks into Ultralytics YOLO segmentation polygons."""
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists():
        pairs = [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        ]
    else:
        pairs = find_crackairport_pairs(raw_dir / "_extracted")
    if not pairs:
        raise RuntimeError(f"No CrackAirport image/mask pairs found under {raw_dir}.")

    train_pairs, temp_pairs = train_test_split(pairs, train_size=0.70, random_state=seed)
    val_pairs, test_pairs = train_test_split(temp_pairs, train_size=0.50, random_state=seed)
    split_map = {"train": train_pairs, "val": val_pairs, "test": test_pairs}

    label_count = 0
    empty_count = 0
    for split, split_pairs in split_map.items():
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index, (image_path, mask_path) in enumerate(tqdm(split_pairs, desc=f"yolo-seg {split}")):
            stem = f"{image_path.stem}_{index:05d}"
            image_target = output_dir / "images" / split / f"{stem}{image_path.suffix.lower()}"
            label_target = output_dir / "labels" / split / f"{stem}.txt"
            shutil.copy2(image_path, image_target)
            lines = mask_to_yolo_polygons(mask_path)
            if not lines:
                empty_count += 1
                if not keep_empty:
                    image_target.unlink(missing_ok=True)
                    continue
            label_target.write_text("\n".join(lines) + ("\n" if lines else ""))
            label_count += len(lines)

    data_yaml = output_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {output_dir.resolve()}",
                "train: images/train",
                "val: images/val",
                "test: images/test",
                "names:",
                "  0: crack",
                "",
            ]
        )
    )
    result = SegPrepResult(
        output_dir=output_dir,
        data_yaml=data_yaml,
        pair_count=len(pairs),
        label_count=label_count,
        empty_mask_count=empty_count,
        split_counts={split: len(items) for split, items in split_map.items()},
    )
    write_json(
        output_dir / "prep_metadata.json",
        {
            "output_dir": str(output_dir),
            "data_yaml": str(data_yaml),
            "pair_count": result.pair_count,
            "label_count": result.label_count,
            "empty_mask_count": result.empty_mask_count,
            "split_counts": result.split_counts,
        },
    )
    return result


def mask_to_yolo_polygons(mask_path: Path, min_area: float = 6.0) -> list[str]:
    with Image.open(mask_path) as image:
        gray = np.asarray(image.convert("L"))
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    height, width = binary.shape[:2]
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines: list[str] = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        epsilon = max(1.0, 0.0025 * cv2.arcLength(contour, True))
        approx = cv2.approxPolyDP(contour, epsilon, True).reshape(-1, 2)
        if len(approx) < 3:
            continue
        coords: list[str] = []
        for x, y in approx:
            coords.append(f"{min(max(float(x) / width, 0.0), 1.0):.6f}")
            coords.append(f"{min(max(float(y) / height, 0.0), 1.0):.6f}")
        lines.append("0 " + " ".join(coords))
    return lines
