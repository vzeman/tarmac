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
from tarmac.datasets.crackforest import find_crackforest_pairs
from tarmac.yolo.common import SEED, write_json


@dataclass(frozen=True)
class SegPrepResult:
    output_dir: Path
    data_yaml: Path
    pair_count: int
    label_count: int
    empty_mask_count: int
    split_counts: dict[str, int]


@dataclass(frozen=True)
class ExpandedSegPrepResult:
    output_dir: Path
    data_yaml: Path
    pair_count: int
    label_count: int
    empty_mask_count: int
    split_counts: dict[str, int]
    source_counts: dict[str, int]
    split_source_counts: dict[str, dict[str, int]]


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


def prepare_yolo_seg_expanded(
    crackairport_dir: Path = Path("data/raw/crackairport"),
    crackforest_dir: Path = Path("data/raw/crackforest"),
    output_dir: Path = Path("data/processed/yolo_seg_expanded"),
    seed: int = SEED,
    keep_empty: bool = True,
) -> ExpandedSegPrepResult:
    """Convert CrackAirport and CrackForest masks into a combined YOLO segmentation dataset."""
    records: list[tuple[str, Path, Path]] = []
    records.extend(("crackairport", image, mask) for image, mask in _crackairport_pairs(crackairport_dir))
    records.extend(("crackforest", image, mask) for image, mask in find_crackforest_pairs(crackforest_dir))
    if not records:
        raise RuntimeError(
            f"No CrackAirport or CrackForest image/mask pairs found under {crackairport_dir} and {crackforest_dir}."
        )
    source_counts: dict[str, int] = {}
    for source, _image, _mask in records:
        source_counts[source] = source_counts.get(source, 0) + 1
    missing_sources = {"crackairport", "crackforest"} - set(source_counts)
    if missing_sources:
        raise RuntimeError(f"Missing required segmentation sources: {', '.join(sorted(missing_sources))}.")

    split_map = _split_records_by_source(records, seed)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    label_count = 0
    empty_count = 0
    manifest_rows: list[dict[str, object]] = []
    split_source_counts: dict[str, dict[str, int]] = {}
    for split, split_records in split_map.items():
        split_source_counts[split] = {}
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index, (source, image_path, mask_path) in enumerate(tqdm(split_records, desc=f"yolo-seg-expanded {split}")):
            split_source_counts[split][source] = split_source_counts[split].get(source, 0) + 1
            stem = f"{source}_{image_path.stem}_{index:05d}"
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
            manifest_rows.append(
                {
                    "source_dataset": source,
                    "source_image": str(image_path.resolve()),
                    "source_mask": str(mask_path.resolve()),
                    "image_path": str(image_target),
                    "label_path": str(label_target),
                    "split": split,
                    "label_count": len(lines),
                }
            )

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
    (output_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in manifest_rows) + ("\n" if manifest_rows else "")
    )
    result = ExpandedSegPrepResult(
        output_dir=output_dir,
        data_yaml=data_yaml,
        pair_count=len(records),
        label_count=label_count,
        empty_mask_count=empty_count,
        split_counts={split: len(items) for split, items in split_map.items()},
        source_counts=source_counts,
        split_source_counts=split_source_counts,
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
            "source_counts": result.source_counts,
            "split_source_counts": result.split_source_counts,
            "seed": seed,
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


def _crackairport_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists():
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        ]
    return find_crackairport_pairs(raw_dir / "_extracted")


def _split_records_by_source(
    records: list[tuple[str, Path, Path]],
    seed: int,
) -> dict[str, list[tuple[str, Path, Path]]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    split_map: dict[str, list[tuple[str, Path, Path]]] = {"train": [], "val": [], "test": []}
    by_source: dict[str, list[tuple[str, Path, Path]]] = {}
    for record in records:
        by_source.setdefault(record[0], []).append(record)
    for source in sorted(by_source):
        source_records = sorted(by_source[source], key=lambda item: str(item[1]))
        indexes = np.arange(len(source_records))
        rng.shuffle(indexes)
        train_end, val_end = _split_bounds(len(indexes))
        split_map["train"].extend(source_records[index] for index in indexes[:train_end])
        split_map["val"].extend(source_records[index] for index in indexes[train_end:val_end])
        split_map["test"].extend(source_records[index] for index in indexes[val_end:])
    for split in split_map:
        split_map[split].sort(key=lambda item: (item[0], str(item[1])))
    return split_map


def _split_bounds(count: int) -> tuple[int, int]:
    if count <= 0:
        return 0, 0
    if count == 1:
        return 1, 1
    if count == 2:
        return 1, 2
    train_count = max(1, int(round(0.70 * count)))
    val_count = max(1, int(round(0.15 * count)))
    if train_count + val_count >= count:
        train_count = max(1, count - 2)
        val_count = 1
    return train_count, train_count + val_count
