from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from tarmac.datasets.rdd2022 import CLASSES, count_rdd_classes, voc_objects
from tarmac.yolo.common import SEED, write_json

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class RddPrepResult:
    output_dir: Path
    data_yaml: Path
    image_count: int
    label_count: int
    split_counts: dict[str, int]
    class_counts: dict[str, int]


def prepare_yolo_rdd(
    raw_dir: Path = Path("data/raw/rdd2022/Czech"),
    output_dir: Path = Path("data/processed/yolo_rdd"),
    seed: int = SEED,
) -> RddPrepResult:
    images_dir = raw_dir / "images"
    annotations_dir = raw_dir / "annotations"
    pairs = _annotation_image_pairs(images_dir, annotations_dir)
    if not pairs:
        raise RuntimeError(f"No RDD2022 image/XML pairs found under {raw_dir}.")

    split_map = _split_pairs(pairs, seed)
    if output_dir.exists():
        shutil.rmtree(output_dir)

    class_to_id = {name: index for index, name in enumerate(CLASSES)}
    label_count = 0
    class_counts = {name: 0 for name in CLASSES}
    manifest_rows: list[dict[str, object]] = []
    for split, split_pairs in split_map.items():
        (output_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (output_dir / "labels" / split).mkdir(parents=True, exist_ok=True)
        for index, (image_path, xml_path) in enumerate(tqdm(split_pairs, desc=f"yolo-rdd {split}")):
            stem = f"{image_path.stem}_{index:05d}"
            image_target = output_dir / "images" / split / f"{stem}{image_path.suffix.lower()}"
            label_target = output_dir / "labels" / split / f"{stem}.txt"
            shutil.copy2(image_path, image_target)
            lines, per_image_counts = _voc_to_yolo_lines(xml_path, image_path, class_to_id)
            label_target.write_text("\n".join(lines) + ("\n" if lines else ""))
            label_count += len(lines)
            for label, count in per_image_counts.items():
                class_counts[label] += count
            manifest_rows.append(
                {
                    "source_image": str(image_path.resolve()),
                    "source_annotation": str(xml_path.resolve()),
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
                *[f"  {index}: {name}" for index, name in enumerate(CLASSES)],
                "",
            ]
        )
    )
    (output_dir / "manifest.jsonl").write_text(
        "\n".join(json.dumps(row) for row in manifest_rows) + ("\n" if manifest_rows else "")
    )
    result = RddPrepResult(
        output_dir=output_dir,
        data_yaml=data_yaml,
        image_count=len(pairs),
        label_count=label_count,
        split_counts={split: len(items) for split, items in split_map.items()},
        class_counts=class_counts,
    )
    write_json(
        output_dir / "prep_metadata.json",
        {
            "output_dir": str(output_dir),
            "data_yaml": str(data_yaml),
            "raw_dir": str(raw_dir),
            "image_count": result.image_count,
            "label_count": result.label_count,
            "split_counts": result.split_counts,
            "class_counts": result.class_counts,
            "classes": list(CLASSES),
            "raw_class_counts": count_rdd_classes(annotations_dir),
        },
    )
    return result


def _annotation_image_pairs(images_dir: Path, annotations_dir: Path) -> list[tuple[Path, Path]]:
    images = {
        image_path.stem: image_path
        for image_path in sorted(images_dir.glob("*"))
        if image_path.suffix.lower() in IMAGE_EXTENSIONS
    }
    pairs = []
    for xml_path in sorted(annotations_dir.glob("*.xml")):
        image_path = images.get(xml_path.stem)
        if image_path is not None:
            pairs.append((image_path, xml_path))
    return pairs


def _split_pairs(
    pairs: list[tuple[Path, Path]],
    seed: int,
) -> dict[str, list[tuple[Path, Path]]]:
    import numpy as np

    rng = np.random.default_rng(seed)
    indexes = np.arange(len(pairs))
    rng.shuffle(indexes)
    train_end, val_end = _split_bounds(len(indexes))
    return {
        "train": [pairs[index] for index in indexes[:train_end]],
        "val": [pairs[index] for index in indexes[train_end:val_end]],
        "test": [pairs[index] for index in indexes[val_end:]],
    }


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


def _voc_to_yolo_lines(
    xml_path: Path,
    image_path: Path,
    class_to_id: dict[str, int],
) -> tuple[list[str], dict[str, int]]:
    width, height = _image_size(image_path)
    lines: list[str] = []
    counts = {name: 0 for name in CLASSES}
    for name, (xmin, ymin, xmax, ymax) in voc_objects(xml_path):
        if name not in class_to_id:
            continue
        xmin = min(max(xmin, 0), width)
        xmax = min(max(xmax, 0), width)
        ymin = min(max(ymin, 0), height)
        ymax = min(max(ymax, 0), height)
        box_width = xmax - xmin
        box_height = ymax - ymin
        if box_width <= 0 or box_height <= 0:
            continue
        x_center = xmin + box_width / 2.0
        y_center = ymin + box_height / 2.0
        lines.append(
            f"{class_to_id[name]} "
            f"{x_center / width:.6f} {y_center / height:.6f} "
            f"{box_width / width:.6f} {box_height / height:.6f}"
        )
        counts[name] += 1
    return lines, counts


def _image_size(image_path: Path) -> tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size
