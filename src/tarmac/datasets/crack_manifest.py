from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SEED = 42


@dataclass(frozen=True)
class CrackManifestResult:
    path: Path
    row_count: int
    stats: pd.DataFrame


def build_crack_manifest(
    raw_dir: Path = Path("data/raw"),
    output_path: Path = Path("data/processed/crack_manifest.parquet"),
) -> CrackManifestResult:
    rows: list[dict[str, object]] = []
    rows.extend(_concrete_pavement_rows(raw_dir / "cracks_concrete_pavement"))
    rows.extend(_folder_binary_rows(raw_dir / "crack500", "crack500"))
    rows.extend(_folder_binary_rows(raw_dir / "deepcrack", "deepcrack"))
    rows.extend(_runway_rows(raw_dir / "runway_roboflow"))
    rows.extend(_khanh11k_rows(raw_dir / "khanh11k"))
    rows.extend(_crack500_seg_rows(raw_dir / "crack500_seg"))
    rows.extend(_mendeley5y9_rows(raw_dir / "mendeley5y9"))
    rows.extend(_rdd2022_rows(raw_dir / "rdd2022"))
    rows.extend(_seg_pairs_binary_rows(raw_dir / "metu_crack_seg", "metu_crack_seg"))
    rows.extend(_seg_pairs_binary_rows(raw_dir / "find_crack", "find_crack"))
    rows.extend(_seg_pairs_binary_rows(raw_dir / "masonry_crack", "masonry_crack"))
    rows.extend(_seg_pairs_binary_rows(raw_dir / "hf_crack", "hf_crack"))
    rows.extend(_seg_pairs_binary_rows(raw_dir / "paggnet_crack", "paggnet_crack"))
    if not rows:
        raise RuntimeError(
            f"No crack datasets found under {raw_dir}. Run `uv run tarmac download cracks-concrete-pavement` first."
        )

    frame = pd.DataFrame(rows).drop_duplicates(subset=["image_path", "source_dataset", "tile"]).reset_index(drop=True)
    frame["has_crack"] = frame["has_crack"].astype("int8")
    frame["split"] = _stratified_source_splits(frame, seed=SEED)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    stats = (
        frame.groupby(["source_dataset", "split", "has_crack"], observed=True)
        .size()
        .reset_index(name="count")
        .sort_values(["source_dataset", "split", "has_crack"])
    )
    return CrackManifestResult(path=output_path, row_count=len(frame), stats=stats)


def _concrete_pavement_rows(dataset_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for folder, label in (("positive", 1), ("negative", 0)):
        path = dataset_dir / folder
        if not path.exists():
            continue
        for image_path in sorted(p for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "source_dataset": "cracks_concrete_pavement",
                    "tile": "full",
                    "has_crack": label,
                }
            )
    return rows


def _folder_binary_rows(dataset_dir: Path, source_dataset: str) -> list[dict[str, object]]:
    if not dataset_dir.exists():
        return []
    rows: list[dict[str, object]] = []
    for image_path in sorted(p for p in dataset_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        parts = [part.lower() for part in image_path.parts]
        if any("mask" in part or "label" in part for part in parts):
            continue
        label = 1 if any("crack" in part for part in parts) else None
        if label is None:
            continue
        rows.append(
            {
                "image_path": str(image_path.resolve()),
                "source_dataset": source_dataset,
                "tile": "full",
                "has_crack": label,
            }
        )
    return rows


def _runway_rows(dataset_dir: Path) -> list[dict[str, object]]:
    labels_path = dataset_dir / "tile_labels.parquet"
    if not labels_path.exists():
        labels_path = dataset_dir / "tile_labels.csv"
    if not labels_path.exists():
        labels_path = dataset_dir / "tile_labels.jsonl"
    if not labels_path.exists():
        return []
    rows: list[dict[str, object]] = []
    if labels_path.suffix == ".parquet":
        records = pd.read_parquet(labels_path).to_dict("records")
    elif labels_path.suffix == ".csv":
        records = pd.read_csv(labels_path).to_dict("records")
    else:
        records = [json.loads(line) for line in labels_path.read_text().splitlines() if line.strip()]
    for row in records:
        rows.append(
            {
                "image_path": str(Path(row["image_path"]).resolve()),
                "source_dataset": "runway_roboflow",
                "tile": str(row.get("tile", "full")),
                "has_crack": int(row["has_crack"]),
            }
        )
    return rows


def _mendeley5y9_rows(dataset_dir: Path) -> list[dict[str, object]]:
    from tarmac.datasets.mendeley5y9 import find_mendeley5y9_images
    images_by_label = find_mendeley5y9_images(dataset_dir)
    rows: list[dict[str, object]] = []
    for label_name, label_value in (("positive", 1), ("negative", 0)):
        for image_path in images_by_label.get(label_name, []):
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "source_dataset": "mendeley5y9",
                    "tile": "full",
                    "has_crack": label_value,
                }
            )
    return rows


def _khanh11k_rows(dataset_dir: Path) -> list[dict[str, object]]:
    return _seg_pairs_binary_rows(dataset_dir, "khanh11k")


def _crack500_seg_rows(dataset_dir: Path) -> list[dict[str, object]]:
    return _seg_pairs_binary_rows(dataset_dir, "crack500_seg")


def _seg_pairs_binary_rows(dataset_dir: Path, source_dataset: str) -> list[dict[str, object]]:
    """Binary has_crack=1 rows from any dataset that writes a pairs.jsonl index."""
    pairs_path = dataset_dir / "pairs.jsonl"
    if not pairs_path.exists() or pairs_path.stat().st_size <= 10:
        return []
    rows: list[dict[str, object]] = []
    for line in pairs_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        rows.append(
            {
                "image_path": row["image_path"],
                "source_dataset": source_dataset,
                "tile": "full",
                "has_crack": 1,
            }
        )
    return rows


def _rdd2022_rows(rdd2022_dir: Path) -> list[dict[str, object]]:
    """Binary rows from all downloaded RDD2022 country subdirectories (all annotated = has_crack=1)."""
    from tarmac.datasets.rdd2022 import find_rdd2022_country_dirs
    rows: list[dict[str, object]] = []
    if not rdd2022_dir.exists():
        return rows
    for country_dir in find_rdd2022_country_dirs(rdd2022_dir):
        images_dir = country_dir / "images"
        for image_path in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "source_dataset": f"rdd2022_{country_dir.name.lower()}",
                    "tile": "full",
                    "has_crack": 1,
                }
            )
    return rows


def _stratified_source_splits(frame: pd.DataFrame, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    splits = np.empty(len(frame), dtype=object)
    for (_source, _label), group in frame.groupby(["source_dataset", "has_crack"], sort=True, observed=True):
        indexes = group.index.to_numpy().copy()
        rng.shuffle(indexes)
        train_end, val_end = _split_bounds(len(indexes))
        splits[indexes[:train_end]] = "train"
        splits[indexes[train_end:val_end]] = "val"
        splits[indexes[val_end:]] = "test"
    return [str(x) for x in splits]


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


def _stratified_splits(labels: np.ndarray, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    splits = np.empty(len(labels), dtype=object)
    for label in sorted(set(int(x) for x in labels)):
        indexes = np.flatnonzero(labels == label)
        rng.shuffle(indexes)
        train_end = int(round(0.70 * len(indexes)))
        val_end = train_end + int(round(0.15 * len(indexes)))
        splits[indexes[:train_end]] = "train"
        splits[indexes[train_end:val_end]] = "val"
        splits[indexes[val_end:]] = "test"
    return [str(x) for x in splits]
