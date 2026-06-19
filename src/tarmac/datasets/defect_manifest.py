from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from tarmac.datasets.codebrim import build_codebrim_annotations

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_VOCAB = ["crack", "spalling", "efflorescence", "exposed_rebar", "corrosion", "none"]
SEED = 42


@dataclass(frozen=True)
class DefectManifestResult:
    path: Path
    row_count: int
    source_domain_stats: pd.DataFrame
    label_totals: pd.DataFrame


def build_defect_manifest(
    raw_dir: Path = Path("data/raw"),
    output_path: Path = Path("data/processed/defect_manifest.parquet"),
) -> DefectManifestResult:
    """Build a unified multi-domain, multi-label structural-defect manifest."""
    rows: list[dict[str, object]] = []
    rows.extend(_codebrim_rows(raw_dir / "codebrim"))
    rows.extend(_sdnet2018_rows(raw_dir / "sdnet2018"))
    rows.extend(_concrete_pavement_rows(raw_dir / "cracks_concrete_pavement"))
    rows.extend(_crackairport_rows(raw_dir / "crackairport"))
    rows.extend(_rdd2022_defect_rows(raw_dir / "rdd2022"))
    if not rows:
        raise RuntimeError(
            f"No supported defect datasets found under {raw_dir}. Run CODEBRIM, SDNET2018, "
            "CrackAirport, or concrete/pavement crack downloaders first."
        )

    frame = pd.DataFrame(rows).drop_duplicates(subset=["image_path", "source_dataset"]).reset_index(drop=True)
    frame["labels"] = frame["labels"].map(_normalise_labels)
    frame["has_crack"] = frame["labels"].map(lambda labels: int("crack" in labels)).astype("int8")
    frame["split"] = _stratified_source_crack_splits(frame, seed=SEED)
    frame = frame[
        [
            "image_path",
            "source_dataset",
            "domain",
            "structure_material",
            "labels",
            "has_crack",
            "split",
        ]
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_path, index=False)
    source_domain_stats = summary_by_source_domain(frame)
    label_totals = label_total_table(frame)
    return DefectManifestResult(
        path=output_path,
        row_count=len(frame),
        source_domain_stats=source_domain_stats,
        label_totals=label_totals,
    )


def summary_by_source_domain(frame: pd.DataFrame) -> pd.DataFrame:
    return (
        frame.groupby(["source_dataset", "domain", "has_crack"], observed=True)
        .size()
        .reset_index(name="rows")
        .sort_values(["source_dataset", "domain", "has_crack"])
        .reset_index(drop=True)
    )


def label_total_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in LABEL_VOCAB:
        rows.append({"label": label, "rows": int(frame["labels"].map(lambda labels: label in labels).sum())})
    return pd.DataFrame(rows)


def _codebrim_rows(dataset_dir: Path) -> list[dict[str, object]]:
    if not dataset_dir.exists():
        return []
    annotations_path = dataset_dir / "annotations.parquet"
    if annotations_path.exists():
        annotations = pd.read_parquet(annotations_path)
    else:
        extracted_roots = list((dataset_dir / "_extracted").rglob("metadata"))
        if not extracted_roots:
            return []
        annotations = build_codebrim_annotations(extracted_roots[0].parent)
        annotations.to_parquet(annotations_path, index=False)
    rows: list[dict[str, object]] = []
    for record in annotations.to_dict("records"):
        labels = _codebrim_to_unified(record.get("codebrim_labels"))
        rows.append(
            {
                "image_path": str(Path(record["image_path"]).resolve()),
                "source_dataset": "codebrim",
                "domain": "bridge",
                "structure_material": "concrete",
                "labels": labels,
            }
        )
    return rows


def _sdnet2018_rows(dataset_dir: Path) -> list[dict[str, object]]:
    domain_map = {"D": "bridge", "W": "building", "P": "pavement"}
    rows: list[dict[str, object]] = []
    for code, domain in domain_map.items():
        for label_dir, labels in (("cracked", ["crack"]), ("uncracked", ["none"])):
            root = dataset_dir / code / label_dir
            if not root.exists():
                continue
            for image_path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
                rows.append(
                    {
                        "image_path": str(image_path.resolve()),
                        "source_dataset": "sdnet2018",
                        "domain": domain,
                        "structure_material": "concrete",
                        "labels": labels,
                    }
                )
    return rows


def _concrete_pavement_rows(dataset_dir: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for folder, labels in (("positive", ["crack"]), ("negative", ["none"])):
        root = dataset_dir / folder
        if not root.exists():
            continue
        for image_path in sorted(p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "source_dataset": "cracks_concrete_pavement",
                    "domain": "concrete_generic",
                    "structure_material": "concrete",
                    "labels": labels,
                }
            )
    return rows


def _crackairport_rows(dataset_dir: Path) -> list[dict[str, object]]:
    pairs_path = dataset_dir / "pairs.jsonl"
    if not pairs_path.exists():
        return []
    rows: list[dict[str, object]] = []
    for line in pairs_path.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        image_path = Path(record["image_path"]).resolve()
        mask_path = Path(record["mask_path"]).resolve()
        labels = ["crack"] if _mask_has_foreground(mask_path) else ["none"]
        rows.append(
            {
                "image_path": str(image_path),
                "source_dataset": "crackairport",
                "domain": "runway",
                "structure_material": "asphalt",
                "labels": labels,
            }
        )
    return rows


def _rdd2022_defect_rows(rdd2022_dir: Path) -> list[dict[str, object]]:
    """Multi-label rows from all downloaded RDD2022 country subdirectories.

    D00/D10/D20 (longitudinal/transverse/alligator crack) → 'crack'.
    D40 (pothole) → 'crack' (pavement surface failure, closest vocab match).
    """
    from tarmac.datasets.rdd2022 import find_rdd2022_country_dirs, rdd2022_image_labels
    rows: list[dict[str, object]] = []
    if not rdd2022_dir.exists():
        return rows
    for country_dir in find_rdd2022_country_dirs(rdd2022_dir):
        images_dir = country_dir / "images"
        labels_by_stem = rdd2022_image_labels(country_dir / "annotations")
        for image_path in sorted(p for p in images_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
            raw_labels = labels_by_stem.get(image_path.stem, ["crack"])
            unified = ["crack"] if any(label in ("crack", "pothole") for label in raw_labels) else ["none"]
            rows.append(
                {
                    "image_path": str(image_path.resolve()),
                    "source_dataset": f"rdd2022_{country_dir.name.lower()}",
                    "domain": "pavement",
                    "structure_material": "asphalt",
                    "labels": unified,
                }
            )
    return rows


def _mask_has_foreground(mask_path: Path) -> bool:
    with Image.open(mask_path) as image:
        extrema = image.convert("L").getextrema()
    return bool(extrema and extrema[1] > 0)


def _codebrim_to_unified(labels: object) -> list[str]:
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    if isinstance(labels, str):
        labels = [labels]
    label_set = {str(label) for label in labels}
    out: set[str] = set()
    mapping = {
        "crack": "crack",
        "spallation": "spalling",
        "spalling": "spalling",
        "efflorescence": "efflorescence",
        "exposed_bars": "exposed_rebar",
        "exposed_rebar": "exposed_rebar",
        "corrosion_stain": "corrosion",
        "corrosion": "corrosion",
    }
    for label in label_set:
        mapped = mapping.get(label)
        if mapped:
            out.add(mapped)
    if not out:
        out.add("none")
    return sorted(out, key=LABEL_VOCAB.index)


def _normalise_labels(labels: object) -> list[str]:
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    if isinstance(labels, str):
        labels = [labels]
    normalised = sorted({str(label) for label in labels}, key=lambda label: LABEL_VOCAB.index(label))
    unknown = set(normalised) - set(LABEL_VOCAB)
    if unknown:
        raise ValueError(f"Unknown defect labels: {sorted(unknown)}")
    if "none" in normalised and len(normalised) > 1:
        normalised = [label for label in normalised if label != "none"]
    return normalised or ["none"]


def _stratified_source_crack_splits(frame: pd.DataFrame, seed: int) -> list[str]:
    rng = np.random.default_rng(seed)
    splits = np.empty(len(frame), dtype=object)
    for (_source, _has_crack), group in frame.groupby(["source_dataset", "has_crack"], sort=True, observed=True):
        indexes = group.index.to_numpy().copy()
        rng.shuffle(indexes)
        train_end, val_end = _split_bounds(len(indexes))
        splits[indexes[:train_end]] = "train"
        splits[indexes[train_end:val_end]] = "val"
        splits[indexes[val_end:]] = "test"
    return [str(split) for split in splits]


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
