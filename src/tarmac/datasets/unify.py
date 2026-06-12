from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

QUALITY_MAP = {
    "excellent": 1,
    "good": 2,
    "intermediate": 3,
    "bad": 4,
    "very bad": 5,
    "very_bad": 5,
    "verybad": 5,
}
SURFACE_MAP = {
    "asphalt": "asphalt",
    "concrete": "concrete",
    "paving stones": "paving_stones",
    "paving_stones": "paving_stones",
    "sett": "sett",
    "gravel": "gravel",
    "mud": "mud",
    "unpaved": "unpaved",
}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


@dataclass(frozen=True)
class ManifestResult:
    path: Path
    row_count: int
    stats: pd.DataFrame


def _image_index(root: Path) -> dict[str, Path]:
    images: dict[str, Path] = {}
    for path in root.rglob("*"):
        if path.suffix.lower() in IMAGE_SUFFIXES:
            images[path.stem] = path
    return images


def _normalise_surface(value: object) -> str:
    key = str(value).strip().lower().replace("-", "_")
    if key not in SURFACE_MAP:
        raise ValueError(f"Unsupported surface_type: {value!r}")
    return SURFACE_MAP[key]


def _normalise_quality(value: object) -> int:
    if pd.isna(value):
        raise ValueError("Missing surface quality.")
    if isinstance(value, int):
        quality = value
    else:
        key = str(value).strip().lower().replace("-", " ")
        quality = QUALITY_MAP.get(key)
    if quality not in {1, 2, 3, 4, 5}:
        raise ValueError(f"Unsupported quality: {value!r}")
    return int(quality)


def _split_without_recommended_split(df: pd.DataFrame) -> pd.DataFrame:
    labels = df["surface_type"] + "_" + df["quality"].astype(str)
    train_df, temp_df = train_test_split(
        df,
        train_size=0.70,
        random_state=42,
        stratify=labels,
    )
    temp_labels = temp_df["surface_type"] + "_" + temp_df["quality"].astype(str)
    val_df, test_df = train_test_split(
        temp_df,
        train_size=0.50,
        random_state=42,
        stratify=temp_labels,
    )
    train_df = train_df.assign(split="train")
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], ignore_index=True)


def _split_from_train_flag(df: pd.DataFrame) -> pd.DataFrame:
    train_mask = df["train"].astype(str).str.lower().isin({"true", "1", "yes"})
    train_df = df.loc[train_mask].copy().assign(split="train")
    holdout_df = df.loc[~train_mask].copy()
    if holdout_df.empty:
        return _split_without_recommended_split(df.drop(columns=["train"]))

    labels = holdout_df["surface_type"] + "_" + holdout_df["quality"].astype(str)
    counts = labels.value_counts()
    if counts.min() >= 2:
        val_df, test_df = train_test_split(
            holdout_df,
            train_size=0.50,
            random_state=42,
            stratify=labels,
        )
    else:
        val_df, test_df = train_test_split(
            holdout_df,
            train_size=0.50,
            random_state=42,
        )
    val_df = val_df.assign(split="val")
    test_df = test_df.assign(split="test")
    return pd.concat([train_df, val_df, test_df], ignore_index=True).drop(columns=["train"])


def _streetsurfacevis_manifest(root: Path) -> pd.DataFrame:
    csv_path = root / "streetSurfaceVis_v1_0.csv"
    if not csv_path.exists():
        return pd.DataFrame()

    images = _image_index(root)
    source = pd.read_csv(csv_path)
    rows: list[dict[str, object]] = []
    missing = 0
    for record in source.to_dict("records"):
        image_id = str(record["mapillary_image_id"])
        image_path = images.get(image_id)
        if image_path is None:
            missing += 1
            continue
        rows.append(
            {
                "image_path": str(image_path),
                "source_dataset": "streetsurfacevis",
                "surface_type": _normalise_surface(record["surface_type"]),
                "quality": _normalise_quality(record["surface_quality"]),
                "train": record.get("train"),
            }
        )

    if missing:
        print(f"StreetSurfaceVis: skipped {missing} CSV rows without matching images.")

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if "train" in source.columns:
        return _split_from_train_flag(df)
    return _split_without_recommended_split(df.drop(columns=["train"]))


def build_manifest(
    raw_dir: Path = Path("data/raw"),
    output_path: Path = Path("data/processed/manifest.parquet"),
) -> ManifestResult:
    """Build a unified manifest from datasets that are present on disk."""
    frames = [_streetsurfacevis_manifest(raw_dir / "streetsurfacevis")]
    frames = [frame for frame in frames if not frame.empty]
    if not frames:
        raise RuntimeError(f"No supported datasets found under {raw_dir}.")

    manifest = pd.concat(frames, ignore_index=True)
    manifest = manifest[["image_path", "source_dataset", "surface_type", "quality", "split"]]
    manifest["quality"] = manifest["quality"].astype("int8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(output_path, index=False)

    stats = (
        manifest.groupby(["surface_type", "quality", "split"], observed=True)
        .size()
        .reset_index(name="rows")
        .sort_values(["surface_type", "quality", "split"])
    )
    return ManifestResult(path=output_path, row_count=len(manifest), stats=stats)
