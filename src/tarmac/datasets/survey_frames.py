"""Import video survey run frames into a parquet manifest for use in the labeling UI."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

_SURVEY_FRAMES_MANIFEST = Path("data/processed/survey_frames_manifest.parquet")
_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def _image_id(image_path: str) -> str:
    return hashlib.md5(image_path.encode()).hexdigest()[:12]


def import_survey_frames(
    run_dir: Path,
    output_path: Path = _SURVEY_FRAMES_MANIFEST,
    *,
    split: str = "train",
    append: bool = True,
) -> dict:
    """Collect all frames from a survey run directory into a parquet manifest.

    Looks for images in run_dir/frames/ and run_dir/problem_images/ (then run_dir root).
    Appends to existing manifest if append=True and the file exists.
    Returns {"added": N, "total": M, "output": path_str, "source": source_name}.
    """
    run_dir = run_dir.resolve()
    source_name = run_dir.name

    frame_paths: list[Path] = []
    seen_paths: set[Path] = set()
    for subdir in ("frames", "problem_images", ""):
        search_dir = run_dir / subdir if subdir else run_dir
        if search_dir.is_dir():
            for p in sorted(search_dir.iterdir()):
                if p.suffix.lower() in _IMAGE_EXTS and p.is_file() and p not in seen_paths:
                    frame_paths.append(p)
                    seen_paths.add(p)

    if not frame_paths:
        raise FileNotFoundError(
            f"No images found in {run_dir} (checked frames/, problem_images/, root). "
            "Run `uv run tarmac survey <video>` first to extract frames."
        )

    new_rows = [
        {
            "image_path": str(p),
            "id": _image_id(str(p)),
            "source_dataset": source_name,
            "split": split,
            "original_label": -1,
        }
        for p in frame_paths
    ]
    new_df = pd.DataFrame(new_rows)

    if append and output_path.exists():
        existing = pd.read_parquet(output_path)
        existing_paths = set(existing["image_path"].tolist())
        new_df = new_df[~new_df["image_path"].isin(existing_paths)]
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(output_path, index=False)

    return {
        "added": len(new_df),
        "total": len(combined),
        "output": str(output_path),
        "source": source_name,
    }


def export_labeled_frames(
    corrections_path: Path = Path("data/processed/corrections.parquet"),
    survey_manifest_path: Path = _SURVEY_FRAMES_MANIFEST,
    output_path: Path = Path("data/processed/survey_labeled_manifest.parquet"),
) -> dict:
    """Export survey frames that have been labeled (via corrections) as a SupCon manifest.

    Only exports frames present in survey_frames_manifest.parquet AND corrections.parquet.
    Output columns: image_path, split, surface_type, quality, source_dataset, composite_label.
    """
    if not survey_manifest_path.exists():
        raise FileNotFoundError(
            f"Survey frames manifest not found: {survey_manifest_path}. "
            "Run `uv run tarmac import-frames <run_dir>` first."
        )
    if not corrections_path.exists():
        raise FileNotFoundError(
            f"Corrections file not found: {corrections_path}. "
            "Label images in the labeling UI first."
        )

    survey_df = pd.read_parquet(survey_manifest_path)
    corr_df = pd.read_parquet(corrections_path)

    corr_records: dict[str, dict] = {}
    for _, row in corr_df.iterrows():
        labels: dict = json.loads(row["labels_json"]) if "labels_json" in corr_df.columns else {}
        corr_records[str(row["id"])] = {"image_path": str(row["image_path"]), "labels": labels}

    rows = []
    for _, srow in survey_df.iterrows():
        img_id = str(srow["id"])
        if img_id not in corr_records:
            continue
        labels = corr_records[img_id]["labels"]
        rows.append({
            "image_path": str(srow["image_path"]),
            "split": str(srow.get("split", "train")),
            "source_dataset": str(srow.get("source_dataset", "")),
            "surface_type": str(labels.get("surface_type", "unknown")),
            "quality": str(labels.get("quality", "unknown")),
            "has_crack": int(labels.get("has_crack", -1)),
            "composite_label": (
                f"{labels.get('surface_type', 'unknown')}_{labels.get('quality', 'unknown')}"
            ),
        })

    if not rows:
        return {"exported": 0, "output": str(output_path)}

    out_df = pd.DataFrame(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path, index=False)
    return {"exported": len(rows), "output": str(output_path)}
