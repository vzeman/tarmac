from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> None:
    _check_crackforest()
    _check_expanded_seg()
    _check_rdd_if_present()
    print("smoke_seg_data: ok")


def _check_crackforest() -> None:
    root = ROOT / "data" / "raw" / "crackforest"
    images = sorted(path for path in (root / "images").glob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    masks = sorted((root / "masks").glob("*.png"))
    pairs_path = root / "pairs.jsonl"
    assert images, f"No CrackForest images found under {root / 'images'}"
    assert masks, f"No CrackForest masks found under {root / 'masks'}"
    assert pairs_path.exists(), f"Missing CrackForest pairs index: {pairs_path}"
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    assert len(images) == len(masks) == len(pairs), (
        f"CrackForest count mismatch: images={len(images)} masks={len(masks)} pairs={len(pairs)}"
    )
    for row in pairs:
        assert Path(row["image_path"]).exists(), f"Missing CrackForest image: {row['image_path']}"
        assert Path(row["mask_path"]).exists(), f"Missing CrackForest mask: {row['mask_path']}"


def _check_expanded_seg() -> None:
    root = ROOT / "data" / "processed" / "yolo_seg_expanded"
    data_yaml = root / "data.yaml"
    manifest_path = root / "manifest.jsonl"
    metadata_path = root / "prep_metadata.json"
    assert data_yaml.exists(), f"Missing expanded segmentation data.yaml: {data_yaml}"
    assert manifest_path.exists(), f"Missing expanded segmentation manifest: {manifest_path}"
    metadata = json.loads(metadata_path.read_text())
    assert metadata["source_counts"].get("crackairport", 0) > 0, "CrackAirport missing from expanded seg metadata"
    assert metadata["source_counts"].get("crackforest", 0) > 0, "CrackForest missing from expanded seg metadata"

    non_empty_by_source = {"crackairport": 0, "crackforest": 0}
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    assert rows, f"No rows in expanded segmentation manifest: {manifest_path}"
    for row in rows:
        label_path = Path(row["label_path"])
        assert label_path.exists(), f"Missing expanded segmentation label: {label_path}"
        if label_path.read_text().strip():
            source = row["source_dataset"]
            if source in non_empty_by_source:
                non_empty_by_source[source] += 1
    assert non_empty_by_source["crackairport"] > 0, "No non-empty CrackAirport labels in expanded seg dataset"
    assert non_empty_by_source["crackforest"] > 0, "No non-empty CrackForest labels in expanded seg dataset"


def _check_rdd_if_present() -> None:
    root = ROOT / "data" / "processed" / "yolo_rdd"
    data_yaml = root / "data.yaml"
    if not data_yaml.exists():
        return
    text = data_yaml.read_text()
    for label in ("D00", "D10", "D20", "D40"):
        assert label in text, f"Missing RDD class {label} in {data_yaml}"
    metadata_path = root / "prep_metadata.json"
    assert metadata_path.exists(), f"Missing RDD prep metadata: {metadata_path}"
    metadata = json.loads(metadata_path.read_text())
    assert metadata.get("classes") == ["D00", "D10", "D20", "D40"], "RDD dataset must expose exactly 4 classes"


if __name__ == "__main__":
    main()
