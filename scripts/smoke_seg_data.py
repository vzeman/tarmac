from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def main() -> None:
    _check_crackairport()
    _check_crackforest()
    _check_dense_seg_manifest_if_present()
    _check_rdd_raw_if_present()
    print("smoke_seg_data: ok")


def _check_crackairport() -> None:
    root = ROOT / "data" / "raw" / "crackairport"
    pairs_path = root / "pairs.jsonl"
    assert pairs_path.exists(), f"Missing CrackAirport pairs index: {pairs_path}"
    pairs = [json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip()]
    assert pairs, f"No CrackAirport pairs in {pairs_path}"
    for row in pairs[:20]:
        assert Path(row["image_path"]).exists(), f"Missing CrackAirport image: {row['image_path']}"
        assert Path(row["mask_path"]).exists(), f"Missing CrackAirport mask: {row['mask_path']}"


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


def _check_dense_seg_manifest_if_present() -> None:
    manifest_path = ROOT / "data" / "processed" / "crack_seg_expanded" / "manifest.jsonl"
    if not manifest_path.exists():
        return
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
    assert rows, f"No rows in dense segmentation manifest: {manifest_path}"
    source_counts: dict[str, int] = {}
    for row in rows:
        source = str(row.get("source_dataset", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
        image_path = Path(row.get("source_image") or row["image_path"])
        mask_path = Path(row.get("source_mask") or row.get("mask_path") or row["label_path"])
        if not image_path.is_absolute():
            image_path = ROOT / image_path
        if not mask_path.is_absolute():
            mask_path = ROOT / mask_path
        assert image_path.exists(), f"Missing dense segmentation image: {image_path}"
        assert mask_path.exists(), f"Missing dense segmentation mask: {mask_path}"
    assert source_counts.get("crackairport", 0) > 0, "CrackAirport missing from dense segmentation manifest"
    assert source_counts.get("crackforest", 0) > 0, "CrackForest missing from dense segmentation manifest"


def _check_rdd_raw_if_present() -> None:
    root = ROOT / "data" / "raw" / "rdd2022" / "Czech"
    if not root.exists():
        return
    image_paths = sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_EXTENSIONS)
    annotation_paths = sorted(root.rglob("*.xml"))
    assert image_paths, f"No RDD2022 raw images found under {root}"
    assert annotation_paths, f"No RDD2022 Pascal VOC annotations found under {root}"


if __name__ == "__main__":
    main()
