"""Smoke checks for the learned crack segmentation head."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image

from tarmac.crack.segment import segment_cracks


ROOT = Path(__file__).resolve().parents[1]


def _find_crackairport_image() -> Path:
    manifest = ROOT / "data/processed/yolo_seg_expanded/manifest.jsonl"
    if not manifest.exists():
        raise AssertionError(f"missing manifest: {manifest}")
    with manifest.open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("source_dataset") != "crackairport":
                continue
            image_path = Path(row.get("source_image") or row.get("image_path", ""))
            if not image_path.is_absolute():
                image_path = ROOT / image_path
            if image_path.exists():
                return image_path
    raise AssertionError("no CrackAirport image found in expanded segmentation manifest")


def main() -> None:
    checkpoint = ROOT / "models/crack_seg_head.pt"
    metrics_path = ROOT / "reports/crack_seg_head_metrics.json"
    examples = sorted((ROOT / "reports/examples").glob("08_*.png"))

    if not checkpoint.exists():
        raise AssertionError(f"missing learned segmenter checkpoint: {checkpoint}")
    if not metrics_path.exists():
        raise AssertionError(f"missing metrics JSON: {metrics_path}")
    if not examples:
        raise AssertionError("missing learned crack segmentation example overlay: reports/examples/08_*.png")

    metrics = json.loads(metrics_path.read_text())
    test_metrics = metrics.get("test", {}).get("overall", {})
    for key in ("iou", "dice"):
        value = float(test_metrics.get(key, 0.0))
        if value <= 0.0:
            raise AssertionError(f"missing or invalid test {key}: {value}")

    image_path = _find_crackairport_image()
    result = segment_cracks(Image.open(image_path), crack_head=None, embedder=None)
    if result.segmenter != "dinov3_dense_head":
        raise AssertionError(f"segment_cracks did not prefer learned head: {result.segmenter}")
    if result.mask.shape[:2] != result.heatmap.shape[:2]:
        raise AssertionError("mask and heatmap shapes differ")
    if int(result.mask.sum()) <= 0:
        raise AssertionError("learned segmenter produced an empty mask on smoke image")

    print(
        "seg-head smoke ok: "
        f"test_iou={test_metrics['iou']:.4f} "
        f"test_dice={test_metrics['dice']:.4f} "
        f"example={examples[0].relative_to(ROOT)}"
    )


if __name__ == "__main__":
    main()
