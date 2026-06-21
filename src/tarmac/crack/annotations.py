"""Crack annotation storage: per-image drawn masks + metadata.

Masks are stored as RGBA PNGs where white (R>30) = crack pixel, transparent = background.
This format loads directly into an HTML canvas without pixel manipulation.
"""
from __future__ import annotations

import base64
import io
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

_ANNOTATIONS_DIR = Path("data/processed/crack_annotations")
_INDEX_PATH = Path("data/processed/crack_annotations/index.json")


def _ensure_dir() -> Path:
    _ANNOTATIONS_DIR.mkdir(parents=True, exist_ok=True)
    return _ANNOTATIONS_DIR


def load_index() -> dict[str, dict]:
    if _INDEX_PATH.exists():
        try:
            return json.loads(_INDEX_PATH.read_text())
        except Exception:
            return {}
    return {}


def _save_index(index: dict[str, dict]) -> None:
    _ensure_dir()
    _INDEX_PATH.write_text(json.dumps(index, indent=2))


def get_annotation(img_id: str) -> dict[str, Any] | None:
    return load_index().get(img_id)


def save_annotation(
    img_id: str,
    image_path: str,
    mask_b64: str,
    *,
    source: str = "manual",
    confidence: float | None = None,
    nat_w: int | None = None,
    nat_h: int | None = None,
) -> dict[str, Any]:
    """Persist a drawn crack mask.

    mask_b64: base64-encoded PNG canvas export (any color, alpha channel used for presence).
              White-on-transparent (from HTML canvas) or white-on-black both accepted.
    """
    _ensure_dir()
    mask_path = _ANNOTATIONS_DIR / f"{img_id}_mask.png"

    mask_bytes = base64.b64decode(mask_b64)
    with Image.open(io.BytesIO(mask_bytes)) as raw:
        if raw.mode in ("RGBA", "LA"):
            arr = np.asarray(raw.convert("RGBA"), dtype=np.uint8)
            # presence determined by alpha channel
            present = arr[:, :, 3] > 30
        else:
            gray = np.asarray(raw.convert("L"), dtype=np.uint8)
            present = gray > 30

        # Save as RGBA: white+opaque where crack, fully transparent elsewhere
        rgba = np.zeros((*present.shape, 4), dtype=np.uint8)
        rgba[present] = [255, 255, 255, 255]
        Image.fromarray(rgba, mode="RGBA").save(mask_path, format="PNG")

    index = load_index()
    entry: dict[str, Any] = {
        "img_id": img_id,
        "image_path": image_path,
        "mask_path": str(mask_path),
        "source": source,
    }
    if confidence is not None:
        entry["confidence"] = float(confidence)
    if nat_w:
        entry["width"] = nat_w
    if nat_h:
        entry["height"] = nat_h
    index[img_id] = entry
    _save_index(index)
    return entry


def delete_annotation(img_id: str) -> bool:
    index = load_index()
    if img_id not in index:
        return False
    mask_path = Path(index[img_id].get("mask_path", ""))
    if mask_path.exists():
        mask_path.unlink(missing_ok=True)
    del index[img_id]
    _save_index(index)
    return True


def export_seg_manifest(output_path: Path) -> int:
    """Write a JSONL manifest of all user annotations for seg-head fine-tuning.

    Each entry includes image_path, mask_path (RGBA PNG), source_dataset='user_annotation',
    split='train'.  Compatible with load_seg_records() in seg_head.py.
    """
    index = load_index()
    lines: list[str] = []
    for entry in index.values():
        image_path = entry.get("image_path", "")
        mask_path = entry.get("mask_path", "")
        if not Path(image_path).exists() or not Path(mask_path).exists():
            continue
        lines.append(json.dumps({
            "image_path": image_path,
            "mask_path": mask_path,
            "source_dataset": "user_annotation",
            "split": "train",
        }))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def annotation_stats() -> dict[str, int]:
    index = load_index()
    by_source: dict[str, int] = {}
    for entry in index.values():
        src = str(entry.get("source", "manual"))
        by_source[src] = by_source.get(src, 0) + 1
    return {"total": len(index), **by_source}
