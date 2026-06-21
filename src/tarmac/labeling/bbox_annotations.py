"""Bounding-box annotation storage: per-image labeled rectangles.

Stored normalized to [0,1] relative to image native dimensions.
Compatible with COCO JSON and YOLO TXT export.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_BBOX_DIR = Path("data/processed/bbox_annotations")
_INDEX_PATH = Path("data/processed/bbox_annotations/index.json")


def _ensure_dir() -> Path:
    _BBOX_DIR.mkdir(parents=True, exist_ok=True)
    return _BBOX_DIR


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


def get_bboxes(img_id: str) -> dict[str, Any] | None:
    return load_index().get(img_id)


def save_bboxes(
    img_id: str,
    image_path: str,
    bboxes: list[dict],
    *,
    nat_w: int | None = None,
    nat_h: int | None = None,
) -> dict[str, Any]:
    """Persist bounding-box annotations for one image.

    bboxes: list of dicts with keys x, y, w, h (normalized 0..1), label (class name string).
    """
    index = load_index()
    entry: dict[str, Any] = {
        "img_id": img_id,
        "image_path": image_path,
        "bboxes": bboxes,
    }
    if nat_w:
        entry["width"] = nat_w
    if nat_h:
        entry["height"] = nat_h
    index[img_id] = entry
    _save_index(index)
    return entry


def delete_bboxes(img_id: str) -> bool:
    index = load_index()
    if img_id not in index:
        return False
    del index[img_id]
    _save_index(index)
    return True


def all_class_names() -> list[str]:
    index = load_index()
    classes: set[str] = set()
    for entry in index.values():
        for bbox in entry.get("bboxes", []):
            label = bbox.get("label", "")
            if label:
                classes.add(label)
    return sorted(classes)


def export_coco_json(output_path: Path) -> dict[str, int]:
    """Export all bbox annotations as a COCO-format JSON file."""
    index = load_index()
    class_names = all_class_names()
    cat_to_id = {name: i + 1 for i, name in enumerate(class_names)}

    categories = [{"id": cat_to_id[n], "name": n, "supercategory": "object"} for n in class_names]
    images = []
    annotations = []
    ann_id = 1

    for img_idx, (img_id, entry) in enumerate(index.items(), start=1):
        image_path = entry.get("image_path", "")
        nat_w = entry.get("width", 0)
        nat_h = entry.get("height", 0)

        if not nat_w or not nat_h:
            try:
                from PIL import Image as _Image
                with _Image.open(image_path) as im:
                    nat_w, nat_h = im.size
            except Exception:
                nat_w, nat_h = 0, 0

        images.append({
            "id": img_idx,
            "file_name": image_path,
            "width": nat_w,
            "height": nat_h,
        })

        for bbox in entry.get("bboxes", []):
            label = bbox.get("label", "")
            if not label:
                continue
            cat_id = cat_to_id.get(label, 1)
            x_abs = float(bbox["x"]) * nat_w
            y_abs = float(bbox["y"]) * nat_h
            w_abs = float(bbox["w"]) * nat_w
            h_abs = float(bbox["h"]) * nat_h
            annotations.append({
                "id": ann_id,
                "image_id": img_idx,
                "category_id": cat_id,
                "bbox": [round(x_abs, 2), round(y_abs, 2), round(w_abs, 2), round(h_abs, 2)],
                "area": round(w_abs * h_abs, 2),
                "iscrowd": 0,
            })
            ann_id += 1

    coco = {
        "info": {"description": "Tarmac bbox annotations", "version": "1.0"},
        "categories": categories,
        "images": images,
        "annotations": annotations,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(coco, indent=2))
    return {"images": len(images), "annotations": len(annotations), "categories": len(categories)}


def export_yolo_txt(output_dir: Path) -> dict[str, int]:
    """Export all bbox annotations as YOLO TXT files (one .txt per image).

    Also writes classes.txt with the class index → name mapping.
    """
    index = load_index()
    class_names = all_class_names()
    cat_to_id = {name: i for i, name in enumerate(class_names)}

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "classes.txt").write_text("\n".join(class_names) + "\n")

    images_written = 0
    total_anns = 0

    for img_id, entry in index.items():
        bboxes = entry.get("bboxes", [])
        if not bboxes:
            continue
        lines = []
        for bbox in bboxes:
            label = bbox.get("label", "")
            if not label:
                continue
            cat_id = cat_to_id.get(label, 0)
            cx = float(bbox["x"]) + float(bbox["w"]) / 2
            cy = float(bbox["y"]) + float(bbox["h"]) / 2
            w = float(bbox["w"])
            h = float(bbox["h"])
            lines.append(f"{cat_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_anns += 1
        if lines:
            (output_dir / f"{img_id}.txt").write_text("\n".join(lines) + "\n")
            images_written += 1

    return {"images": images_written, "annotations": total_anns, "classes": len(class_names)}


def bbox_stats() -> dict[str, Any]:
    index = load_index()
    by_class: dict[str, int] = {}
    total_anns = 0
    for entry in index.values():
        for bbox in entry.get("bboxes", []):
            label = bbox.get("label", "unknown")
            by_class[label] = by_class.get(label, 0) + 1
            total_anns += 1
    return {"images": len(index), "total_annotations": total_anns, "by_class": by_class}
