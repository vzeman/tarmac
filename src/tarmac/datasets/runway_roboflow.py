from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import requests
import pandas as pd
from PIL import Image
from requests import HTTPError
from tqdm import tqdm

from tarmac.embedding.tiling import tile_boxes

PROJECT = "revathi-deusp/runway-crack-detection-1iq1l"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CRACK_CLASSES = {"crack", "mildcrack", "severecrack"}


@dataclass(frozen=True)
class RunwayRoboflowResult:
    output_dir: Path
    image_count: int
    tile_label_count: int
    positive_tile_count: int
    negative_tile_count: int


def download_runway_roboflow(
    output_dir: Path = Path("data/raw/runway_roboflow"),
    api_key: str | None = None,
    version: int | None = None,
) -> RunwayRoboflowResult:
    api_key = api_key or os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ROBOFLOW_API_KEY is not set. Create a free Roboflow account, copy your API key "
            "from Account Settings, then run: export ROBOFLOW_API_KEY=... && "
            "uv run tarmac download runway-roboflow"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    version = version or _latest_version(api_key)
    if version is None:
        labels = download_roboflow_images_to_tile_labels(output_dir, api_key)
        image_count = _count_labeled_images(labels)
        positive_count = int(labels["has_crack"].sum()) if not labels.empty else 0
        negative_count = int(len(labels) - positive_count)
        return RunwayRoboflowResult(output_dir, image_count, len(labels), positive_count, negative_count)
    zip_path = output_dir / f"runway_roboflow_v{version}.zip"
    extract_dir = output_dir / f"v{version}"
    if not extract_dir.exists():
        url = f"https://api.roboflow.com/{PROJECT}/{version}/coco?api_key={api_key}"
        response = requests.get(url, timeout=60)
        _raise_for_status_redacted(response, api_key)
        payload = response.json()
        download_url = payload.get("export", {}).get("link") or payload.get("download")
        if not download_url:
            raise RuntimeError(f"Roboflow did not return a COCO export URL for {PROJECT} v{version}.")
        _stream_download(str(download_url), zip_path)
        with ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
    labels = convert_roboflow_bboxes_to_tile_labels(extract_dir, output_dir / "tile_labels")
    image_count = _count_labeled_images(labels)
    positive_count = int(labels["has_crack"].sum()) if not labels.empty else 0
    negative_count = int(len(labels) - positive_count)
    return RunwayRoboflowResult(output_dir, image_count, len(labels), positive_count, negative_count)


def convert_roboflow_bboxes_to_tile_labels(dataset_dir: Path, output_stem: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for annotation_path in dataset_dir.rglob("_annotations.coco.json"):
        coco = json.loads(annotation_path.read_text())
        images = {int(item["id"]): item for item in coco.get("images", [])}
        categories = {
            int(item["id"]): str(item.get("name", "")).strip().lower()
            for item in coco.get("categories", [])
        }
        by_image: dict[int, list[list[float]]] = {}
        for ann in coco.get("annotations", []):
            category = _normalize_class(categories.get(int(ann.get("category_id", -1)), ""))
            if category not in CRACK_CLASSES:
                continue
            by_image.setdefault(int(ann["image_id"]), []).append([float(x) for x in ann["bbox"]])
        for image_id, image_info in images.items():
            image_path = annotation_path.parent / str(image_info["file_name"])
            if not image_path.exists():
                continue
            width = int(image_info.get("width") or 0)
            height = int(image_info.get("height") or 0)
            if width <= 0 or height <= 0:
                with Image.open(image_path) as image:
                    width, height = image.size
            for tile_index, tile_box in enumerate(tile_boxes(width, height)):
                has_crack = any(_overlap_fraction(tile_box, bbox) > 0.10 for bbox in by_image.get(image_id, []))
                rows.append(
                    {
                        "image_path": str(image_path),
                        "source_dataset": "runway_roboflow",
                        "tile": f"tile_{tile_index}",
                        "has_crack": int(has_crack),
                    }
                )
    labels = pd.DataFrame(rows, columns=["image_path", "source_dataset", "tile", "has_crack"])
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_stem.with_suffix(".jsonl")
    csv_path = output_stem.with_suffix(".csv")
    parquet_path = output_stem.with_suffix(".parquet")
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))
    labels.to_csv(csv_path, index=False)
    labels.to_parquet(parquet_path, index=False)
    return labels


def download_roboflow_images_to_tile_labels(output_dir: Path, api_key: str) -> pd.DataFrame:
    images_dir = output_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    image_records = _search_roboflow_images(api_key)
    rows: list[dict[str, object]] = []
    metadata: list[dict[str, object]] = []
    for item in tqdm(image_records, desc="Roboflow images", unit="image"):
        image_id = str(item["id"])
        detail = _roboflow_image_detail(api_key, image_id)
        image_info = detail.get("image", {})
        image_name = str(image_info.get("name") or item.get("name") or f"{image_id}.jpg")
        image_path = images_dir / f"{image_id}_{_safe_filename(image_name)}"
        image_url = image_info.get("urls", {}).get("original") or image_info.get("urls", {}).get("thumb")
        if not image_path.exists() and image_url:
            _stream_download(str(image_url), image_path)
        if not image_path.exists():
            continue
        annotation = image_info.get("annotation") or {}
        width = int(float(annotation.get("width") or 0))
        height = int(float(annotation.get("height") or 0))
        if width <= 0 or height <= 0:
            with Image.open(image_path) as image:
                width, height = image.size
        bboxes = []
        for box in annotation.get("boxes") or []:
            category = _normalize_class(str(box.get("label", "")))
            if category not in CRACK_CLASSES:
                continue
            bboxes.append(_xywh_center_to_bbox(box))
        for tile_index, tile_box in enumerate(tile_boxes(width, height)):
            has_crack = any(_overlap_fraction(tile_box, bbox) > 0.10 for bbox in bboxes)
            rows.append(
                {
                    "image_path": str(image_path),
                    "source_dataset": "runway_roboflow",
                    "tile": f"tile_{tile_index}",
                    "has_crack": int(has_crack),
                }
            )
        metadata.append(
            {
                "id": image_id,
                "name": image_name,
                "split": image_info.get("split"),
                "width": width,
                "height": height,
                "box_count": len(bboxes),
                "image_path": str(image_path),
            }
        )
    (output_dir / "image_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    labels = pd.DataFrame(rows, columns=["image_path", "source_dataset", "tile", "has_crack"])
    output_stem = output_dir / "tile_labels"
    labels.to_json(output_stem.with_suffix(".jsonl"), orient="records", lines=True)
    labels.to_csv(output_stem.with_suffix(".csv"), index=False)
    labels.to_parquet(output_stem.with_suffix(".parquet"), index=False)
    return labels


def _latest_version(api_key: str) -> int | None:
    response = requests.get(f"https://api.roboflow.com/{PROJECT}?api_key={api_key}", timeout=60)
    _raise_for_status_redacted(response, api_key)
    versions = response.json().get("project", {}).get("versions") or response.json().get("versions") or []
    version_numbers = [int(v.get("version")) for v in versions if v.get("version")]
    if not version_numbers:
        return None
    return max(version_numbers)


def _search_roboflow_images(api_key: str) -> list[dict[str, object]]:
    url = f"https://api.roboflow.com/{PROJECT}/search"
    offset = 0
    results: list[dict[str, object]] = []
    while True:
        response = requests.post(
            url,
            params={"api_key": api_key},
            json={
                "in_dataset": True,
                "limit": 250,
                "offset": offset,
                "fields": ["id", "name", "annotations", "labels", "split"],
            },
            timeout=60,
        )
        _raise_for_status_redacted(response, api_key)
        payload = response.json()
        batch = payload.get("results") or []
        results.extend(batch)
        total = int(payload.get("total") or len(results))
        if len(results) >= total or not batch:
            return results
        offset += len(batch)


def _roboflow_image_detail(api_key: str, image_id: str) -> dict[str, object]:
    response = requests.get(
        f"https://api.roboflow.com/{PROJECT}/images/{image_id}",
        params={"api_key": api_key},
        timeout=60,
    )
    _raise_for_status_redacted(response, api_key)
    return response.json()


def _xywh_center_to_bbox(box: dict[str, object]) -> list[float]:
    width = float(box.get("width") or 0.0)
    height = float(box.get("height") or 0.0)
    x_center = float(box.get("x") or 0.0)
    y_center = float(box.get("y") or 0.0)
    return [x_center - width / 2.0, y_center - height / 2.0, width, height]


def _normalize_class(name: str) -> str:
    cleaned = name.strip().lower().replace("_", "").replace("-", "")
    if cleaned.endswith("s"):
        cleaned = cleaned[:-1]
    return cleaned


def _safe_filename(name: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe or "image.jpg"


def _raise_for_status_redacted(response: requests.Response, api_key: str) -> None:
    try:
        response.raise_for_status()
    except HTTPError as error:
        message = str(error).replace(api_key, "***")
        raise RuntimeError(message) from None


def _stream_download(url: str, destination: Path) -> None:
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        tmp_path = destination.with_suffix(destination.suffix + ".part")
        with tmp_path.open("wb") as handle, tqdm(
            total=total or None,
            unit="B",
            unit_scale=True,
            desc=destination.name,
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
        tmp_path.replace(destination)


def _overlap_fraction(tile: tuple[int, int, int, int], bbox: list[float]) -> float:
    left, top, right, bottom = tile
    bx, by, bw, bh = bbox
    bleft, btop, bright, bbottom = bx, by, bx + bw, by + bh
    ix0 = max(float(left), bleft)
    iy0 = max(float(top), btop)
    ix1 = min(float(right), bright)
    iy1 = min(float(bottom), bbottom)
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    tile_area = max(1.0, float((right - left) * (bottom - top)))
    return ((ix1 - ix0) * (iy1 - iy0)) / tile_area


def _count_images(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _count_labeled_images(labels: pd.DataFrame) -> int:
    if labels.empty:
        return 0
    return int(labels["image_path"].nunique())
