from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import requests
from PIL import Image
from tqdm import tqdm

from tarmac.embedding.tiling import tile_boxes

PROJECT = "revathi-deusp/runway-crack-detection-1iq1l"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass(frozen=True)
class RunwayRoboflowResult:
    output_dir: Path
    image_count: int
    tile_label_count: int


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
    zip_path = output_dir / f"runway_roboflow_v{version}.zip"
    extract_dir = output_dir / f"v{version}"
    if not extract_dir.exists():
        url = f"https://api.roboflow.com/{PROJECT}/{version}?api_key={api_key}&format=coco"
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        payload = response.json()
        download_url = payload.get("export", {}).get("link") or payload.get("download")
        if not download_url:
            raise RuntimeError(f"Roboflow did not return a COCO export URL for {PROJECT} v{version}.")
        _stream_download(str(download_url), zip_path)
        with ZipFile(zip_path) as archive:
            archive.extractall(extract_dir)
    labels = convert_roboflow_bboxes_to_tile_labels(extract_dir, output_dir / "tile_labels.jsonl")
    return RunwayRoboflowResult(output_dir, _count_images(extract_dir), labels)


def convert_roboflow_bboxes_to_tile_labels(dataset_dir: Path, output_jsonl: Path) -> int:
    rows: list[dict[str, object]] = []
    for annotation_path in dataset_dir.rglob("_annotations.coco.json"):
        coco = json.loads(annotation_path.read_text())
        images = {int(item["id"]): item for item in coco.get("images", [])}
        by_image: dict[int, list[list[float]]] = {}
        for ann in coco.get("annotations", []):
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
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    output_jsonl.write_text("\n".join(json.dumps(row) for row in rows) + ("\n" if rows else ""))
    return len(rows)


def _latest_version(api_key: str) -> int:
    response = requests.get(f"https://api.roboflow.com/{PROJECT}?api_key={api_key}", timeout=60)
    response.raise_for_status()
    versions = response.json().get("project", {}).get("versions") or response.json().get("versions") or []
    version_numbers = [int(v.get("version")) for v in versions if v.get("version")]
    if not version_numbers:
        return 1
    return max(version_numbers)


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
