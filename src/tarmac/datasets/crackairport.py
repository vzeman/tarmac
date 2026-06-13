from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import requests
from PIL import Image
from tqdm import tqdm

DATASET_ID = "3v5r2fxf89"
VERSION = 1
PUBLIC_FILES_API = (
    "https://data.mendeley.com/public-api/datasets/"
    f"{DATASET_ID}/files?folder_id={{folder_id}}&version={VERSION}"
)
PUBLIC_ZIP_URL = f"https://data.mendeley.com/public-api/zip/{DATASET_ID}/download/{VERSION}"
HEADERS = {
    "Accept": "application/vnd.mendeley-public-dataset.1+json",
    "User-Agent": "tarmac/0.1 (+https://github.com/)",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Mendeley v1 archive layout observed on 2026-06-13 after extraction:
# CrackAirport A Dataset for Segmentation of Cracks/CrackAirport/train_images
# CrackAirport A Dataset for Segmentation of Cracks/CrackAirport/train_masks
# The public dataset page describes 2226 examples; the downloadable v1 archive
# currently resolves to 2251 image/mask pairs.


@dataclass(frozen=True)
class CrackAirportResult:
    output_dir: Path
    image_count: int
    mask_count: int
    pair_count: int
    pairs_path: Path
    layout: str
    archive_paths: list[Path]


@dataclass(frozen=True)
class MendeleyFile:
    filename: str
    size: int
    download_url: str


def list_crackairport_files() -> list[MendeleyFile]:
    """Resolve CrackAirport public archive download URLs from Mendeley Data."""
    payload = _get_json(PUBLIC_FILES_API.format(folder_id="root"))
    files: list[MendeleyFile] = []
    for entry in payload:
        details = entry.get("content_details")
        if details:
            files.append(
                MendeleyFile(
                    filename=str(entry["filename"]),
                    size=int(details["size"]),
                    download_url=str(details["download_url"]),
                )
            )
    return files


def download_crackairport(output_dir: Path = Path("data/raw/crackairport")) -> CrackAirportResult:
    """Download CrackAirport and detect image/mask pairs.

    CrackAirport is a Mendeley CC BY 4.0 airport-pavement crack segmentation
    dataset. Version 1 is distributed as archive files containing 512x512 RGB
    images and binary segmentation masks. The archive layout is detected after
    extraction; paths whose directory/name contains mask-like tokens are treated
    as masks, and the remaining image files are candidate source images. Pairing
    is by normalized stem after removing common suffixes such as ``_mask`` and
    ``_label``. A stable ``pairs.jsonl`` index is written for downstream YOLO
    conversion.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = output_dir / "archives"
    extracted_dir = output_dir / "_extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    files = list_crackairport_files()
    if not files:
        files = [MendeleyFile(filename=f"{DATASET_ID}-{VERSION}.zip", size=0, download_url=PUBLIC_ZIP_URL)]

    for file_info in files:
        destination = archives_dir / file_info.filename
        _stream_download(file_info.download_url, destination, file_info.size or None)
        downloaded.append(destination)
        if destination.suffix.lower() == ".zip":
            marker = extracted_dir / f".{destination.stem}.extracted"
            if not marker.exists():
                with ZipFile(destination) as archive:
                    archive.extractall(extracted_dir)
                marker.write_text("ok\n")

    pairs = find_crackairport_pairs(extracted_dir)
    if len(pairs) < 2000:
        raise RuntimeError(
            f"Expected roughly 2226 CrackAirport image/mask pairs, found {len(pairs)} in {extracted_dir}."
        )
    pairs_path = output_dir / "pairs.jsonl"
    with pairs_path.open("w") as handle:
        for image_path, mask_path in pairs:
            handle.write(
                json.dumps(
                    {
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "image_relpath": str(image_path.relative_to(extracted_dir)),
                        "mask_relpath": str(mask_path.relative_to(extracted_dir)),
                    }
                )
                + "\n"
            )

    layout = describe_layout(extracted_dir, pairs)
    (output_dir / "LAYOUT.md").write_text(layout + "\n")
    return CrackAirportResult(
        output_dir=output_dir,
        image_count=len({p[0] for p in pairs}),
        mask_count=len({p[1] for p in pairs}),
        pair_count=len(pairs),
        pairs_path=pairs_path,
        layout=layout,
        archive_paths=downloaded,
    )


def find_crackairport_pairs(root: Path) -> list[tuple[Path, Path]]:
    files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    masks: dict[str, Path] = {}
    images: dict[str, Path] = {}
    for path in files:
        key = _pair_key(path)
        if _looks_like_mask(path):
            masks.setdefault(key, path)
        else:
            images.setdefault(key, path)
    pairs = [(image, masks[key]) for key, image in images.items() if key in masks]
    return sorted(pairs, key=lambda pair: str(pair[0]))


def describe_layout(root: Path, pairs: list[tuple[Path, Path]]) -> str:
    image_dirs = sorted({str(image.parent.relative_to(root)) for image, _ in pairs})
    mask_dirs = sorted({str(mask.parent.relative_to(root)) for _, mask in pairs})
    return "\n".join(
        [
            "# CrackAirport detected layout",
            "",
            f"Root: `{root}`",
            f"Pairs: {len(pairs)}",
            f"Image directories: {', '.join(image_dirs[:12])}",
            f"Mask directories: {', '.join(mask_dirs[:12])}",
            "",
            "Pairing rule: normalized image stem matched to normalized mask stem after removing mask/label suffix tokens.",
        ]
    )


def _looks_like_mask(path: Path) -> bool:
    text = " ".join(part.lower() for part in path.parts)
    if any(token in text for token in ("mask", "label", "groundtruth", "ground_truth", "annotation")):
        return True
    try:
        with Image.open(path) as image:
            arr = image.convert("L").resize((32, 32))
            values = set(arr.getdata())
            return len(values) <= 4
    except Exception:
        return False


def _pair_key(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("_mask", "-mask", " mask", "_label", "-label", "_gt", "-gt", "_groundtruth"):
        stem = stem.replace(token, "")
    return "".join(ch for ch in stem if ch.isalnum())


def _stream_download(url: str, destination: Path, expected_size: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_size and destination.stat().st_size == expected_size:
        return
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60, headers={"User-Agent": HEADERS["User-Agent"]}) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or expected_size or 0)
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
    except requests.HTTPError:
        subprocess.run(["curl", "-L", "--fail", "-o", str(tmp_path), url], check=True)
    tmp_path.replace(destination)


def _get_json(url: str) -> list[dict[str, object]]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError:
        completed = subprocess.run(
            ["curl", "-L", "-H", f"Accept: {HEADERS['Accept']}", url],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)
