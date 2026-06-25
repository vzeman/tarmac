from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

DATASET_ID = "5y9wdsg2zt"
VERSION = 2
PUBLIC_FILES_API = (
    "https://data.mendeley.com/public-api/datasets/"
    f"{DATASET_ID}/files?folder_id={{folder_id}}&version={VERSION}"
)
PUBLIC_ZIP_URL = f"https://data.mendeley.com/public-api/zip/{DATASET_ID}/download/{VERSION}"
HEADERS = {
    "Accept": "application/vnd.mendeley-public-dataset.1+json",
    "User-Agent": "tarmac/0.1 (+https://github.com/)",
}
DATASET_URL = f"https://data.mendeley.com/datasets/{DATASET_ID}/{VERSION}"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

# Archive layout: "Negative/" (no crack) and "Positive/" (crack), each containing
# 227×227 JPG images. ~40,000 images total. CC BY 4.0 license.
# Source: Middle East Technical University (METU).


@dataclass(frozen=True)
class Mendeley5y9Result:
    output_dir: Path
    positive_count: int
    negative_count: int
    archive_paths: list[Path]


@dataclass(frozen=True)
class MendeleyFile:
    filename: str
    size: int
    download_url: str


def download_mendeley5y9(output_dir: Path = Path("data/raw/mendeley5y9")) -> Mendeley5y9Result:
    """Download Mendeley 5y9wdsg2zt — 40,000 binary crack/no-crack images (METU).

    This dataset from Middle East Technical University contains approximately
    40,000 images at 227×227 pixels, split into ``Negative/`` (no crack) and
    ``Positive/`` (crack present) folders. It is binary classification only;
    no segmentation masks are provided. License: CC BY 4.0.

    Primary source: https://data.mendeley.com/datasets/5y9wdsg2zt/2
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = output_dir / "archives"
    extracted_dir = output_dir / "_extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    files = _list_files()
    if not files:
        files = [MendeleyFile(filename=f"{DATASET_ID}-{VERSION}.zip", size=0, download_url=PUBLIC_ZIP_URL)]

    for file_info in files:
        destination = archives_dir / file_info.filename
        _stream_download(file_info.download_url, destination, file_info.size or None)
        downloaded.append(destination)
        ext = destination.suffix.lower()
        marker = extracted_dir / f".{destination.stem}.extracted"
        if not marker.exists():
            if ext == ".zip":
                with ZipFile(destination) as archive:
                    archive.extractall(extracted_dir)
            elif ext == ".rar":
                _extract_rar(destination, extracted_dir)
            marker.write_text("ok\n")

    positive_count = _count_images(_find_class_dir(extracted_dir, ("Positive", "positive", "crack", "cracked")))
    negative_count = _count_images(_find_class_dir(extracted_dir, ("Negative", "negative", "noncrack", "no_crack")))

    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(_describe_layout(extracted_dir, positive_count, negative_count) + "\n")
    return Mendeley5y9Result(
        output_dir=output_dir,
        positive_count=positive_count,
        negative_count=negative_count,
        archive_paths=downloaded,
    )


def find_mendeley5y9_images(raw_dir: Path = Path("data/raw/mendeley5y9")) -> dict[str, list[Path]]:
    """Return {label: [image_paths]} for 'positive' and 'negative' classes."""
    extracted_dir = raw_dir / "_extracted"
    return {
        "positive": _list_images(_find_class_dir(extracted_dir, ("Positive", "positive", "crack", "cracked"))),
        "negative": _list_images(_find_class_dir(extracted_dir, ("Negative", "negative", "noncrack", "no_crack"))),
    }


def _list_files() -> list[MendeleyFile]:
    try:
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
    except Exception:
        return []


def _find_class_dir(root: Path, names: tuple[str, ...]) -> Path | None:
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name in names:
            return candidate
    return None


def _count_images(directory: Path | None) -> int:
    if directory is None or not directory.exists():
        return 0
    return sum(1 for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _list_images(directory: Path | None) -> list[Path]:
    if directory is None or not directory.exists():
        return []
    return sorted(p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _extract_rar(archive_path: Path, output_dir: Path) -> None:
    extractor = shutil.which("bsdtar") or shutil.which("unrar") or shutil.which("7z")
    if extractor is None:
        raise RuntimeError(f"No RAR extractor found (need bsdtar, unrar, or 7z): {archive_path}")
    name = Path(extractor).name
    if name == "bsdtar":
        cmd = ["bsdtar", "-x", "-f", str(archive_path), "-C", str(output_dir)]
    elif name == "unrar":
        cmd = ["unrar", "x", "-y", str(archive_path), str(output_dir) + "/"]
    else:
        cmd = ["7z", "x", str(archive_path), f"-o{output_dir}", "-y"]
    subprocess.run(cmd, check=True)


def _stream_download(url: str, destination: Path, expected_size: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_size and destination.stat().st_size == expected_size:
        return
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120, headers={"User-Agent": HEADERS["User-Agent"]}) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or expected_size or 0)
            with tmp_path.open("wb") as handle, tqdm(
                total=total or None, unit="B", unit_scale=True, desc=destination.name
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


def _describe_layout(root: Path, positive_count: int, negative_count: int) -> str:
    return "\n".join([
        "# Mendeley 5y9wdsg2zt detected layout",
        "",
        f"Source: {DATASET_URL}",
        f"Root: `{root}`",
        f"Positive (crack): {positive_count} images",
        f"Negative (no crack): {negative_count} images",
        f"Total: {positive_count + negative_count} images",
        "",
        "License: CC BY 4.0",
        "Resolution: 227×227 pixels",
        "Annotation: binary classification only (no segmentation masks).",
        "Source: Middle East Technical University (METU).",
    ])
