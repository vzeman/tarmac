from __future__ import annotations

import shutil
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

DATASET_ID = "429vzbgmbx"
VERSION = 1
PUBLIC_FILES_API = (
    "https://data.mendeley.com/public-api/datasets/"
    f"{DATASET_ID}/files?folder_id={{folder_id}}&version={VERSION}"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
HEADERS = {
    "Accept": "application/vnd.mendeley-public-dataset.1+json",
    "User-Agent": "tarmac/0.1 (+https://github.com/)",
}


@dataclass(frozen=True)
class CrackConcretePavementResult:
    output_dir: Path
    positive_count: int
    negative_count: int
    archive_paths: list[Path]


@dataclass(frozen=True)
class MendeleyFile:
    filename: str
    size: int
    download_url: str


def list_crack_concrete_pavement_files() -> list[MendeleyFile]:
    payload = _get_json(PUBLIC_FILES_API.format(folder_id="root"))
    files: list[MendeleyFile] = []
    for entry in payload:
        details = entry.get("content_details")
        if not details:
            continue
        files.append(
            MendeleyFile(
                filename=str(entry["filename"]),
                size=int(details["size"]),
                download_url=str(details["download_url"]),
            )
        )
    if not files:
        raise RuntimeError("No files were returned for Mendeley dataset 429vzbgmbx version 1.")
    return files


def download_cracks_concrete_pavement(
    output_dir: Path = Path("data/raw/cracks_concrete_pavement"),
) -> CrackConcretePavementResult:
    """Download and normalize the Mendeley concrete/pavement crack dataset.

    The raw zip contains binary crack/non-crack folders. This function keeps the
    archive and extracts/copies images into stable positive/ and negative/ folders.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = output_dir / "archives"
    extracted_dir = output_dir / "_extracted"
    positive_dir = output_dir / "positive"
    negative_dir = output_dir / "negative"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    positive_dir.mkdir(parents=True, exist_ok=True)
    negative_dir.mkdir(parents=True, exist_ok=True)

    downloaded: list[Path] = []
    for file_info in list_crack_concrete_pavement_files():
        destination = archives_dir / file_info.filename
        _stream_download(file_info.download_url, destination, file_info.size)
        downloaded.append(destination)
        if destination.suffix.lower() == ".zip":
            marker = extracted_dir / f".{destination.stem}.extracted"
            if not marker.exists():
                with ZipFile(destination) as archive:
                    archive.extractall(extracted_dir)
                marker.write_text("ok\n")

    if not any(positive_dir.iterdir()) or not any(negative_dir.iterdir()):
        _normalize_binary_folders(extracted_dir, positive_dir, negative_dir)

    positive_count = _count_images(positive_dir)
    negative_count = _count_images(negative_dir)
    if positive_count == 0 or negative_count == 0:
        raise RuntimeError(
            f"Expected positive and negative images after extraction, got "
            f"positive={positive_count}, negative={negative_count} in {output_dir}."
        )
    return CrackConcretePavementResult(
        output_dir=output_dir,
        positive_count=positive_count,
        negative_count=negative_count,
        archive_paths=downloaded,
    )


def _normalize_binary_folders(extracted_dir: Path, positive_dir: Path, negative_dir: Path) -> None:
    for image_path in sorted(p for p in extracted_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        label = _label_from_path(image_path)
        if label is None:
            continue
        target_dir = positive_dir if label == 1 else negative_dir
        target = target_dir / image_path.name
        if target.exists():
            target = target_dir / f"{image_path.parent.name}_{image_path.name}"
        shutil.copy2(image_path, target)


def _label_from_path(path: Path) -> int | None:
    parts = [part.lower() for part in path.parts]
    if any(part in {"positive", "crack", "cracked"} or "positive" in part for part in parts):
        return 1
    if any(
        part in {"negative", "non-crack", "noncrack", "uncracked"} or "negative" in part
        for part in parts
    ):
        return 0
    return None


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


def _count_images(path: Path) -> int:
    return sum(1 for p in path.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS)


def _get_json(url: str) -> list[dict[str, object]]:
    try:
        response = requests.get(url, headers=HEADERS, timeout=60)
        response.raise_for_status()
        return response.json()
    except requests.HTTPError:
        completed = subprocess.run(
            [
                "curl",
                "-L",
                "-H",
                f"Accept: {HEADERS['Accept']}",
                url,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(completed.stdout)
