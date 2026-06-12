from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

ZENODO_RECORD_API = "https://zenodo.org/api/records/11449977"
IMAGE_ARCHIVE = "s_1024.zip"
CSV_FILE = "streetSurfaceVis_v1_0.csv"
EXPECTED_IMAGE_COUNT = 9122


@dataclass(frozen=True)
class StreetSurfaceVisDownload:
    csv_path: Path
    archive_path: Path
    image_count: int


def _stream_download(url: str, destination: Path, expected_size: int | None = None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and expected_size and destination.stat().st_size == expected_size:
        return

    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as response:
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
    tmp_path.replace(destination)


def _zenodo_files() -> dict[str, dict[str, object]]:
    response = requests.get(ZENODO_RECORD_API, timeout=60)
    response.raise_for_status()
    record = response.json()
    return {file_info["key"]: file_info for file_info in record["files"]}


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    marker = output_dir / ".s_1024_extracted"
    if marker.exists():
        return

    with ZipFile(archive_path) as archive:
        archive.extractall(output_dir)
    marker.write_text("ok\n")


def count_images(directory: Path) -> int:
    return sum(
        1
        for path in directory.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    )


def download_streetsurfacevis(output_dir: Path = Path("data/raw/streetsurfacevis")) -> StreetSurfaceVisDownload:
    """Download StreetSurfaceVis v1.0 1024px images and metadata CSV."""
    files = _zenodo_files()
    missing = {IMAGE_ARCHIVE, CSV_FILE} - set(files)
    if missing:
        raise RuntimeError(f"Zenodo record is missing expected files: {sorted(missing)}")

    output_dir.mkdir(parents=True, exist_ok=True)
    archive_path = output_dir / IMAGE_ARCHIVE
    csv_path = output_dir / CSV_FILE

    for key, destination in ((IMAGE_ARCHIVE, archive_path), (CSV_FILE, csv_path)):
        file_info = files[key]
        url = str(file_info["links"]["self"])
        size = int(file_info["size"])
        _stream_download(url, destination, size)

    _extract_archive(archive_path, output_dir)
    image_count = count_images(output_dir)
    if not 9000 <= image_count <= 9300:
        raise RuntimeError(
            f"Expected about {EXPECTED_IMAGE_COUNT} images after extraction, found {image_count}."
        )

    return StreetSurfaceVisDownload(
        csv_path=csv_path,
        archive_path=archive_path,
        image_count=image_count,
    )
