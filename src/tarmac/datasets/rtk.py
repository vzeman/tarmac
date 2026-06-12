from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

DATASET_ID = "fxy5khmhpb"
VERSION = 1
PUBLIC_FILES_API = (
    "https://data.mendeley.com/public-api/datasets/"
    f"{DATASET_ID}/files?folder_id={{folder_id}}&version={VERSION}"
)


@dataclass(frozen=True)
class MendeleyFile:
    file_id: str
    filename: str
    size: int
    download_url: str
    folder_id: str | None = None


def _list_folder(folder_id: str = "root") -> list[dict[str, object]]:
    response = requests.get(
        PUBLIC_FILES_API.format(folder_id=folder_id),
        headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"},
        timeout=60,
    )
    response.raise_for_status()
    return response.json()


def list_rtk_files() -> list[MendeleyFile]:
    """Resolve public RTK file download URLs from Mendeley Data."""
    pending = ["root"]
    files: list[MendeleyFile] = []

    while pending:
        folder_id = pending.pop()
        for entry in _list_folder(folder_id):
            if "content_details" in entry:
                details = entry["content_details"]
                files.append(
                    MendeleyFile(
                        file_id=str(entry["id"]),
                        filename=str(entry["filename"]),
                        size=int(details["size"]),
                        download_url=str(details["download_url"]),
                        folder_id=entry.get("folder_id"),
                    )
                )
            elif entry.get("id"):
                pending.append(str(entry["id"]))

    if not files:
        raise RuntimeError("No RTK files were returned by the Mendeley public API.")
    return files


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


def download_rtk(output_dir: Path = Path("data/raw/rtk"), extract: bool = True) -> list[Path]:
    """Download RTK from Mendeley Data.

    This function is implemented but intentionally not wired to the Phase 1 CLI run.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for file_info in list_rtk_files():
        destination = output_dir / file_info.filename
        _stream_download(file_info.download_url, destination, file_info.size)
        downloaded.append(destination)
        if extract and destination.suffix.lower() == ".zip":
            extract_dir = output_dir / destination.stem
            marker = extract_dir / ".extracted"
            if not marker.exists():
                extract_dir.mkdir(parents=True, exist_ok=True)
                with ZipFile(destination) as archive:
                    archive.extractall(extract_dir)
                marker.write_text("ok\n")
    return downloaded
