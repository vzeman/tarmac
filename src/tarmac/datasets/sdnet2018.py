from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

DIGITALCOMMONS_URL = (
    "https://digitalcommons.usu.edu/cgi/viewcontent.cgi?"
    "filename=2&article=1047&context=all_datasets&type=additional"
)
README_URL = (
    "https://digitalcommons.usu.edu/cgi/viewcontent.cgi?"
    "filename=1&article=1047&context=all_datasets&type=additional"
)
IEEE_DATAPORT_URL = (
    "https://ieee-dataport.org/documents/"
    "sdnet2018-concrete-crack-image-dataset-machine-learning-applications"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DOMAIN_LABELS = {"D": "bridge_decks", "W": "walls", "P": "pavements"}
CRACKED_LABELS = {"CD", "CW", "CP", "cracked", "crack"}
UNCRACKED_LABELS = {"UD", "UW", "UP", "uncracked", "noncrack", "non-crack", "no_crack"}


@dataclass(frozen=True)
class Sdnet2018Result:
    output_dir: Path
    downloaded: bool
    source: str
    image_count: int
    counts: dict[str, dict[str, int]]
    message: str
    archive_path: Path | None = None


def download_sdnet2018(output_dir: Path = Path("data/raw/sdnet2018")) -> Sdnet2018Result:
    """Download and normalize SDNET2018 when a keyless source is reachable.

    The primary source named by the dataset is IEEE DataPort, which can require
    login. This downloader first tries the Utah State DigitalCommons mirror
    linked by the authors at DOI https://doi.org/10.15142/T3TD19. If that mirror
    is unavailable, the command does not fail the whole workflow; it writes
    ``MANUAL_DOWNLOAD.md`` and returns ``downloaded=False``.

    Manual fallback:
      1. Download ``SDNET2018.zip`` from IEEE DataPort or the Utah State page:
         https://digitalcommons.usu.edu/all_datasets/48/
      2. Place it at ``data/raw/sdnet2018/archives/SDNET2018.zip``.
      3. Re-run ``uv run tarmac download sdnet2018``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = output_dir / "archives"
    extracted_dir = output_dir / "_extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archives_dir / "SDNET2018.zip"

    source = "local archive"
    if not archive_path.exists():
        try:
            _stream_download(DIGITALCOMMONS_URL, archive_path, expected_content_type="application/zip")
            source = "Utah State DigitalCommons keyless mirror"
            _download_readme(output_dir)
        except Exception as exc:
            LOGGER.warning("SDNET2018 download skipped: %s", exc)
            instructions = _manual_instructions(str(exc))
            (output_dir / "MANUAL_DOWNLOAD.md").write_text(instructions + "\n")
            return Sdnet2018Result(
                output_dir=output_dir,
                downloaded=False,
                source="skipped",
                image_count=0,
                counts={},
                message=instructions,
                archive_path=None,
            )

    _extract_archive(archive_path, extracted_dir)
    normalized = normalize_sdnet2018(extracted_dir, output_dir)
    counts = count_sdnet2018(output_dir)
    image_count = sum(sum(label_counts.values()) for label_counts in counts.values())
    if image_count == 0:
        raise RuntimeError(f"SDNET2018 archive extracted, but no D/W/P cracked/uncracked images were found in {output_dir}.")
    layout = describe_layout(output_dir, counts, source)
    (output_dir / "LAYOUT.md").write_text(layout + "\n")
    return Sdnet2018Result(
        output_dir=output_dir,
        downloaded=True,
        source=source,
        image_count=image_count,
        counts=counts,
        message=f"SDNET2018 ready: {image_count} images from {source}; normalized={normalized}",
        archive_path=archive_path,
    )


def normalize_sdnet2018(extracted_dir: Path, output_dir: Path) -> int:
    moved = 0
    for image_path in sorted(p for p in extracted_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS):
        domain, cracked = _labels_from_path(image_path)
        if domain is None or cracked is None:
            continue
        target_dir = output_dir / domain / cracked
        target_dir.mkdir(parents=True, exist_ok=True)
        target = target_dir / image_path.name
        if target.exists() and target.stat().st_size == image_path.stat().st_size:
            continue
        if target.exists():
            target = target_dir / f"{image_path.parent.name}_{image_path.name}"
        target.write_bytes(image_path.read_bytes())
        moved += 1
    return moved


def count_sdnet2018(output_dir: Path) -> dict[str, dict[str, int]]:
    counts: dict[str, dict[str, int]] = {}
    for domain in DOMAIN_LABELS:
        domain_counts: dict[str, int] = {}
        for cracked in ("cracked", "uncracked"):
            domain_counts[cracked] = _count_images(output_dir / domain / cracked)
        if any(domain_counts.values()):
            counts[domain] = domain_counts
    return counts


def describe_layout(output_dir: Path, counts: dict[str, dict[str, int]], source: str) -> str:
    lines = [
        "# SDNET2018 detected layout",
        "",
        f"Source: {source}",
        f"Root: `{output_dir}`",
        "",
        "| code | domain | cracked | uncracked |",
        "|---|---|---:|---:|",
    ]
    for code, domain_name in DOMAIN_LABELS.items():
        domain_counts = counts.get(code, {})
        lines.append(
            f"| {code} | {domain_name} | {domain_counts.get('cracked', 0)} | "
            f"{domain_counts.get('uncracked', 0)} |"
        )
    return "\n".join(lines)


def _labels_from_path(path: Path) -> tuple[str | None, str | None]:
    parts = [part.lower() for part in path.parts]
    domain: str | None = None
    cracked: str | None = None
    for part in parts:
        upper = part.upper()
        if upper in DOMAIN_LABELS:
            domain = upper
        if upper in CRACKED_LABELS or part in CRACKED_LABELS:
            cracked = "cracked"
        if upper in UNCRACKED_LABELS or part in UNCRACKED_LABELS:
            cracked = "uncracked"
    return domain, cracked


def _stream_download(url: str, destination: Path, expected_content_type: str | None = None) -> None:
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "tarmac/0.1"}) as response:
            response.raise_for_status()
            content_type = response.headers.get("content-type", "")
            if expected_content_type and expected_content_type not in content_type:
                raise RuntimeError(f"Expected {expected_content_type}, got {content_type!r} from {url}")
            total = int(response.headers.get("content-length") or 0)
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


def _download_readme(output_dir: Path) -> None:
    try:
        response = requests.get(README_URL, timeout=60)
        response.raise_for_status()
        (output_dir / "ReadMe_SDNET2018.txt").write_text(response.text)
    except Exception as exc:
        LOGGER.warning("Could not download SDNET2018 readme: %s", exc)


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    marker = output_dir / f".{archive_path.stem}.extracted"
    if marker.exists():
        return
    with ZipFile(archive_path) as archive:
        archive.extractall(output_dir)
    marker.write_text("ok\n")


def _count_images(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for image_path in path.rglob("*") if image_path.suffix.lower() in IMAGE_EXTENSIONS)


def _manual_instructions(reason: str) -> str:
    return "\n".join(
        [
            "# SDNET2018 manual download required",
            "",
            f"Automatic download was skipped because: {reason}",
            "",
            f"Primary source: {IEEE_DATAPORT_URL}",
            "Keyless mirror attempted: https://digitalcommons.usu.edu/all_datasets/48/",
            "",
            "Download `SDNET2018.zip`, place it at `data/raw/sdnet2018/archives/SDNET2018.zip`, "
            "then re-run `UV_CACHE_DIR=.uv-cache uv run tarmac download sdnet2018`.",
        ]
    )
