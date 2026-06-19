from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

LOGGER = logging.getLogger(__name__)

ONEDRIVE_BUNDLE_URL = (
    "https://tuprd-my.sharepoint.com/:u:/r/personal/tug13683_temple_edu/Documents/"
    "CrackDataSet/pavement%20crack%20datasets-20210103T153625Z-001.zip"
    "?csf=1&web=1&e=cnWIC3"
)
PAPER_URL = "https://ieeexplore.ieee.org/document/7533052"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

SPLIT_FOLDER_NAMES = {
    "traincrop": "train",
    "validationcrop": "val",
    "testcrop": "test",
    "train": "train",
    "val": "val",
    "validation": "val",
    "test": "test",
}
MASK_FOLDER_TOKENS = {"mask", "gt", "groundtruth", "ground_truth", "label"}


@dataclass(frozen=True)
class Crack500SegResult:
    output_dir: Path
    downloaded: bool
    image_count: int
    mask_count: int
    pair_count: int
    pairs_path: Path
    layout_path: Path


def download_crack500_seg(output_dir: Path = Path("data/raw/crack500_seg")) -> Crack500SegResult:
    """Download CRACK500 segmentation pairs (Yang et al., ICIP 2016).

    CRACK500 contains 500+ pavement images at 3264×2448 with pixel-level
    crack masks, split into train/validation/test. The dataset is distributed
    in a bundle ZIP via OneDrive from Temple University. This downloader
    attempts the OneDrive link; on failure it writes ``MANUAL_DOWNLOAD.md``
    and returns ``downloaded=False``.

    If you have the bundle ZIP or the extracted CRACK500 folder from
    fyangneil/pavement-crack-detection, place it as described in
    ``MANUAL_DOWNLOAD.md`` and re-run the command.

    Manual fallback:
      1. Download the bundle ZIP from the OneDrive link in MANUAL_DOWNLOAD.md,
         OR clone https://github.com/fyangneil/pavement-crack-detection and
         extract the CRACK500 portion.
      2. Place the archive at ``<output_dir>/archives/crack500_bundle.zip``,
         or place the extracted CRACK500 folder at ``<output_dir>/_extracted/CRACK500``.
      3. Re-run ``uv run tarmac download crack500-seg``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = output_dir / "archives"
    extracted_dir = output_dir / "_extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archives_dir / "crack500_bundle.zip"

    if _has_existing_pairs(output_dir):
        return _build_result_from_existing(output_dir)

    downloaded = archive_path.exists()
    if not downloaded:
        try:
            _stream_download(ONEDRIVE_BUNDLE_URL, archive_path)
            downloaded = True
        except Exception as exc:
            LOGGER.warning("CRACK500 segmentation download skipped: %s", exc)
            instructions = _manual_instructions(str(exc))
            (output_dir / "MANUAL_DOWNLOAD.md").write_text(instructions + "\n")
            empty_pairs = output_dir / "pairs.jsonl"
            empty_pairs.write_text("")
            layout_path = output_dir / "LAYOUT.md"
            layout_path.write_text(f"# CRACK500 segmentation\n\nDownload skipped: {exc}\n")
            return Crack500SegResult(
                output_dir=output_dir,
                downloaded=False,
                image_count=0,
                mask_count=0,
                pair_count=0,
                pairs_path=empty_pairs,
                layout_path=layout_path,
            )

    _extract_archive(archive_path, extracted_dir)
    pairs = _find_pairs(extracted_dir)

    pairs_path = output_dir / "pairs.jsonl"
    with pairs_path.open("w") as handle:
        for image_path, mask_path, split in pairs:
            handle.write(
                json.dumps(
                    {
                        "image_path": str(image_path.resolve()),
                        "mask_path": str(mask_path.resolve()),
                        "image_relpath": str(image_path.relative_to(output_dir)),
                        "mask_relpath": str(mask_path.relative_to(output_dir)),
                        "source_dataset": "crack500",
                        "split": split,
                    }
                )
                + "\n"
            )

    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(_describe_layout(extracted_dir, pairs) + "\n")
    return Crack500SegResult(
        output_dir=output_dir,
        downloaded=True,
        image_count=len({p[0] for p in pairs}),
        mask_count=len({p[1] for p in pairs}),
        pair_count=len(pairs),
        pairs_path=pairs_path,
        layout_path=layout_path,
    )


def find_crack500_seg_pairs(raw_dir: Path = Path("data/raw/crack500_seg")) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists():
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        ]
    extracted_dir = raw_dir / "_extracted"
    return [(img, mask) for img, mask, _ in _find_pairs(extracted_dir)]


def _has_existing_pairs(output_dir: Path) -> bool:
    pairs_path = output_dir / "pairs.jsonl"
    return pairs_path.exists() and pairs_path.stat().st_size > 10


def _build_result_from_existing(output_dir: Path) -> Crack500SegResult:
    pairs_path = output_dir / "pairs.jsonl"
    pairs = [
        (Path(row["image_path"]), Path(row["mask_path"]))
        for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
    ]
    return Crack500SegResult(
        output_dir=output_dir,
        downloaded=True,
        image_count=len({p[0] for p in pairs}),
        mask_count=len({p[1] for p in pairs}),
        pair_count=len(pairs),
        pairs_path=pairs_path,
        layout_path=output_dir / "LAYOUT.md",
    )


def _find_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    crack500_root = _find_crack500_root(root)
    if crack500_root is None:
        return []

    pairs: list[tuple[Path, Path, str]] = []
    for split_folder, split_name in SPLIT_FOLDER_NAMES.items():
        split_dir = crack500_root / split_folder
        if not split_dir.exists():
            continue
        image_dir, mask_dir = _find_image_mask_dirs(split_dir)
        if image_dir is None or mask_dir is None:
            continue
        images = {
            p.stem: p
            for p in sorted(image_dir.rglob("*"))
            if p.suffix.lower() in IMAGE_EXTENSIONS
        }
        masks = {
            _mask_stem(p): p
            for p in sorted(mask_dir.rglob("*"))
            if p.suffix.lower() in IMAGE_EXTENSIONS
        }
        for stem, image_path in images.items():
            mask_path = masks.get(stem)
            if mask_path is not None:
                pairs.append((image_path, mask_path, split_name))
    return pairs


def _find_crack500_root(root: Path) -> Path | None:
    for candidate_name in ("CRACK500", "crack500", "Crack500"):
        candidate = root / candidate_name
        if candidate.exists():
            return candidate
    for candidate in root.rglob("*"):
        if candidate.is_dir() and candidate.name.lower() == "crack500":
            return candidate
    return None


def _find_image_mask_dirs(split_dir: Path) -> tuple[Path | None, Path | None]:
    children = {d.name.lower(): d for d in split_dir.iterdir() if d.is_dir()}
    image_dir = children.get("image") or children.get("images") or children.get("img")
    mask_dir = None
    for token in MASK_FOLDER_TOKENS:
        if token in children:
            mask_dir = children[token]
            break
    if image_dir is None:
        image_dir = split_dir
    if mask_dir is None:
        for token in MASK_FOLDER_TOKENS:
            for child in split_dir.iterdir():
                if child.is_dir() and token in child.name.lower():
                    mask_dir = child
                    break
            if mask_dir is not None:
                break
    return image_dir, mask_dir


def _mask_stem(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("_mask", "-mask", "_gt", "-gt", "_label", "-label", "_groundtruth"):
        stem = stem.replace(token, "")
    return stem


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    marker = output_dir / ".crack500_seg.extracted"
    if marker.exists():
        return
    with ZipFile(archive_path) as archive:
        archive.extractall(output_dir)
    marker.write_text("ok\n")


def _stream_download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    try:
        with requests.get(url, stream=True, timeout=120, headers={"User-Agent": "tarmac/0.1"}) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length") or 0)
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


def _describe_layout(root: Path, pairs: list[tuple[Path, Path, str]]) -> str:
    split_counts = {}
    for _, _, split in pairs:
        split_counts[split] = split_counts.get(split, 0) + 1
    splits_str = ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items()))
    return "\n".join([
        "# CRACK500 segmentation detected layout",
        "",
        f"Paper: {PAPER_URL}",
        f"Root: `{root}`",
        f"Total pairs: {len(pairs)} ({splits_str})",
        "",
        "Pairing rule: matched by stem within each split folder's image/ and mask/ subdirectories.",
    ])


def _manual_instructions(reason: str) -> str:
    return "\n".join([
        "# CRACK500 segmentation manual download required",
        "",
        f"Automatic download was skipped because: {reason}",
        "",
        "Option A — Bundle ZIP from Temple University OneDrive:",
        f"  URL: {ONEDRIVE_BUNDLE_URL}",
        "  Place at: `data/raw/crack500_seg/archives/crack500_bundle.zip`",
        "",
        "Option B — Clone the GitHub repo and extract:",
        "  git clone https://github.com/fyangneil/pavement-crack-detection",
        "  Extract the CRACK500 folder to: `data/raw/crack500_seg/_extracted/CRACK500/`",
        "",
        "Then re-run: `uv run tarmac download crack500-seg`",
        "",
        f"Paper: {PAPER_URL}",
    ])
