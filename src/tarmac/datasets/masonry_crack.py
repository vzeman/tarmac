from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD_ID = "18458458"
IMAGES_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/masonry_crack_images.zip/content"
MASKS_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/masonry_crack_masks.zip/content"
IMAGES_SIZE_BYTES = 262_056_968  # ~262 MB
MASKS_SIZE_BYTES = 1_052_176     # ~1 MB
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# 100 high-resolution images of UK masonry (brick/stone walls) + binary crack masks.
# License: CC BY 4.0.


@dataclass(frozen=True)
class MasonryCrackResult:
    output_dir: Path
    downloaded: bool
    pair_count: int
    pairs_path: Path
    layout_path: Path


def download_masonry_crack(
    output_dir: Path = Path("data/raw/masonry_crack"),
    max_download_mb: float = 1024.0,
) -> MasonryCrackResult:
    """Download the UK masonry crack segmentation dataset from Zenodo (record 18458458).

    100 high-resolution images of brick/stone masonry with binary pixel masks.

    License: CC BY 4.0.
    DOI: 10.5281/zenodo.18458458.

    Manual fallback:
      1. Download ``images.zip`` and ``masks.zip`` from https://zenodo.org/records/18458458
      2. Place them at ``<output_dir>/archives/images.zip`` and ``<output_dir>/archives/masks.zip``.
      3. Re-run ``uv run tarmac download masonry-crack``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = output_dir / "archives"
    extracted_dir = output_dir / "_extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    images_archive = archives_dir / "masonry_crack_images.zip"
    masks_archive = archives_dir / "masonry_crack_masks.zip"

    max_bytes = max_download_mb * 1024 * 1024

    for archive_path, url, expected_size, name in (
        (images_archive, IMAGES_URL, IMAGES_SIZE_BYTES, "images.zip"),
        (masks_archive, MASKS_URL, MASKS_SIZE_BYTES, "masks.zip"),
    ):
        if archive_path.exists():
            continue
        if expected_size > max_bytes:
            reason = f"{name} ~{expected_size / 1e6:.0f} MB exceeds limit {max_download_mb:.0f} MB"
            (output_dir / "MANUAL_DOWNLOAD.md").write_text(_manual_instructions(reason) + "\n")
            return MasonryCrackResult(
                output_dir=output_dir,
                downloaded=False,
                pair_count=0,
                pairs_path=output_dir / "pairs.jsonl",
                layout_path=output_dir / "LAYOUT.md",
            )
        try:
            _stream_download(url, archive_path, expected_size)
        except Exception as exc:
            (output_dir / "MANUAL_DOWNLOAD.md").write_text(_manual_instructions(str(exc)) + "\n")
            return MasonryCrackResult(
                output_dir=output_dir,
                downloaded=False,
                pair_count=0,
                pairs_path=output_dir / "pairs.jsonl",
                layout_path=output_dir / "LAYOUT.md",
            )

    for archive_path in (images_archive, masks_archive):
        _extract_zip(archive_path, extracted_dir)

    pairs = _find_pairs(extracted_dir)
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
                        "source_dataset": "masonry_crack",
                    }
                )
                + "\n"
            )
    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(_describe_layout(extracted_dir, pairs) + "\n")
    return MasonryCrackResult(
        output_dir=output_dir,
        downloaded=True,
        pair_count=len(pairs),
        pairs_path=pairs_path,
        layout_path=layout_path,
    )


def find_masonry_crack_pairs(
    raw_dir: Path = Path("data/raw/masonry_crack"),
) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists() and pairs_path.stat().st_size > 10:
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        ]
    return _find_pairs(raw_dir / "_extracted")


def _find_pairs(root: Path) -> list[tuple[Path, Path]]:
    if not root.exists():
        return []
    # Layout: images/ and masks/ sibling dirs, matched by stem.
    images_dir = root / "images"
    masks_dir = root / "masks"
    if not images_dir.exists() or not masks_dir.exists():
        return _generic_pairs(root)
    pairs: list[tuple[Path, Path]] = []
    for img_file in sorted(images_dir.rglob("*")):
        if not img_file.is_file() or img_file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        for ext in IMAGE_EXTENSIONS:
            candidate = masks_dir / (img_file.stem + ext)
            if candidate.exists():
                pairs.append((img_file, candidate))
                break
    return pairs


def _generic_pairs(root: Path) -> list[tuple[Path, Path]]:
    all_files = sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    masks: dict[str, Path] = {}
    images: dict[str, Path] = {}
    for path in all_files:
        key = _stem_key(path)
        if _looks_like_mask(path):
            masks.setdefault(key, path)
        else:
            images.setdefault(key, path)
    return sorted(
        [(images[k], masks[k]) for k in images if k in masks],
        key=lambda pair: str(pair[0]),
    )


def _looks_like_mask(path: Path) -> bool:
    name = " ".join(path.parts).lower()
    return any(token in name for token in ("mask", "label", "groundtruth", "ground_truth", "annotation"))


def _stem_key(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("_mask", "-mask", "_label", "-label", "_gt", "-gt", "_annotation"):
        stem = stem.replace(token, "")
    return "".join(ch for ch in stem if ch.isalnum())


def _extract_zip(archive_path: Path, output_dir: Path) -> None:
    marker = output_dir / f".{archive_path.stem}.extracted"
    if marker.exists():
        return
    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall(output_dir)
    marker.write_text("ok\n")


def _stream_download(url: str, destination: Path, expected_size: int | None = None) -> None:
    if destination.exists() and expected_size and destination.stat().st_size == expected_size:
        return
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with requests.get(url, stream=True, timeout=60, headers={"User-Agent": "tarmac/0.1"}) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or expected_size or 0)
        with tmp_path.open("wb") as handle, tqdm(
            total=total or None, unit="B", unit_scale=True, desc=destination.name
        ) as progress:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    handle.write(chunk)
                    progress.update(len(chunk))
    tmp_path.replace(destination)


def _describe_layout(root: Path, pairs: list[tuple[Path, Path]]) -> str:
    return "\n".join(
        [
            "# UK Masonry Crack Segmentation detected layout",
            "",
            f"Zenodo: {ZENODO_URL}",
            f"Root: `{root}`",
            f"Pairs: {len(pairs)}",
            "",
            "License: CC BY 4.0",
            "Source: 100 UK masonry (brick/stone) images with binary crack masks.",
        ]
    )


def _manual_instructions(reason: str) -> str:
    return "\n".join(
        [
            "# UK Masonry Crack Segmentation — manual download required",
            "",
            f"Automatic download was skipped: {reason}",
            "",
            f"1. Visit {ZENODO_URL}",
            "2. Download `images.zip` (~262 MB) and `masks.zip` (~1 MB).",
            "3. Place them at:",
            "   - `data/raw/masonry_crack/archives/images.zip`",
            "   - `data/raw/masonry_crack/archives/masks.zip`",
            "4. Re-run `uv run tarmac download masonry-crack`.",
        ]
    )
