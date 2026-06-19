from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD_ID = "6383044"
ARCHIVE_URL = f"https://zenodo.org/api/records/{ZENODO_RECORD_ID}/files/data.zip/content"
ARCHIVE_SIZE_BYTES = 1_233_000_000  # ~1.15 GB
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# FIND dataset: 2500 image patches, 4 modalities (RGB, NIR, depth, 3D-fused), + binary masks.
# Fused modality used as primary.
# License: CC BY 4.0. Reference: Benz & Rodehorst, BMVC 2022.


@dataclass(frozen=True)
class FindCrackResult:
    output_dir: Path
    downloaded: bool
    pair_count: int
    pairs_path: Path
    layout_path: Path


def download_find_crack(
    output_dir: Path = Path("data/raw/find_crack"),
    max_download_mb: float = 2048.0,
) -> FindCrackResult:
    """Download the FIND crack segmentation dataset from Zenodo (record 6383044).

    2500 image patches with 4 modalities (RGB, NIR, depth, 3D-fused) and binary masks.
    Primary modality is 'fused' (multi-modal fusion).

    License: CC BY 4.0.
    Reference: Benz & Rodehorst, BMVC 2022. DOI 10.5281/zenodo.6383044.

    Manual fallback:
      1. Download ``data.zip`` from https://zenodo.org/records/6383044
      2. Place it at ``<output_dir>/archives/data.zip``.
      3. Re-run ``uv run tarmac download find-crack``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = output_dir / "archives"
    extracted_dir = output_dir / "_extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archives_dir / "data.zip"

    if not archive_path.exists():
        max_bytes = max_download_mb * 1024 * 1024
        if ARCHIVE_SIZE_BYTES > max_bytes:
            reason = f"archive ~{ARCHIVE_SIZE_BYTES / 1e9:.1f} GB exceeds limit {max_download_mb:.0f} MB"
            (output_dir / "MANUAL_DOWNLOAD.md").write_text(_manual_instructions(reason) + "\n")
            return FindCrackResult(
                output_dir=output_dir,
                downloaded=False,
                pair_count=0,
                pairs_path=output_dir / "pairs.jsonl",
                layout_path=output_dir / "LAYOUT.md",
            )
        try:
            _stream_download(ARCHIVE_URL, archive_path, ARCHIVE_SIZE_BYTES)
        except Exception as exc:
            (output_dir / "MANUAL_DOWNLOAD.md").write_text(_manual_instructions(str(exc)) + "\n")
            return FindCrackResult(
                output_dir=output_dir,
                downloaded=False,
                pair_count=0,
                pairs_path=output_dir / "pairs.jsonl",
                layout_path=output_dir / "LAYOUT.md",
            )

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
                        "source_dataset": "find_crack",
                    }
                )
                + "\n"
            )
    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(_describe_layout(extracted_dir, pairs) + "\n")
    return FindCrackResult(
        output_dir=output_dir,
        downloaded=True,
        pair_count=len(pairs),
        pairs_path=pairs_path,
        layout_path=layout_path,
    )


def find_find_crack_pairs(
    raw_dir: Path = Path("data/raw/find_crack"),
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
    # FIND layout: data/{split}/fused/*.png (images), data/{split}/GT/*.png (masks)
    pairs: list[tuple[Path, Path]] = []
    for gt_file in sorted(root.rglob("*")):
        if not gt_file.is_file() or gt_file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        parts_lower = [p.lower() for p in gt_file.parts]
        if not any(p in ("gt", "mask", "label", "groundtruth", "ground_truth") for p in parts_lower):
            continue
        # Prefer fused sibling, fall back to RGB
        for modality in ("fused", "RGB", "rgb"):
            candidate = gt_file.parent.parent / modality / gt_file.name
            if candidate.exists():
                pairs.append((candidate, gt_file))
                break
    return sorted(pairs, key=lambda p: str(p[0]))


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
    img_dirs = sorted({str(img.parent.relative_to(root)) for img, _ in pairs})
    return "\n".join(
        [
            "# FIND Crack Segmentation Dataset detected layout",
            "",
            f"Zenodo: {ZENODO_URL}",
            f"Root: `{root}`",
            f"Pairs: {len(pairs)}",
            f"Image directories (sample): {', '.join(img_dirs[:6])}",
            "",
            "License: CC BY 4.0",
            "Source: Benz & Rodehorst, BMVC 2022. Multi-modal pavement crack dataset.",
        ]
    )


def _manual_instructions(reason: str) -> str:
    return "\n".join(
        [
            "# FIND Crack Segmentation — manual download required",
            "",
            f"Automatic download was skipped: {reason}",
            "",
            f"1. Visit {ZENODO_URL}",
            "2. Download `data.zip` (~1.15 GB).",
            "3. Place it at `data/raw/find_crack/archives/data.zip`.",
            "4. Re-run `uv run tarmac download find-crack`.",
        ]
    )
