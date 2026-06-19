from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

ZENODO_RECORD_ID = "18010179"
ZENODO_URL = f"https://zenodo.org/records/{ZENODO_RECORD_ID}"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# PAGG-Net crack segmentation dataset: Zenodo 18010179, ~30 GB total.
# Large dataset — manual download by default; auto-download gated by max_download_mb.
# License: CC BY 4.0.


@dataclass(frozen=True)
class PaggNetCrackResult:
    output_dir: Path
    downloaded: bool
    pair_count: int
    pairs_path: Path
    layout_path: Path


def download_paggnet_crack(
    output_dir: Path = Path("data/raw/paggnet_crack"),
    max_download_mb: float = 2048.0,
) -> PaggNetCrackResult:
    """Download the PAGG-Net crack segmentation dataset from Zenodo (record 18010179).

    Large pavement crack segmentation dataset (~30 GB).
    Pass ``max_download_mb`` to override the download guard.

    License: CC BY 4.0.
    DOI: 10.5281/zenodo.18010179.

    Manual fallback:
      1. Visit https://zenodo.org/records/18010179
      2. Download the archive(s) and place under ``<output_dir>/archives/``.
      3. Extract to ``<output_dir>/_extracted/``.
      4. Re-run ``uv run tarmac download paggnet-crack``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = output_dir / "archives"
    extracted_dir = output_dir / "_extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)

    # If pairs.jsonl already exists from a prior manual extraction, just scan.
    existing_pairs_path = output_dir / "pairs.jsonl"
    if existing_pairs_path.exists() and existing_pairs_path.stat().st_size > 10:
        pairs = find_paggnet_crack_pairs(output_dir)
        return PaggNetCrackResult(
            output_dir=output_dir,
            downloaded=True,
            pair_count=len(pairs),
            pairs_path=existing_pairs_path,
            layout_path=output_dir / "LAYOUT.md",
        )

    # If _extracted already has images, build pairs from what's there.
    if extracted_dir.exists() and any(extracted_dir.rglob("*.png")):
        pairs = _find_pairs(extracted_dir)
        _write_pairs(pairs, extracted_dir, output_dir)
        layout_path = output_dir / "LAYOUT.md"
        layout_path.write_text(_describe_layout(extracted_dir, pairs) + "\n")
        return PaggNetCrackResult(
            output_dir=output_dir,
            downloaded=True,
            pair_count=len(pairs),
            pairs_path=output_dir / "pairs.jsonl",
            layout_path=layout_path,
        )

    # Try to fetch the Zenodo file listing and download if within budget.
    try:
        file_list = _fetch_zenodo_file_list(ZENODO_RECORD_ID)
    except Exception as exc:
        (output_dir / "MANUAL_DOWNLOAD.md").write_text(_manual_instructions(str(exc)) + "\n")
        return _no_download_result(output_dir)

    total_bytes = sum(f.get("size", 0) for f in file_list)
    max_bytes = max_download_mb * 1024 * 1024
    if total_bytes > max_bytes:
        reason = (
            f"total dataset ~{total_bytes / 1e9:.1f} GB exceeds limit {max_download_mb:.0f} MB; "
            f"use --max-download-mb to raise or download manually"
        )
        (output_dir / "MANUAL_DOWNLOAD.md").write_text(_manual_instructions(reason) + "\n")
        return _no_download_result(output_dir)

    # Download and extract each file.
    for file_info in file_list:
        fname = file_info["key"]
        url = file_info["links"]["content"]
        size = file_info.get("size")
        archive_path = archives_dir / fname
        try:
            _stream_download(url, archive_path, size)
        except Exception as exc:
            (output_dir / "MANUAL_DOWNLOAD.md").write_text(_manual_instructions(str(exc)) + "\n")
            return _no_download_result(output_dir)
        if archive_path.suffix.lower() == ".zip":
            _extract_zip(archive_path, extracted_dir)

    pairs = _find_pairs(extracted_dir)
    _write_pairs(pairs, extracted_dir, output_dir)
    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(_describe_layout(extracted_dir, pairs) + "\n")
    return PaggNetCrackResult(
        output_dir=output_dir,
        downloaded=True,
        pair_count=len(pairs),
        pairs_path=output_dir / "pairs.jsonl",
        layout_path=layout_path,
    )


def find_paggnet_crack_pairs(
    raw_dir: Path = Path("data/raw/paggnet_crack"),
) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists() and pairs_path.stat().st_size > 10:
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        ]
    return _find_pairs(raw_dir / "_extracted")


def _no_download_result(output_dir: Path) -> PaggNetCrackResult:
    return PaggNetCrackResult(
        output_dir=output_dir,
        downloaded=False,
        pair_count=0,
        pairs_path=output_dir / "pairs.jsonl",
        layout_path=output_dir / "LAYOUT.md",
    )


def _write_pairs(pairs: list[tuple[Path, Path]], extracted_dir: Path, output_dir: Path) -> None:
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
                        "source_dataset": "paggnet_crack",
                    }
                )
                + "\n"
            )


def _find_pairs(root: Path) -> list[tuple[Path, Path]]:
    if not root.exists():
        return []
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
    return any(token in name for token in ("mask", "label", "groundtruth", "ground_truth", "gt", "annotation"))


def _stem_key(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("_mask", "-mask", "_label", "-label", "_gt", "-gt", "_annotation"):
        stem = stem.replace(token, "")
    return "".join(ch for ch in stem if ch.isalnum())


def _fetch_zenodo_file_list(record_id: str) -> list[dict]:
    url = f"https://zenodo.org/api/records/{record_id}"
    resp = requests.get(url, timeout=30, headers={"User-Agent": "tarmac/0.1"})
    resp.raise_for_status()
    data = resp.json()
    return data.get("files", [])


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
            "# PAGG-Net Crack Segmentation detected layout",
            "",
            f"Zenodo: {ZENODO_URL}",
            f"Root: `{root}`",
            f"Pairs: {len(pairs)}",
            "",
            "License: CC BY 4.0",
            "Source: PAGG-Net pavement crack segmentation dataset.",
        ]
    )


def _manual_instructions(reason: str) -> str:
    return "\n".join(
        [
            "# PAGG-Net Crack Segmentation — manual download required",
            "",
            f"Automatic download was skipped: {reason}",
            "",
            f"1. Visit {ZENODO_URL}",
            "2. Download the archive files (~30 GB total).",
            "3. Place archives at `data/raw/paggnet_crack/archives/`.",
            "4. Extract contents to `data/raw/paggnet_crack/_extracted/`.",
            "5. Re-run `uv run tarmac download paggnet-crack`.",
            "",
            "Or raise the download limit:",
            "  uv run tarmac download paggnet-crack --max-download-mb 35000",
        ]
    )
