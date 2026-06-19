from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests
from tqdm import tqdm

DATASET_ID = "jwsn7tfbrp"
VERSION = 1
ARCHIVE_DOWNLOAD_URL = (
    "https://data.mendeley.com/public-files/datasets/jwsn7tfbrp/files/"
    "88e685a6-e3c5-423d-845f-89e35a457867/file_downloaded"
)
ARCHIVE_SIZE_BYTES = 745_914_150
PAPER_URL = "https://www.sciencedirect.com/science/article/pii/S0950061819300545"
MENDELEY_URL = f"https://data.mendeley.com/datasets/{DATASET_ID}/{VERSION}"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# Archive: concreteCrackSegmentationDataset.rar (~711 MB)
# Layout after extraction: images are full-resolution JPG, masks are B/W alpha maps (PNG or JPG).
# Source: buildings at Middle East Technical University (METU), Turkey.
# License: CC BY 4.0


@dataclass(frozen=True)
class MetuCrackSegResult:
    output_dir: Path
    downloaded: bool
    image_count: int
    mask_count: int
    pair_count: int
    pairs_path: Path
    layout_path: Path


def download_metu_crack_seg(
    output_dir: Path = Path("data/raw/metu_crack_seg"),
) -> MetuCrackSegResult:
    """Download the METU concrete crack segmentation dataset from Mendeley.

    458 high-resolution images with binary pixel-level segmentation masks.
    Sourced from concrete surfaces of buildings at Middle East Technical University.

    License: CC BY 4.0.
    Reference: Ozgenel & Sorguç, ISARC 2018; Mendeley DOI 10.17632/jwsn7tfbrp.1.

    Manual fallback:
      1. Download ``concreteCrackSegmentationDataset.rar`` from
         https://data.mendeley.com/datasets/jwsn7tfbrp/1
      2. Place it at ``<output_dir>/archives/concreteCrackSegmentationDataset.rar``.
      3. Re-run ``uv run tarmac download metu-crack-seg``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = output_dir / "archives"
    extracted_dir = output_dir / "_extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archives_dir / "concreteCrackSegmentationDataset.rar"

    if not archive_path.exists():
        try:
            _stream_download(ARCHIVE_DOWNLOAD_URL, archive_path, ARCHIVE_SIZE_BYTES)
        except Exception as exc:
            instructions = _manual_instructions(str(exc))
            (output_dir / "MANUAL_DOWNLOAD.md").write_text(instructions + "\n")
            return MetuCrackSegResult(
                output_dir=output_dir,
                downloaded=False,
                image_count=0,
                mask_count=0,
                pair_count=0,
                pairs_path=output_dir / "pairs.jsonl",
                layout_path=output_dir / "LAYOUT.md",
            )

    _extract_rar(archive_path, extracted_dir)
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
                        "source_dataset": "metu_crack_seg",
                    }
                )
                + "\n"
            )
    layout = _describe_layout(extracted_dir, pairs)
    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(layout + "\n")
    return MetuCrackSegResult(
        output_dir=output_dir,
        downloaded=True,
        image_count=len({p[0] for p in pairs}),
        mask_count=len({p[1] for p in pairs}),
        pair_count=len(pairs),
        pairs_path=pairs_path,
        layout_path=layout_path,
    )


def find_metu_crack_seg_pairs(
    raw_dir: Path = Path("data/raw/metu_crack_seg"),
) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists():
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        ]
    return _find_pairs(raw_dir / "_extracted")


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
    return any(token in name for token in ("mask", "label", "groundtruth", "ground_truth", "alpha", "annotation"))


def _stem_key(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("_mask", "-mask", "_label", "-label", "_alpha", "_gt", "-gt", "_annotation"):
        stem = stem.replace(token, "")
    return "".join(ch for ch in stem if ch.isalnum())


def _extract_rar(archive_path: Path, output_dir: Path) -> None:
    marker = output_dir / f".{archive_path.stem}.extracted"
    if marker.exists():
        return
    extractor = shutil.which("bsdtar") or shutil.which("unrar") or shutil.which("7z")
    if extractor is None:
        raise RuntimeError(
            f"No RAR extractor found. Install bsdtar, unrar, or 7z, then retry: {archive_path}"
        )
    name = Path(extractor).name
    if name == "bsdtar":
        cmd = ["bsdtar", "-x", "-f", str(archive_path), "-C", str(output_dir)]
    elif name == "unrar":
        cmd = ["unrar", "x", "-y", str(archive_path), str(output_dir) + "/"]
    else:
        cmd = ["7z", "x", str(archive_path), f"-o{output_dir}", "-y"]
    subprocess.run(cmd, check=True)
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
    image_dirs = sorted({str(img.parent.relative_to(root)) for img, _ in pairs})
    mask_dirs = sorted({str(msk.parent.relative_to(root)) for _, msk in pairs})
    return "\n".join(
        [
            "# METU Concrete Crack Segmentation detected layout",
            "",
            f"Mendeley: {MENDELEY_URL}",
            f"Root: `{root}`",
            f"Pairs: {len(pairs)}",
            f"Image directories: {', '.join(image_dirs[:8])}",
            f"Mask directories: {', '.join(mask_dirs[:8])}",
            "",
            "License: CC BY 4.0",
            "Source: METU buildings, Turkey. Ozgenel & Sorguç, ISARC 2018.",
        ]
    )


def _manual_instructions(reason: str) -> str:
    return "\n".join(
        [
            "# METU Concrete Crack Segmentation — manual download required",
            "",
            f"Automatic download was skipped: {reason}",
            "",
            f"1. Visit {MENDELEY_URL}",
            "2. Download `concreteCrackSegmentationDataset.rar`.",
            "3. Place it at `data/raw/metu_crack_seg/archives/concreteCrackSegmentationDataset.rar`.",
            "4. Re-run `uv run tarmac download metu-crack-seg`.",
            "",
            "Requires bsdtar, unrar, or 7z for extraction.",
        ]
    )
