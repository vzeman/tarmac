from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

GOOGLE_DRIVE_URL = "https://drive.google.com/open?id=1xrOqv0-3uMHjZyEUrerOYiYXW_E8SUMP"
GITHUB_URL = "https://github.com/khanhha/crack_segmentation"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

KNOWN_PREFIXES = {
    "CFD": "cfd",
    "CRACK500": "crack500",
    "AEL": "ael",
    "cracktree200": "cracktree200",
    "CrackTree200": "cracktree200",
    "DeepCrack": "deepcrack",
    "GAPS384": "gaps384",
    "Rissbilder": "rissbilder",
    "noncrack": "noncrack",
    "Volker": "volker",
    "EugenMuller": "eugenmuller",
}


@dataclass(frozen=True)
class Khanh11kResult:
    output_dir: Path
    downloaded: bool
    pair_count: int
    pairs_path: Path
    layout_path: Path


def download_khanh11k(output_dir: Path = Path("data/raw/khanh11k")) -> Khanh11kResult:
    """Loader for the Khanh11k merged crack segmentation dataset.

    This dataset merges ~11,200 images from 12 publicly available crack
    segmentation datasets (CFD, CRACK500, AEL, CrackTree200, DeepCrack,
    GAPS384, and others), all resized to 448×448. Images and masks are
    organized in ``images/train/``, ``images/test/``, ``masks/train/``,
    ``masks/test/`` folders. The source sub-dataset can be inferred from
    the filename prefix (e.g. ``CFD_001.jpg`` → ``cfd``).

    The data is hosted on Google Drive and cannot be downloaded
    programmatically without authentication. This function writes
    ``MANUAL_DOWNLOAD.md`` unless data is already present.

    Manual download:
      1. Download the dataset ZIP from Google Drive: {GOOGLE_DRIVE_URL}
      2. Extract so that ``images/train/`` exists under ``<output_dir>/``.
      3. Re-run ``uv run tarmac download khanh11k``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if _has_existing_pairs(output_dir):
        return _build_result_from_existing(output_dir)

    pairs = _scan_pairs(output_dir)
    if not pairs:
        instructions = _manual_instructions()
        (output_dir / "MANUAL_DOWNLOAD.md").write_text(instructions + "\n")
        empty_pairs = output_dir / "pairs.jsonl"
        empty_pairs.write_text("")
        layout_path = output_dir / "LAYOUT.md"
        layout_path.write_text("# Khanh11k\n\nData not yet downloaded. See MANUAL_DOWNLOAD.md.\n")
        return Khanh11kResult(
            output_dir=output_dir,
            downloaded=False,
            pair_count=0,
            pairs_path=empty_pairs,
            layout_path=layout_path,
        )

    pairs_path = output_dir / "pairs.jsonl"
    with pairs_path.open("w") as handle:
        for image_path, mask_path, split, sub_dataset in pairs:
            handle.write(
                json.dumps(
                    {
                        "image_path": str(image_path.resolve()),
                        "mask_path": str(mask_path.resolve()),
                        "image_relpath": str(image_path.relative_to(output_dir)),
                        "mask_relpath": str(mask_path.relative_to(output_dir)),
                        "source_dataset": "khanh11k",
                        "sub_dataset": sub_dataset,
                        "split": split,
                    }
                )
                + "\n"
            )

    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(_describe_layout(output_dir, pairs) + "\n")
    return Khanh11kResult(
        output_dir=output_dir,
        downloaded=True,
        pair_count=len(pairs),
        pairs_path=pairs_path,
        layout_path=layout_path,
    )


def find_khanh11k_pairs(raw_dir: Path = Path("data/raw/khanh11k")) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists():
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        ]
    return [(img, mask) for img, mask, _, _ in _scan_pairs(raw_dir)]


def _has_existing_pairs(output_dir: Path) -> bool:
    pairs_path = output_dir / "pairs.jsonl"
    return pairs_path.exists() and pairs_path.stat().st_size > 10


def _build_result_from_existing(output_dir: Path) -> Khanh11kResult:
    pairs_path = output_dir / "pairs.jsonl"
    count = sum(1 for line in pairs_path.read_text().splitlines() if line.strip())
    return Khanh11kResult(
        output_dir=output_dir,
        downloaded=True,
        pair_count=count,
        pairs_path=pairs_path,
        layout_path=output_dir / "LAYOUT.md",
    )


def _scan_pairs(root: Path) -> list[tuple[Path, Path, str, str]]:
    pairs: list[tuple[Path, Path, str, str]] = []
    for split_name in ("train", "test"):
        image_dir = root / "images" / split_name
        mask_dir = root / "masks" / split_name
        if not image_dir.exists() or not mask_dir.exists():
            continue
        images = {
            p.stem: p
            for p in sorted(image_dir.rglob("*"))
            if p.suffix.lower() in IMAGE_EXTENSIONS
        }
        masks = {
            p.stem: p
            for p in sorted(mask_dir.rglob("*"))
            if p.suffix.lower() in IMAGE_EXTENSIONS
        }
        for stem, image_path in images.items():
            if stem in masks:
                sub_dataset = _infer_sub_dataset(stem)
                pairs.append((image_path, masks[stem], split_name, sub_dataset))
    return pairs


def _infer_sub_dataset(stem: str) -> str:
    for prefix, label in KNOWN_PREFIXES.items():
        if stem.startswith(prefix):
            return label
    return "unknown"


def _describe_layout(root: Path, pairs: list[tuple[Path, Path, str, str]]) -> str:
    split_counts: dict[str, int] = {}
    sub_counts: dict[str, int] = {}
    for _, _, split, sub in pairs:
        split_counts[split] = split_counts.get(split, 0) + 1
        sub_counts[sub] = sub_counts.get(sub, 0) + 1
    splits_str = ", ".join(f"{k}={v}" for k, v in sorted(split_counts.items()))
    subs_str = ", ".join(f"{k}={v}" for k, v in sorted(sub_counts.items(), key=lambda x: -x[1]))
    return "\n".join([
        "# Khanh11k merged crack segmentation dataset",
        "",
        f"GitHub: {GITHUB_URL}",
        f"Root: `{root}`",
        f"Total pairs: {len(pairs)} ({splits_str})",
        f"Sub-datasets: {subs_str}",
        "",
        "All images resized to 448×448.",
        "Sub-dataset inferred from filename prefix (e.g. CFD_xxx → cfd).",
    ])


def _manual_instructions() -> str:
    return "\n".join([
        "# Khanh11k manual download required",
        "",
        "The dataset is hosted on Google Drive and cannot be downloaded without",
        "browser authentication.",
        "",
        f"1. Download the dataset ZIP from: {GOOGLE_DRIVE_URL}",
        "2. Extract so the following structure exists:",
        "     data/raw/khanh11k/images/train/",
        "     data/raw/khanh11k/images/test/",
        "     data/raw/khanh11k/masks/train/",
        "     data/raw/khanh11k/masks/test/",
        "3. Re-run: `uv run tarmac download khanh11k`",
        "",
        f"GitHub: {GITHUB_URL}",
        "Contains ~11,200 images from 12 crack segmentation datasets (CFD, CRACK500,",
        "AEL, CrackTree200, DeepCrack, GAPS384, and others), all 448×448 pixels.",
        "Please cite the original dataset papers when using this data.",
    ])
