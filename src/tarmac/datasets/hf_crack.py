from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# HuggingFace dataset: Mr-Perfectuz/crack
# 3,378 images + masks across 3 subsets: BJN260, Rain365, Sun520.
# License: MIT.

HF_REPO_ID = "Mr-Perfectuz/crack"
SUBSETS = ["BJN260", "Rain365", "Sun520"]


@dataclass(frozen=True)
class HfCrackResult:
    output_dir: Path
    downloaded: bool
    pair_count: int
    pairs_path: Path
    layout_path: Path


def download_hf_crack(
    output_dir: Path = Path("data/raw/hf_crack"),
) -> HfCrackResult:
    """Download the Mr-Perfectuz/crack dataset from HuggingFace Hub.

    3,378 crack images with segmentation masks across 3 subsets (BJN260, Rain365, Sun520).

    License: MIT.
    HuggingFace: https://huggingface.co/datasets/Mr-Perfectuz/crack

    Manual fallback:
      Run: ``huggingface-cli download Mr-Perfectuz/crack --repo-type dataset --local-dir <output_dir>/_hf``
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    hf_dir = output_dir / "_hf"

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        reason = "huggingface_hub not installed (install transformers or huggingface_hub)"
        (output_dir / "MANUAL_DOWNLOAD.md").write_text(_manual_instructions(reason) + "\n")
        return HfCrackResult(
            output_dir=output_dir,
            downloaded=False,
            pair_count=0,
            pairs_path=output_dir / "pairs.jsonl",
            layout_path=output_dir / "LAYOUT.md",
        )

    try:
        snapshot_download(
            repo_id=HF_REPO_ID,
            repo_type="dataset",
            local_dir=str(hf_dir),
        )
    except Exception as exc:
        (output_dir / "MANUAL_DOWNLOAD.md").write_text(_manual_instructions(str(exc)) + "\n")
        return HfCrackResult(
            output_dir=output_dir,
            downloaded=False,
            pair_count=0,
            pairs_path=output_dir / "pairs.jsonl",
            layout_path=output_dir / "LAYOUT.md",
        )

    pairs = _find_pairs(hf_dir)
    pairs_path = output_dir / "pairs.jsonl"
    with pairs_path.open("w") as handle:
        for image_path, mask_path in pairs:
            handle.write(
                json.dumps(
                    {
                        "image_path": str(image_path),
                        "mask_path": str(mask_path),
                        "image_relpath": str(image_path.relative_to(hf_dir)),
                        "mask_relpath": str(mask_path.relative_to(hf_dir)),
                        "source_dataset": "hf_crack",
                    }
                )
                + "\n"
            )
    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(_describe_layout(hf_dir, pairs) + "\n")
    return HfCrackResult(
        output_dir=output_dir,
        downloaded=True,
        pair_count=len(pairs),
        pairs_path=pairs_path,
        layout_path=layout_path,
    )


def find_hf_crack_pairs(
    raw_dir: Path = Path("data/raw/hf_crack"),
) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists() and pairs_path.stat().st_size > 10:
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        ]
    return _find_pairs(raw_dir / "_hf")


def _find_pairs(root: Path) -> list[tuple[Path, Path]]:
    if not root.exists():
        return []
    # Expected layout per subset: {subset}/images/ and {subset}/masks/ (or GT/).
    pairs: list[tuple[Path, Path]] = []
    for subset in SUBSETS:
        subset_dir = root / subset
        if not subset_dir.exists():
            # Also try top-level images/masks
            subset_dir = root
        images_dir = subset_dir / "images"
        for masks_candidate in ("masks", "GT", "gt", "mask", "label"):
            masks_dir = subset_dir / masks_candidate
            if images_dir.exists() and masks_dir.exists():
                for img_file in sorted(images_dir.rglob("*")):
                    if not img_file.is_file() or img_file.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    for ext in IMAGE_EXTENSIONS:
                        candidate = masks_dir / (img_file.stem + ext)
                        if candidate.exists():
                            pairs.append((img_file, candidate))
                            break
                break
        else:
            # Generic stem-matching fallback within subset
            if subset_dir.exists():
                pairs.extend(_generic_pairs(subset_dir))
    if not pairs:
        pairs = _generic_pairs(root)
    return sorted(set(pairs), key=lambda p: str(p[0]))


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
    return any(token in name for token in ("mask", "label", "groundtruth", "ground_truth", "gt", "annotation"))


def _stem_key(path: Path) -> str:
    stem = path.stem.lower()
    for token in ("_mask", "-mask", "_label", "-label", "_gt", "-gt", "_annotation"):
        stem = stem.replace(token, "")
    return "".join(ch for ch in stem if ch.isalnum())


def _describe_layout(root: Path, pairs: list[tuple[Path, Path]]) -> str:
    return "\n".join(
        [
            "# HuggingFace Mr-Perfectuz/crack detected layout",
            "",
            "HuggingFace: https://huggingface.co/datasets/Mr-Perfectuz/crack",
            f"Root: `{root}`",
            f"Pairs: {len(pairs)}",
            f"Subsets: {', '.join(SUBSETS)}",
            "",
            "License: MIT",
            "Source: BJN260, Rain365, Sun520 subsets.",
        ]
    )


def _manual_instructions(reason: str) -> str:
    return "\n".join(
        [
            "# HuggingFace Mr-Perfectuz/crack — manual download required",
            "",
            f"Automatic download was skipped: {reason}",
            "",
            "1. Run:",
            "   huggingface-cli download Mr-Perfectuz/crack --repo-type dataset --local-dir data/raw/hf_crack/_hf",
            "2. Re-run `uv run tarmac download hf-crack`.",
        ]
    )
