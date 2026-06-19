from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

PAPER_URL = "https://www.researchgate.net/publication/319333841"
PAPER_CITATION = (
    "Yang, L., et al. (2017). Deep Concrete Inspection Using Unmanned Aerial Vehicle "
    "Towards CSSC Database. IROS 2017."
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_PATH_TOKENS = {"mask", "gt", "groundtruth", "ground_truth", "label", "annotation"}
MASK_STEM_TOKENS = ("_mask", "-mask", "_gt", "-gt", "_label", "-label", "_groundtruth", "_annotation")


@dataclass(frozen=True)
class CsscResult:
    output_dir: Path
    downloaded: bool
    pair_count: int
    pairs_path: Path
    layout_path: Path


def download_cssc(output_dir: Path = Path("data/raw/cssc")) -> CsscResult:
    """Loader for the CSSC database (Yang et al., IROS 2017).

    CSSC (Concrete Structure Spalling and Crack) is a UAV-collected concrete
    inspection dataset. The dataset is not publicly downloadable; access
    requires contacting the paper authors directly. This function scans
    ``<output_dir>`` for image/mask pairs if the data has been placed manually,
    and writes ``MANUAL_DOWNLOAD.md`` with contact and citation instructions.

    Manual acquisition:
      1. Contact the paper authors via ResearchGate: {PAPER_URL}
      2. Place the dataset under ``<output_dir>/``.
      3. Re-run ``uv run tarmac download cssc``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if _has_existing_pairs(output_dir):
        return _build_result_from_existing(output_dir)

    pairs = _scan_pairs(output_dir)
    (output_dir / "MANUAL_DOWNLOAD.md").write_text(_manual_instructions() + "\n")

    if not pairs:
        empty_pairs = output_dir / "pairs.jsonl"
        empty_pairs.write_text("")
        layout_path = output_dir / "LAYOUT.md"
        layout_path.write_text("# CSSC Database\n\nData not yet downloaded. See MANUAL_DOWNLOAD.md.\n")
        return CsscResult(
            output_dir=output_dir,
            downloaded=False,
            pair_count=0,
            pairs_path=empty_pairs,
            layout_path=layout_path,
        )

    pairs_path = output_dir / "pairs.jsonl"
    with pairs_path.open("w") as handle:
        for image_path, mask_path in pairs:
            handle.write(
                json.dumps(
                    {
                        "image_path": str(image_path.resolve()),
                        "mask_path": str(mask_path.resolve()),
                        "image_relpath": str(image_path.relative_to(output_dir)),
                        "mask_relpath": str(mask_path.relative_to(output_dir)),
                        "source_dataset": "cssc",
                    }
                )
                + "\n"
            )

    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(_describe_layout(output_dir, pairs) + "\n")
    return CsscResult(
        output_dir=output_dir,
        downloaded=True,
        pair_count=len(pairs),
        pairs_path=pairs_path,
        layout_path=layout_path,
    )


def find_cssc_pairs(raw_dir: Path = Path("data/raw/cssc")) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists():
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        ]
    return _scan_pairs(raw_dir)


def _has_existing_pairs(output_dir: Path) -> bool:
    pairs_path = output_dir / "pairs.jsonl"
    return pairs_path.exists() and pairs_path.stat().st_size > 10


def _build_result_from_existing(output_dir: Path) -> CsscResult:
    pairs_path = output_dir / "pairs.jsonl"
    count = sum(1 for line in pairs_path.read_text().splitlines() if line.strip())
    return CsscResult(
        output_dir=output_dir,
        downloaded=True,
        pair_count=count,
        pairs_path=pairs_path,
        layout_path=output_dir / "LAYOUT.md",
    )


def _scan_pairs(root: Path) -> list[tuple[Path, Path]]:
    all_files = [p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    masks: dict[str, Path] = {}
    images: dict[str, Path] = {}
    for path in sorted(all_files):
        if _looks_like_mask(path):
            masks[_normalize_stem(path.stem)] = path
        else:
            images[path.stem] = path
    return sorted(
        (image_path, masks[stem])
        for stem, image_path in images.items()
        if stem in masks
    )


def _looks_like_mask(path: Path) -> bool:
    text = " ".join(part.lower() for part in path.parts)
    return any(token in text for token in MASK_PATH_TOKENS)


def _normalize_stem(stem: str) -> str:
    result = stem.lower()
    for token in MASK_STEM_TOKENS:
        result = result.replace(token, "")
    return result


def _describe_layout(root: Path, pairs: list[tuple[Path, Path]]) -> str:
    return "\n".join([
        "# CSSC Database detected layout",
        "",
        f"Paper: {PAPER_URL}",
        f"Root: `{root}`",
        f"Pairs found: {len(pairs)}",
        "",
        "CSSC is a UAV-collected concrete structure spalling and crack dataset.",
        "Data is not publicly downloadable; access requires author contact.",
    ])


def _manual_instructions() -> str:
    return "\n".join([
        "# CSSC Database manual download required",
        "",
        "The CSSC (Concrete Structure Spalling and Crack) database is not publicly",
        "downloadable. Access requires contacting the paper authors.",
        "",
        f"Paper (ResearchGate): {PAPER_URL}",
        f"Citation: {PAPER_CITATION}",
        "",
        "Steps:",
        "  1. Contact the authors via ResearchGate or the affiliated institution.",
        "  2. Place the dataset under `data/raw/cssc/`.",
        "     Expected layout: images paired with binary mask files.",
        "  3. Re-run: `uv run tarmac download cssc`",
    ])
