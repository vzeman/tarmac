from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

CRACKTREE260_ONEDRIVE = "https://1drv.ms/f/s!AittnGm6vRKLyiQUk3ViLu8L9Wzb"
CRKWH100_ONEDRIVE = "https://1drv.ms/f/s!AittnGm6vRKLtylBkxVXw5arGn6R"
PAPER_URL = "https://ieeexplore.ieee.org/document/8517148"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_STEM_TOKENS = ("_mask", "-mask", "_gt", "-gt", "_label", "-label", "_groundtruth", "_annotation")


@dataclass(frozen=True)
class CrackTree260Result:
    output_dir: Path
    downloaded: bool
    cracktree260_pairs: int
    crkwh100_pairs: int
    pairs_path: Path
    layout_path: Path


def download_cracktree260(output_dir: Path = Path("data/raw/cracktree260")) -> CrackTree260Result:
    """Loader for CrackTree260 and CRKWH100 (Zou et al., IEEE T-IP 2018).

    Both datasets are distributed via OneDrive shared folders from
    qinnzou/DeepCrack. OneDrive shared-folder links require browser
    authentication and cannot be downloaded programmatically; this function
    writes ``MANUAL_DOWNLOAD.md`` with instructions and returns
    ``downloaded=False`` unless the data is already present under
    ``<output_dir>/cracktree260/`` or ``<output_dir>/crkwh100/``.

    CrackTree260: 260 road pavement images (expansion of CrackTree200).
    CRKWH100: 100 pavement images, white highway markings.
    Both have binary PNG ground-truth masks.

    Manual download:
      1. CrackTree260 + GT: {CRACKTREE260_ONEDRIVE}
         Extract to: ``<output_dir>/cracktree260/``
      2. CRKWH100: {CRKWH100_ONEDRIVE}
         Extract to: ``<output_dir>/crkwh100/``
      3. Re-run ``uv run tarmac download cracktree260``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    ct260_dir = output_dir / "cracktree260"
    crkwh_dir = output_dir / "crkwh100"
    ct260_pairs = _find_pairs_in_dir(ct260_dir, "cracktree260")
    crkwh_pairs = _find_pairs_in_dir(crkwh_dir, "crkwh100")

    all_triplets = ct260_pairs + crkwh_pairs
    downloaded = bool(all_triplets)

    if not downloaded:
        instructions = _manual_instructions()
        (output_dir / "MANUAL_DOWNLOAD.md").write_text(instructions + "\n")
        empty_pairs = output_dir / "pairs.jsonl"
        empty_pairs.write_text("")
        layout_path = output_dir / "LAYOUT.md"
        layout_path.write_text("# CrackTree260 / CRKWH100\n\nData not yet downloaded. See MANUAL_DOWNLOAD.md.\n")
        return CrackTree260Result(
            output_dir=output_dir,
            downloaded=False,
            cracktree260_pairs=0,
            crkwh100_pairs=0,
            pairs_path=empty_pairs,
            layout_path=layout_path,
        )

    pairs_path = output_dir / "pairs.jsonl"
    with pairs_path.open("w") as handle:
        for image_path, mask_path, source_dataset in all_triplets:
            handle.write(
                json.dumps(
                    {
                        "image_path": str(image_path.resolve()),
                        "mask_path": str(mask_path.resolve()),
                        "image_relpath": str(image_path.relative_to(output_dir)),
                        "mask_relpath": str(mask_path.relative_to(output_dir)),
                        "source_dataset": source_dataset,
                    }
                )
                + "\n"
            )

    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(_describe_layout(output_dir, ct260_pairs, crkwh_pairs) + "\n")
    return CrackTree260Result(
        output_dir=output_dir,
        downloaded=True,
        cracktree260_pairs=len(ct260_pairs),
        crkwh100_pairs=len(crkwh_pairs),
        pairs_path=pairs_path,
        layout_path=layout_path,
    )


def find_cracktree260_pairs(raw_dir: Path = Path("data/raw/cracktree260")) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists():
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
            if row.get("source_dataset") == "cracktree260"
        ]
    return [(img, mask) for img, mask, _ in _find_pairs_in_dir(raw_dir / "cracktree260", "cracktree260")]


def find_crkwh100_pairs(raw_dir: Path = Path("data/raw/cracktree260")) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists():
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
            if row.get("source_dataset") == "crkwh100"
        ]
    return [(img, mask) for img, mask, _ in _find_pairs_in_dir(raw_dir / "crkwh100", "crkwh100")]


def _find_pairs_in_dir(dataset_dir: Path, source_dataset: str) -> list[tuple[Path, Path, str]]:
    if not dataset_dir.exists():
        return []
    all_images = [p for p in dataset_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    masks: dict[str, Path] = {}
    images: dict[str, Path] = {}
    for path in sorted(all_images):
        if _looks_like_mask(path):
            masks[_normalize_stem(path.stem)] = path
        else:
            images[path.stem] = path
    pairs = []
    for stem, image_path in sorted(images.items()):
        mask_path = masks.get(stem)
        if mask_path is not None:
            pairs.append((image_path, mask_path, source_dataset))
    return pairs


def _looks_like_mask(path: Path) -> bool:
    text = " ".join(part.lower() for part in path.parts)
    return any(token in text for token in ("mask", "groundtruth", "ground_truth", "gt", "label", "annotation"))


def _normalize_stem(stem: str) -> str:
    result = stem.lower()
    for token in MASK_STEM_TOKENS:
        result = result.replace(token, "")
    return result


def _describe_layout(output_dir: Path, ct260: list, crkwh: list) -> str:
    return "\n".join([
        "# CrackTree260 / CRKWH100 detected layout",
        "",
        f"Paper: {PAPER_URL}",
        f"Root: `{output_dir}`",
        f"CrackTree260 pairs: {len(ct260)}",
        f"CRKWH100 pairs: {len(crkwh)}",
        "",
        "CrackTree260 OneDrive: " + CRACKTREE260_ONEDRIVE,
        "CRKWH100 OneDrive: " + CRKWH100_ONEDRIVE,
        "",
        "Pairing rule: normalized stem (mask suffix tokens removed) matched across image and GT files.",
    ])


def _manual_instructions() -> str:
    return "\n".join([
        "# CrackTree260 / CRKWH100 manual download required",
        "",
        "Both datasets are distributed via OneDrive shared folders and require",
        "manual download (browser authentication required).",
        "",
        "1. CrackTree260 + GT dataset:",
        f"   URL: {CRACKTREE260_ONEDRIVE}",
        "   Extract to: `data/raw/cracktree260/cracktree260/`",
        "   Expected layout: images (JPG) and GT masks (PNG) in the same or parallel folders.",
        "",
        "2. CRKWH100 dataset:",
        f"   URL: {CRKWH100_ONEDRIVE}",
        "   Extract to: `data/raw/cracktree260/crkwh100/`",
        "",
        "3. Re-run: `uv run tarmac download cracktree260`",
        "",
        f"Paper: {PAPER_URL}",
        "GitHub (qinnzou/DeepCrack): https://github.com/qinnzou/DeepCrack",
    ])
