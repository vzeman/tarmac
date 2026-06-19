from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from zipfile import ZipFile

import requests
from tqdm import tqdm

DATASET_URL = "https://raw.githubusercontent.com/yhlleo/DeepCrack/master/dataset/DeepCrack.zip"
PAPER_URL = "https://www.sciencedirect.com/science/article/pii/S0925231219300566"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLIT_FOLDER_MAP = {
    "train_img": "train",
    "test_img": "test",
}
LABEL_FOLDER_MAP = {
    "train_lab": "train",
    "test_lab": "test",
}


@dataclass(frozen=True)
class DeepCrackLiuResult:
    output_dir: Path
    downloaded: bool
    image_count: int
    mask_count: int
    pair_count: int
    pairs_path: Path
    layout_path: Path


def download_deepcrack_liu(output_dir: Path = Path("data/raw/deepcrack_liu")) -> DeepCrackLiuResult:
    """Download DeepCrack (Liu et al., Neurocomputing 2019) from GitHub.

    The dataset is hosted as a single ZIP at the yhlleo/DeepCrack repository
    (``dataset/DeepCrack.zip``). It contains 537 images with binary pixel-level
    crack annotations split into ``train_img``/``train_lab`` and
    ``test_img``/``test_lab`` folders.

    License: RESTRICTED to non-commercial research and educational purposes.
    See https://github.com/yhlleo/DeepCrack for the upstream license statement.

    Manual fallback:
      1. Download ``DeepCrack.zip`` from https://github.com/yhlleo/DeepCrack
         (navigate to ``dataset/DeepCrack.zip`` and click Download).
      2. Place it at ``<output_dir>/archives/DeepCrack.zip``.
      3. Re-run ``uv run tarmac download deepcrack-liu``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    archives_dir = output_dir / "archives"
    extracted_dir = output_dir / "_extracted"
    archives_dir.mkdir(parents=True, exist_ok=True)
    extracted_dir.mkdir(parents=True, exist_ok=True)
    archive_path = archives_dir / "DeepCrack.zip"

    downloaded = archive_path.exists()
    if not downloaded:
        try:
            _stream_download(DATASET_URL, archive_path)
            downloaded = True
        except Exception as exc:
            instructions = _manual_instructions(str(exc))
            (output_dir / "MANUAL_DOWNLOAD.md").write_text(instructions + "\n")
            empty_pairs = output_dir / "pairs.jsonl"
            empty_pairs.write_text("")
            layout_path = output_dir / "LAYOUT.md"
            layout_path.write_text(f"# DeepCrack (Liu 2019)\n\nDownload skipped: {exc}\n")
            return DeepCrackLiuResult(
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
                        "source_dataset": "deepcrack_liu",
                        "split": split,
                    }
                )
                + "\n"
            )

    layout_path = output_dir / "LAYOUT.md"
    layout_path.write_text(_describe_layout(extracted_dir, pairs) + "\n")
    return DeepCrackLiuResult(
        output_dir=output_dir,
        downloaded=True,
        image_count=len({p[0] for p in pairs}),
        mask_count=len({p[1] for p in pairs}),
        pair_count=len(pairs),
        pairs_path=pairs_path,
        layout_path=layout_path,
    )


def find_deepcrack_liu_pairs(raw_dir: Path = Path("data/raw/deepcrack_liu")) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists():
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        ]
    extracted_dir = raw_dir / "_extracted"
    return [(img, mask) for img, mask, _ in _find_pairs(extracted_dir)]


def _find_pairs(root: Path) -> list[tuple[Path, Path, str]]:
    images: dict[str, tuple[Path, str]] = {}
    masks: dict[str, tuple[Path, str]] = {}

    for folder_name, split in SPLIT_FOLDER_MAP.items():
        folder = _find_folder(root, folder_name)
        if folder is None:
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                images[path.stem] = (path, split)

    for folder_name, split in LABEL_FOLDER_MAP.items():
        folder = _find_folder(root, folder_name)
        if folder is None:
            continue
        for path in sorted(folder.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                masks[path.stem] = (path, split)

    pairs: list[tuple[Path, Path, str]] = []
    for stem, (image_path, split) in sorted(images.items()):
        if stem in masks:
            pairs.append((image_path, masks[stem][0], split))
    return pairs


def _find_folder(root: Path, name: str) -> Path | None:
    direct = root / name
    if direct.exists():
        return direct
    for candidate in root.rglob(name):
        if candidate.is_dir():
            return candidate
    return None


def _extract_archive(archive_path: Path, output_dir: Path) -> None:
    marker = output_dir / ".deepcrack_liu.extracted"
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
    train_count = sum(1 for _, _, split in pairs if split == "train")
    test_count = sum(1 for _, _, split in pairs if split == "test")
    return "\n".join([
        "# DeepCrack (Liu et al., Neurocomputing 2019) detected layout",
        "",
        f"Source: {DATASET_URL}",
        f"Paper: {PAPER_URL}",
        f"Root: `{root}`",
        f"Total pairs: {len(pairs)} (train={train_count}, test={test_count})",
        "",
        "License: RESTRICTED to non-commercial research and educational purposes.",
        "Pairing rule: matched by stem across train_img/train_lab and test_img/test_lab folders.",
    ])


def _manual_instructions(reason: str) -> str:
    return "\n".join([
        "# DeepCrack (Liu 2019) manual download required",
        "",
        f"Automatic download was skipped because: {reason}",
        "",
        f"Dataset ZIP: {DATASET_URL}",
        "GitHub repo: https://github.com/yhlleo/DeepCrack",
        "",
        "Steps:",
        "  1. Download `DeepCrack.zip` from the GitHub repo (dataset/ folder).",
        "  2. Place it at `data/raw/deepcrack_liu/archives/DeepCrack.zip`.",
        "  3. Re-run `uv run tarmac download deepcrack-liu`.",
        "",
        "License: RESTRICTED to non-commercial research and educational purposes.",
    ])
