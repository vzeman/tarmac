"""RSCD downloader notes.

RSCD is documented at https://thu-rsxd.com/rscd/ and
https://github.com/ztsrxh/RSCD-Road_Surface_Classification_Dataset.

The public project pages describe the dataset and labels, but the full image
archive is not exposed through a stable direct download endpoint in the same way
as Zenodo or Mendeley. If the project page presents a gated form or a manually
issued archive link, download it yourself and place it under:

    data/raw/rscd/

Expected labels are material classes such as asphalt, concrete, mud, and gravel,
with unevenness/friction annotations. Future unification should map material to
the common `surface_type` schema and unevenness to an approximate quality grade.
"""

from __future__ import annotations

from pathlib import Path

import requests

GITHUB_API = "https://api.github.com/repos/ztsrxh/RSCD-Road_Surface_Classification_Dataset/releases/latest"


def download_rscd(output_dir: Path = Path("data/raw/rscd")) -> None:
    """Attempt to find automatable RSCD release assets, otherwise fail helpfully."""
    response = requests.get(GITHUB_API, timeout=60)
    if response.status_code == 404:
        raise RuntimeError(__doc__)
    response.raise_for_status()
    release = response.json()
    assets = release.get("assets", [])
    if not assets:
        raise RuntimeError(__doc__)

    output_dir.mkdir(parents=True, exist_ok=True)
    raise RuntimeError(
        "RSCD release assets were found, but automatic selection is not implemented "
        "for Phase 1. Review the release assets and download the dataset manually "
        f"into {output_dir}."
    )
