from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import requests


@dataclass(frozen=True)
class CrackMaskDownloadResult:
    dataset: str
    output_dir: Path
    downloaded: bool
    message: str


CRACK500_REPOS = [
    "https://github.com/fyangneil/pavement-crack-detection.git",
    "https://github.com/fyangneil/Crack500.git",
]
DEEPCRACK_REPOS = [
    "https://github.com/yhlleo/DeepCrack.git",
]


def download_crack500(output_dir: Path = Path("data/raw/crack500")) -> CrackMaskDownloadResult:
    return _download_git_dataset("CRACK500", CRACK500_REPOS, output_dir)


def download_deepcrack(output_dir: Path = Path("data/raw/deepcrack")) -> CrackMaskDownloadResult:
    return _download_git_dataset("DeepCrack", DEEPCRACK_REPOS, output_dir)


def _download_git_dataset(
    dataset: str,
    repos: list[str],
    output_dir: Path,
) -> CrackMaskDownloadResult:
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists() and any(output_dir.iterdir()):
        return CrackMaskDownloadResult(dataset, output_dir, True, f"{dataset} already exists.")
    if shutil.which("git") is None:
        raise RuntimeError(f"git is required to download {dataset}. Install git and retry.")

    errors: list[str] = []
    for repo in repos:
        if not _url_resolves(repo):
            errors.append(f"{repo}: did not resolve quickly")
            continue
        try:
            subprocess.run(
                ["git", "clone", "--depth", "1", repo, str(output_dir)],
                check=True,
                timeout=180,
            )
            return CrackMaskDownloadResult(dataset, output_dir, True, f"Downloaded from {repo}.")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
            errors.append(f"{repo}: {exc}")
            if output_dir.exists() and not any(output_dir.iterdir()):
                output_dir.rmdir()
    raise RuntimeError(
        f"Could not download {dataset} from known GitHub URLs. "
        f"The downloader is implemented and can be rerun when a mirror is available. "
        f"Tried: {'; '.join(errors)}"
    )


def _url_resolves(url: str) -> bool:
    try:
        response = requests.head(url, allow_redirects=True, timeout=10)
        return response.status_code < 400
    except requests.RequestException:
        return False
