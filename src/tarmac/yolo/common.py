from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def require_mps(device: str = "mps") -> str:
    normalized = device.lower()
    if normalized != "mps":
        raise RuntimeError("YOLO mobile training is configured for MPS only. Pass --device mps.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available; refusing to silently fall back to CPU.")
    return "mps"


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def file_size_mb(path: Path) -> float:
    return path.stat().st_size / (1024 * 1024)
