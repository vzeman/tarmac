from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm.auto import tqdm

from tarmac.crack.model import CrackHead
from tarmac.crack.train import embed_crack_manifest
from tarmac.embedding.embedder import HFBackboneEmbedder
from tarmac.inference.analyze import load_active_artifacts


def evaluate_crack_head(
    manifest_path: Path = Path("data/processed/crack_manifest.parquet"),
    checkpoint_path: Path = Path("models/crack_head.pt"),
    metrics_path: Path = Path("reports/crack_metrics.json"),
    report_path: Path = Path("reports/CRACK_DETECTION.md"),
    batch_size: int = 128,
    device_name: str = "auto",
) -> dict[str, Any]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing crack head checkpoint: {checkpoint_path}")
    active = load_active_artifacts()
    embedder = HFBackboneEmbedder(
        model_name=active.model_name,
        checkpoint_path=active.checkpoint_path,
        allow_fallback=False,
        attn_implementation="eager",
        device_name=device_name,
    )
    manifest = pd.read_parquet(manifest_path)
    eval_manifest = manifest[manifest["split"].isin(["val", "test"])].reset_index(drop=True)
    embeddings, labels, splits = embed_crack_manifest(eval_manifest, embedder, batch_size=batch_size)
    state = torch.load(checkpoint_path, map_location="cpu")
    input_dim = int(state.get("input_dim", embeddings.shape[1]))
    head = CrackHead(input_dim=input_dim)
    head.load_state_dict(state["head_state_dict"])
    head.eval()
    with torch.inference_mode():
        probs = torch.sigmoid(head(torch.from_numpy(embeddings).float())).numpy()

    val_mask = splits == "val"
    threshold = choose_threshold(labels[val_mask].astype("int64"), probs[val_mask])
    result = {
        "threshold": threshold,
        "val": _split_metrics(labels[val_mask], probs[val_mask], threshold),
        "test": _split_metrics(labels[splits == "test"], probs[splits == "test"], threshold),
        "checkpoint": str(checkpoint_path),
        "manifest": str(manifest_path),
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    report_path.write_text(_markdown_report(result) + "\n")
    return result


def choose_threshold(y_true: np.ndarray, probs: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (probs >= threshold).astype("int64")
        score = f1_score(y_true, pred, zero_division=0)
        if score > best_f1:
            best_f1 = float(score)
            best_threshold = float(threshold)
    return best_threshold


def _split_metrics(labels: np.ndarray, probs: np.ndarray, threshold: float) -> dict[str, float]:
    y_true = labels.astype("int64")
    pred = (probs >= threshold).astype("int64")
    auc = roc_auc_score(y_true, probs) if len(set(y_true.tolist())) > 1 else float("nan")
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "roc_auc": float(auc),
        "count": int(len(y_true)),
        "positive_count": int(y_true.sum()),
    }


def _markdown_report(result: dict[str, Any]) -> str:
    val = result["val"]
    test = result["test"]
    return f"""# Crack Detection

Crack detection is trained and evaluated as a separate binary track from quality grading.

Chosen threshold: `{result["threshold"]:.3f}` (max F1 on validation).

| Split | Precision | Recall | F1 | Accuracy | ROC-AUC | Count | Positives |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| val | {val["precision"]:.4f} | {val["recall"]:.4f} | {val["f1"]:.4f} | {val["accuracy"]:.4f} | {val["roc_auc"]:.4f} | {val["count"]} | {val["positive_count"]} |
| test | {test["precision"]:.4f} | {test["recall"]:.4f} | {test["f1"]:.4f} | {test["accuracy"]:.4f} | {test["roc_auc"]:.4f} | {test["count"]} | {test["positive_count"]} |
"""
