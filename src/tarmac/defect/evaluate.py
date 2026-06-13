from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from tarmac.defect import DEFECT_LABELS
from tarmac.defect.embeddings import labels_to_multihot, load_defect_embedding_frame
from tarmac.defect.model import DefectHead


def evaluate_defect_head(
    embeddings_path: Path = Path("data/processed/defect_embeddings.parquet"),
    checkpoint_path: Path = Path("models/defect_head.pt"),
    metadata_path: Path = Path("models/defect_head.json"),
    metrics_path: Path = Path("reports/defect_metrics.json"),
    report_path: Path = Path("reports/DEFECT_DETECTION.md"),
) -> dict[str, Any]:
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Missing defect embeddings: {embeddings_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing defect head checkpoint: {checkpoint_path}")

    frame = load_defect_embedding_frame(embeddings_path)
    embeddings = np.vstack(frame["embedding"].to_numpy()).astype("float32")
    y_true = np.vstack(frame["labels"].map(labels_to_multihot).to_numpy()).astype("int64")
    head, thresholds, checkpoint_meta = load_defect_head(checkpoint_path=checkpoint_path, metadata_path=metadata_path)
    with torch.inference_mode():
        probs = torch.sigmoid(head(torch.from_numpy(embeddings).float())).numpy().astype("float32")

    result: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "metadata": str(metadata_path),
        "embeddings": str(embeddings_path),
        "label_vocab": DEFECT_LABELS,
        "thresholds": {label: float(thresholds[index]) for index, label in enumerate(DEFECT_LABELS)},
        "best_epoch": checkpoint_meta.get("best_epoch"),
        "best_val_macro_ap": checkpoint_meta.get("best_val_macro_ap"),
    }
    for split in ("val", "test"):
        mask = frame["split"].astype(str).to_numpy() == split
        split_frame = frame[mask].reset_index(drop=True)
        split_y = y_true[mask]
        split_probs = probs[mask]
        result[split] = {
            "per_label": per_label_metrics(split_y, split_probs, thresholds),
            "per_domain": per_domain_metrics(split_frame, split_y, split_probs, thresholds),
            "summary": multilabel_summary(split_y, split_probs, thresholds),
        }

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    report_path.write_text(markdown_report(result) + "\n")
    return result


def load_defect_head(
    checkpoint_path: Path = Path("models/defect_head.pt"),
    metadata_path: Path = Path("models/defect_head.json"),
) -> tuple[DefectHead, np.ndarray, dict[str, Any]]:
    state = torch.load(checkpoint_path, map_location="cpu")
    input_dim = int(state.get("input_dim", 768))
    hidden_dim = int(state.get("hidden_dim", 512))
    dropout = float(state.get("dropout", 0.3))
    label_vocab = list(state.get("label_vocab", DEFECT_LABELS))
    if label_vocab != DEFECT_LABELS:
        raise RuntimeError(f"Unsupported defect label vocab in checkpoint: {label_vocab}")
    head = DefectHead(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=len(DEFECT_LABELS), dropout=dropout)
    head.load_state_dict(state["head_state_dict"])
    head.eval()

    metadata: dict[str, Any] = {}
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    thresholds_payload = metadata.get("thresholds") or state.get("thresholds") or {}
    thresholds = np.array(
        [float(thresholds_payload.get(label, 0.5)) for label in DEFECT_LABELS],
        dtype="float32",
    )
    merged_meta = {**state, **metadata}
    return head, thresholds, merged_meta


def per_label_metrics(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict[str, dict[str, float | int]]:
    pred = (probs >= thresholds.reshape(1, -1)).astype("int64")
    rows: dict[str, dict[str, float | int]] = {}
    for index, label in enumerate(DEFECT_LABELS):
        rows[label] = {
            "precision": float(precision_score(y_true[:, index], pred[:, index], zero_division=0)),
            "recall": float(recall_score(y_true[:, index], pred[:, index], zero_division=0)),
            "f1": float(f1_score(y_true[:, index], pred[:, index], zero_division=0)),
            "ap": _average_precision(y_true[:, index], probs[:, index]),
            "support": int(y_true[:, index].sum()),
            "predicted_positive": int(pred[:, index].sum()),
            "threshold": float(thresholds[index]),
        }
    return rows


def per_domain_metrics(
    frame: pd.DataFrame,
    y_true: np.ndarray,
    probs: np.ndarray,
    thresholds: np.ndarray,
) -> dict[str, dict[str, Any]]:
    domains = sorted(str(domain) for domain in frame["domain"].dropna().astype(str).unique())
    rows: dict[str, dict[str, Any]] = {}
    for domain in ["overall", *domains]:
        if domain == "overall":
            mask = np.ones(len(frame), dtype=bool)
        else:
            mask = frame["domain"].astype(str).to_numpy() == domain
        if int(mask.sum()) == 0:
            continue
        domain_y = y_true[mask]
        domain_probs = probs[mask]
        domain_pred = (domain_probs >= thresholds.reshape(1, -1)).astype("int64")
        active_indexes = [
            index
            for index in range(len(DEFECT_LABELS))
            if int(domain_y[:, index].sum()) > 0 or int(domain_pred[:, index].sum()) > 0
        ] or list(range(len(DEFECT_LABELS)))
        label_ap = [_average_precision(domain_y[:, i], domain_probs[:, i]) for i in range(len(DEFECT_LABELS))]
        rows[domain] = {
            "rows": int(mask.sum()),
            "positive_label_assignments": int(domain_y.sum()),
            "macro_label_count": int(len(active_indexes)),
            "macro_precision": float(
                precision_score(domain_y[:, active_indexes], domain_pred[:, active_indexes], average="macro", zero_division=0)
            ),
            "macro_recall": float(
                recall_score(domain_y[:, active_indexes], domain_pred[:, active_indexes], average="macro", zero_division=0)
            ),
            "macro_f1": float(
                f1_score(domain_y[:, active_indexes], domain_pred[:, active_indexes], average="macro", zero_division=0)
            ),
            "micro_f1": float(f1_score(domain_y, domain_pred, average="micro", zero_division=0)),
            "macro_ap": _nanmean(label_ap),
            "per_label": per_label_metrics(domain_y, domain_probs, thresholds),
        }
    return rows


def multilabel_summary(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict[str, float | int]:
    pred = (probs >= thresholds.reshape(1, -1)).astype("int64")
    true_positive = int(((y_true == 1) & (pred == 1)).sum())
    false_positive = int(((y_true == 0) & (pred == 1)).sum())
    false_negative = int(((y_true == 1) & (pred == 0)).sum())
    exact_match = float(np.mean(np.all(y_true == pred, axis=1))) if len(y_true) else 0.0
    return {
        "rows": int(len(y_true)),
        "exact_match": exact_match,
        "true_label_assignments": int(y_true.sum()),
        "predicted_label_assignments": int(pred.sum()),
        "true_positive_assignments": true_positive,
        "false_positive_assignments": false_positive,
        "false_negative_assignments": false_negative,
        "images_with_any_true_defect": int((y_true.sum(axis=1) > 0).sum()),
        "images_with_any_predicted_defect": int((pred.sum(axis=1) > 0).sum()),
        "mean_true_labels_per_image": float(y_true.sum(axis=1).mean()) if len(y_true) else 0.0,
        "mean_predicted_labels_per_image": float(pred.sum(axis=1).mean()) if len(y_true) else 0.0,
    }


def markdown_report(result: dict[str, Any]) -> str:
    parts = [
        "# Structural Defect Detection",
        "",
        "The defect head is a multi-label classifier trained on frozen active-backbone DINOv3 embeddings. "
        "It predicts crack, spalling, efflorescence, exposed rebar, and corrosion; pure `none` examples are used "
        "only as negatives.",
        "",
        f"Checkpoint: `{result['checkpoint']}`",
        f"Embeddings: `{result['embeddings']}`",
        f"Best validation macro-AP: `{float(result.get('best_val_macro_ap') or 0.0):.4f}`",
        "",
        "Thresholds are chosen per label by maximizing F1 on the validation split.",
        "",
        "| Label | Threshold |",
        "| --- | ---: |",
        *[
            f"| {label} | {float(result['thresholds'][label]):.3f} |"
            for label in DEFECT_LABELS
        ],
        "",
    ]
    for split in ("val", "test"):
        parts.extend(
            [
                f"## {split.title()} Per-label Metrics",
                "",
                "| Label | Precision | Recall | F1 | AP | Support | Predicted + |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
                *[
                    "| {label} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {ap:.4f} | {support} | {predicted_positive} |".format(
                        label=label,
                        **result[split]["per_label"][label],
                    )
                    for label in DEFECT_LABELS
                ],
                "",
                f"## {split.title()} Per-domain Metrics",
                "",
                "| Domain | Rows | Positive Labels | Macro Labels | Macro Precision | Macro Recall | Macro F1 | Micro F1 | Macro AP |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
                *[
                    "| {domain} | {rows} | {positive_label_assignments} | {macro_label_count} | {macro_precision:.4f} | {macro_recall:.4f} | {macro_f1:.4f} | {micro_f1:.4f} | {macro_ap:.4f} |".format(
                        domain=domain,
                        **metrics,
                    )
                    for domain, metrics in result[split]["per_domain"].items()
                ],
                "",
                f"## {split.title()} Multi-label Summary",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
                *[
                    f"| {key} | {value:.4f} |" if isinstance(value, float) else f"| {key} | {value} |"
                    for key, value in result[split]["summary"].items()
                ],
                "",
            ]
        )
    return "\n".join(parts)


def _average_precision(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y_true.astype("int64"))) < 2:
        return float("nan")
    return float(average_precision_score(y_true.astype("int64"), probs))


def _nanmean(values: list[float]) -> float:
    finite = [value for value in values if not np.isnan(value)]
    if not finite:
        return float("nan")
    return float(np.mean(finite))
