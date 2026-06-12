from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, silhouette_score
from sklearn.neighbors import KNeighborsClassifier
import umap

from tarmac.cluster.cluster import cluster_purity, load_full_embeddings
from tarmac.report.umap_html import reference_scatter_html

SEED = 42


def run_evaluation(
    embeddings_path: Path,
    assignments_path: Path,
    embed_metadata_path: Path,
    cluster_metadata_path: Path,
    metrics_path: Path,
    report_path: Path,
    umap_html_path: Path,
    umap_png_path: Path,
) -> dict[str, object]:
    df, embeddings = load_full_embeddings(embeddings_path)
    assignments = pd.read_parquet(assignments_path)
    embed_meta = _read_json(embed_metadata_path)
    cluster_meta = _read_json(cluster_metadata_path)

    metrics: dict[str, object] = {
        "requested_model": embed_meta.get("requested_model", "unknown"),
        "backbone": embed_meta.get("backbone", "unknown"),
        "model_name": embed_meta.get("model_name", "unknown"),
        "device": embed_meta.get("device", "unknown"),
        "chosen_k": cluster_meta.get("chosen_k"),
        "knn": {},
        "silhouette": {},
        "cluster_purity": {},
    }

    for target in ["surface_type", "quality"]:
        metrics["knn"][target] = _knn_metrics(df, embeddings, target)

    metrics["silhouette"]["surface_type"] = float(
        silhouette_score(embeddings, df["surface_type"].to_numpy(), metric="cosine")
    )
    combo_labels = (df["surface_type"].astype(str) + "__q" + df["quality"].astype(str)).to_numpy()
    metrics["silhouette"]["surface_type_quality"] = float(
        silhouette_score(embeddings, combo_labels, metric="cosine")
    )

    assignments = assignments.copy()
    assignments["surface_quality"] = assignments["surface_type"].astype(str) + "__q" + assignments["quality"].astype(str)
    metrics["cluster_purity"]["surface_type"] = cluster_purity(assignments, "surface_type")
    metrics["cluster_purity"]["quality"] = cluster_purity(assignments, "quality")
    metrics["cluster_purity"]["surface_type_quality"] = cluster_purity(assignments, "surface_quality")

    projection = umap.UMAP(
        n_components=2,
        metric="cosine",
        random_state=SEED,
        n_neighbors=30,
        min_dist=0.05,
    ).fit_transform(embeddings)
    _write_umap_html(df, projection, umap_html_path)
    _write_umap_png(df, projection, umap_png_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n")
    report_path.write_text(_markdown_report(metrics), encoding="utf-8")
    return metrics


def write_phase3_comparison_report(
    frozen_metrics_path: Path,
    finetuned_metrics_path: Path,
    report_path: Path,
) -> dict[str, object]:
    frozen = _read_json(frozen_metrics_path)
    finetuned = _read_json(finetuned_metrics_path)
    if not frozen or not finetuned:
        raise RuntimeError("Both frozen and fine-tuned metrics are required for the Phase 3 report.")

    frozen_quality_f1 = _metric(frozen, "knn", "quality", "val_test", "macro_f1")
    finetuned_quality_f1 = _metric(finetuned, "knn", "quality", "val_test", "macro_f1")
    frozen_silhouette = _metric(frozen, "silhouette", "surface_type_quality")
    finetuned_silhouette = _metric(finetuned, "silhouette", "surface_type_quality")
    quality_pass = finetuned_quality_f1 > frozen_quality_f1
    silhouette_pass = finetuned_silhouette > frozen_silhouette
    gate_pass = quality_pass and silhouette_pass

    lines = [
        "# Phase 3 Supervised-Contrastive Fine-Tuning",
        "",
        f"Frozen backbone: `{frozen.get('backbone')}` (`{frozen.get('model_name')}`).",
        f"Fine-tuned backbone: `{finetuned.get('backbone')}` (`{finetuned.get('model_name')}`).",
        "",
        "## kNN Metrics",
        "",
        "| Target | Split | Metric | Frozen | Fine-tuned | Delta |",
        "|---|---|---|---:|---:|---:|",
    ]
    for target in ["surface_type", "quality"]:
        for split in ["val", "test", "val_test"]:
            metric_names = ["accuracy", "macro_f1"]
            if target == "quality":
                metric_names.extend(["mae", "off_by_one_accuracy"])
            for metric_name in metric_names:
                frozen_value = _metric(frozen, "knn", target, split, metric_name)
                finetuned_value = _metric(finetuned, "knn", target, split, metric_name)
                lines.append(
                    f"| {target} | {split} | {metric_name} | "
                    f"{frozen_value:.4f} | {finetuned_value:.4f} | {finetuned_value - frozen_value:+.4f} |"
                )

    lines.extend(
        [
            "",
            "## Embedding And Cluster Metrics",
            "",
            "| Metric | Frozen | Fine-tuned | Delta |",
            "|---|---:|---:|---:|",
        ]
    )
    metric_rows = [
        ("Silhouette vs surface_type", ("silhouette", "surface_type")),
        ("Silhouette vs surface_type + quality", ("silhouette", "surface_type_quality")),
        ("k-means cluster purity: surface_type", ("cluster_purity", "surface_type")),
        ("k-means cluster purity: quality", ("cluster_purity", "quality")),
        ("k-means cluster purity: surface_type + quality", ("cluster_purity", "surface_type_quality")),
    ]
    for label, path in metric_rows:
        frozen_value = _metric(frozen, *path)
        finetuned_value = _metric(finetuned, *path)
        lines.append(
            f"| {label} | {frozen_value:.4f} | {finetuned_value:.4f} | {finetuned_value - frozen_value:+.4f} |"
        )

    verdict = "PASSED" if gate_pass else "FAILED"
    lines.extend(
        [
            "",
            "## Acceptance Gate",
            "",
            f"Gate verdict: **{verdict}**.",
            "",
            "| Requirement | Frozen | Fine-tuned | Passed |",
            "|---|---:|---:|---:|",
            f"| val+test quality kNN macro-F1 improves | {frozen_quality_f1:.4f} | {finetuned_quality_f1:.4f} | {_yes_no(quality_pass)} |",
            f"| silhouette(surface_type + quality) improves | {frozen_silhouette:.4f} | {finetuned_silhouette:.4f} | {_yes_no(silhouette_pass)} |",
            "",
            "Artifacts: `reports/phase3_metrics.json`, `reports/umap_scatter_finetuned.html`, `reports/umap_quality_finetuned.png`.",
        ]
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {
        "gate_pass": gate_pass,
        "quality_pass": quality_pass,
        "silhouette_pass": silhouette_pass,
        "frozen_quality_val_test_macro_f1": frozen_quality_f1,
        "finetuned_quality_val_test_macro_f1": finetuned_quality_f1,
        "frozen_surface_type_quality_silhouette": frozen_silhouette,
        "finetuned_surface_type_quality_silhouette": finetuned_silhouette,
    }


def _knn_metrics(df: pd.DataFrame, embeddings: np.ndarray, target: str) -> dict[str, object]:
    train_mask = df["split"].to_numpy() == "train"
    classifier = KNeighborsClassifier(n_neighbors=10, metric="cosine", weights="distance")
    y_train = df.loc[train_mask, target].to_numpy()
    classifier.fit(embeddings[train_mask], y_train)

    results: dict[str, object] = {}
    for split_name in ["val", "test", "val_test"]:
        if split_name == "val_test":
            mask = df["split"].isin(["val", "test"]).to_numpy()
        else:
            mask = df["split"].to_numpy() == split_name
        y_true = df.loc[mask, target].to_numpy()
        y_pred = classifier.predict(embeddings[mask])
        split_metrics: dict[str, float] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }
        if target == "quality":
            true_ord = y_true.astype(int)
            pred_ord = y_pred.astype(int)
            split_metrics["mae"] = float(mean_absolute_error(true_ord, pred_ord))
            split_metrics["off_by_one_accuracy"] = float(np.mean(np.abs(true_ord - pred_ord) <= 1))
        results[split_name] = split_metrics
    return results


def _write_umap_html(df: pd.DataFrame, projection: np.ndarray, path: Path) -> None:
    reference_scatter_html(df, projection, path)


def _write_umap_png(df: pd.DataFrame, projection: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 8), dpi=160)
    scatter = plt.scatter(
        projection[:, 0],
        projection[:, 1],
        c=df["quality"].astype(int),
        cmap="viridis",
        s=6,
        alpha=0.75,
        linewidths=0,
    )
    plt.title("UMAP projection colored by road quality")
    plt.xlabel("UMAP 1")
    plt.ylabel("UMAP 2")
    cbar = plt.colorbar(scatter, ticks=[1, 2, 3, 4, 5])
    cbar.set_label("quality")
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


def _markdown_report(metrics: dict[str, object]) -> str:
    knn = metrics["knn"]
    lines = [
        "# Phase 2 Frozen-Backbone Baseline",
        "",
        f"Backbone used: `{metrics['backbone']}` (`{metrics['model_name']}`) on `{metrics['device']}`.",
        f"Chosen k-means k: `{metrics['chosen_k']}`.",
    ]
    if metrics.get("requested_model") != metrics.get("model_name"):
        lines.append(
            f"Requested backbone `{metrics['requested_model']}` was unavailable, so the run used the fallback backbone."
        )
    lines.extend(
        [
            "",
            "## kNN Metrics",
            "",
            "| Target | Split | Accuracy | Macro-F1 | MAE | Off-by-one accuracy |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for target in ["surface_type", "quality"]:
        for split in ["val", "test", "val_test"]:
            row = knn[target][split]
            lines.append(
                f"| {target} | {split} | {row['accuracy']:.4f} | {row['macro_f1']:.4f} | "
                f"{_fmt_optional(row.get('mae'))} | {_fmt_optional(row.get('off_by_one_accuracy'))} |"
            )
    lines.extend(
        [
            "",
            "## Embedding And Cluster Metrics",
            "",
            "| Metric | Value |",
            "|---|---:|",
            f"| Silhouette vs surface_type | {metrics['silhouette']['surface_type']:.4f} |",
            f"| Silhouette vs surface_type + quality | {metrics['silhouette']['surface_type_quality']:.4f} |",
            f"| k-means cluster purity: surface_type | {metrics['cluster_purity']['surface_type']:.4f} |",
            f"| k-means cluster purity: quality | {metrics['cluster_purity']['quality']:.4f} |",
            f"| k-means cluster purity: surface_type + quality | {metrics['cluster_purity']['surface_type_quality']:.4f} |",
            "",
            "Artifacts: `reports/phase2_metrics.json`, `reports/umap_scatter.html`, `reports/umap_quality.png`.",
        ]
    )
    return "\n".join(lines) + "\n"


def _read_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text()) if path.exists() else {}


def _fmt_optional(value: object) -> str:
    if value is None:
        return "-"
    return f"{float(value):.4f}"


def _metric(metrics: dict[str, object], *path: str) -> float:
    value: object = metrics
    for key in path:
        value = value[key]  # type: ignore[index]
    return float(value)


def _yes_no(value: bool) -> str:
    return "yes" if value else "no"
