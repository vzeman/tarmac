from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

SEED = 42
K_CANDIDATES = [5, 8, 10, 12, 15, 18, 22, 26, 30]


def load_full_embeddings(path: Path) -> tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_parquet(path)
    df = df[df["kind"] == "full"].reset_index(drop=True)
    embeddings = np.vstack(df["embedding"].to_numpy()).astype("float32")
    return df, embeddings


def run_clustering(
    embeddings_path: Path,
    centroids_path: Path,
    assignments_path: Path,
    profile_path: Path,
    metadata_path: Path,
) -> dict[str, object]:
    df, embeddings = load_full_embeddings(embeddings_path)
    train_mask = df["split"].to_numpy() == "train"
    train_embeddings = embeddings[train_mask]

    scores: dict[int, float] = {}
    best_k = K_CANDIDATES[0]
    best_score = -1.0
    for k in K_CANDIDATES:
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(train_embeddings)
        score = float(silhouette_score(train_embeddings, labels, metric="cosine"))
        scores[k] = score
        if score > best_score:
            best_k = k
            best_score = score

    kmeans = KMeans(n_clusters=best_k, random_state=SEED, n_init=10)
    kmeans.fit(train_embeddings)
    kmeans_labels = kmeans.predict(embeddings)

    hdbscan_labels = hdbscan.HDBSCAN(min_cluster_size=50, metric="euclidean").fit_predict(embeddings)

    assignments = df.drop(columns=["embedding"]).copy()
    assignments["kmeans_cluster"] = kmeans_labels.astype("int16")
    assignments["hdbscan_cluster"] = hdbscan_labels.astype("int16")

    centroids_path.parent.mkdir(parents=True, exist_ok=True)
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    profile_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)

    np.save(centroids_path, kmeans.cluster_centers_.astype("float32"))
    assignments.to_parquet(assignments_path, index=False)
    profile = _cluster_profile(assignments[assignments["split"] == "train"])
    profile.to_csv(profile_path, index=False)

    metadata = {
        "chosen_k": int(best_k),
        "k_silhouette_scores": {str(k): v for k, v in scores.items()},
        "best_train_silhouette": float(best_score),
        "hdbscan_clusters": int(len(set(hdbscan_labels)) - (1 if -1 in hdbscan_labels else 0)),
        "hdbscan_noise_share": float(np.mean(hdbscan_labels == -1)),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def _cluster_profile(assignments: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cluster_id, group in assignments.groupby("kmeans_cluster", sort=True):
        surface_counts = group["surface_type"].value_counts(normalize=True).sort_index()
        quality_counts = group["quality"].value_counts(normalize=True).sort_index()
        rows.append(
            {
                "cluster": int(cluster_id),
                "size": int(len(group)),
                "surface_type_distribution": json.dumps(surface_counts.round(4).to_dict()),
                "quality_distribution": json.dumps({int(k): round(float(v), 4) for k, v in quality_counts.items()}),
                "top_surface_type": str(group["surface_type"].mode().iloc[0]),
                "top_quality": int(group["quality"].mode().iloc[0]),
                "examples": json.dumps(group["image_path"].head(5).tolist()),
            }
        )
    return pd.DataFrame(rows)


def cluster_purity(assignments: pd.DataFrame, label_column: str) -> float:
    total = 0
    correct = 0
    for _, group in assignments.groupby("kmeans_cluster"):
        counts = Counter(group[label_column])
        total += len(group)
        correct += counts.most_common(1)[0][1]
    return float(correct / total)
