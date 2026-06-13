from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image, ImageFile
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from tarmac.defect import DEFECT_LABELS, NONE_LABEL, SEED
from tarmac.embedding.embedder import DINOV3_MODEL, HFBackboneEmbedder
from tarmac.inference.analyze import load_active_artifacts

ImageFile.LOAD_TRUNCATED_IMAGES = True


@dataclass(frozen=True)
class DefectEmbeddingResult:
    embeddings_path: Path
    metadata_path: Path
    row_count: int
    positive_rows: int
    pure_none_rows: int
    embedding_dim: int
    input_size: int
    model_name: str
    checkpoint: str


class DefectImageDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame, processor: Any, input_size: int) -> None:
        self.frame = frame.reset_index(drop=True)
        self.processor = processor
        self.input_size = input_size

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = str(row["image_path"])
        with Image.open(path) as image:
            rgb = image.convert("RGB").resize((self.input_size, self.input_size))
        pixel_values = self.processor(images=rgb, return_tensors="pt")["pixel_values"][0]
        labels = _label_list(row["labels"])
        return {
            "pixel_values": pixel_values,
            "manifest_index": int(row["manifest_index"]),
            "image_path": path,
            "source_dataset": str(row["source_dataset"]),
            "domain": str(row["domain"]),
            "structure_material": str(row["structure_material"]),
            "labels": labels,
            "has_crack": int(row["has_crack"]),
            "split": str(row["split"]),
            "selected_reason": str(row["selected_reason"]),
        }


def collate_defect_embedding_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "manifest_index",
        "image_path",
        "source_dataset",
        "domain",
        "structure_material",
        "labels",
        "has_crack",
        "split",
        "selected_reason",
    ]
    out: dict[str, Any] = {key: [item[key] for item in batch] for key in keys}
    out["pixel_values"] = torch.stack([item["pixel_values"] for item in batch], dim=0)
    return out


def build_defect_embeddings(
    manifest_path: Path = Path("data/processed/defect_manifest.parquet"),
    output_path: Path = Path("data/processed/defect_embeddings.parquet"),
    metadata_path: Path = Path("data/processed/defect_embeddings.json"),
    none_cap: int = 20_000,
    batch_size: int = 64,
    num_workers: int = 0,
    seed: int = SEED,
    force: bool = False,
) -> DefectEmbeddingResult:
    _seed_everything(seed)
    if not force and output_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        return DefectEmbeddingResult(
            embeddings_path=output_path,
            metadata_path=metadata_path,
            row_count=int(metadata["row_count"]),
            positive_rows=int(metadata["positive_rows"]),
            pure_none_rows=int(metadata["pure_none_rows"]),
            embedding_dim=int(metadata["embedding_dim"]),
            input_size=int(metadata["input_size"]),
            model_name=str(metadata["model_name"]),
            checkpoint=str(metadata["checkpoint"]),
        )
    if not torch.backends.mps.is_available():
        raise RuntimeError("Defect embedding requires Apple MPS. CPU fallback is disabled by design.")

    active = load_active_artifacts()
    manifest = pd.read_parquet(manifest_path).reset_index(drop=True)
    subset = select_defect_embedding_subset(manifest, none_cap=none_cap, seed=seed)

    embedder = HFBackboneEmbedder(
        model_name=active.model_name or DINOV3_MODEL,
        checkpoint_path=active.checkpoint_path,
        allow_fallback=False,
        attn_implementation="eager",
        device_name="mps",
    )
    if embedder.device.type != "mps":
        raise RuntimeError(f"Defect embedding requires MPS; selected device was {embedder.device.type}.")
    info = embedder.info

    dataset = DefectImageDataset(subset, embedder.processor, embedder.input_size)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_defect_embedding_batch,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    writer: pq.ParquetWriter | None = None
    try:
        for batch in tqdm(loader, desc="Embedding defect images", unit="batch"):
            embeddings = embedder.embed_pixel_values(batch["pixel_values"]).numpy().astype("float32")
            table = _batch_to_table(batch, embeddings, info.embedding_dim)
            if writer is None:
                writer = pq.ParquetWriter(output_path, table.schema)
            writer.write_table(table)
    finally:
        if writer is not None:
            writer.close()

    positive_rows = int((subset["selected_reason"] == "defect_positive").sum())
    pure_none_rows = int((subset["selected_reason"] == "pure_none_subset").sum())
    metadata = {
        "manifest_path": str(manifest_path),
        "embeddings_path": str(output_path),
        "model_name": info.model_name,
        "requested_model": info.requested_model,
        "backbone": info.backbone,
        "checkpoint": str(active.checkpoint_path),
        "device": info.device,
        "embedding_dim": info.embedding_dim,
        "input_size": info.input_size,
        "seed": seed,
        "none_cap": int(none_cap),
        "row_count": int(len(subset)),
        "positive_rows": positive_rows,
        "pure_none_rows": pure_none_rows,
        "label_vocab": DEFECT_LABELS,
        "subset_counts": _subset_counts(subset),
        "source_domain_split_counts": _group_counts(subset, ["source_dataset", "domain", "split", "selected_reason"]),
        "label_totals": _label_totals(subset),
    }
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    return DefectEmbeddingResult(
        embeddings_path=output_path,
        metadata_path=metadata_path,
        row_count=int(len(subset)),
        positive_rows=positive_rows,
        pure_none_rows=pure_none_rows,
        embedding_dim=info.embedding_dim,
        input_size=info.input_size,
        model_name=info.model_name,
        checkpoint=str(active.checkpoint_path),
    )


def select_defect_embedding_subset(manifest: pd.DataFrame, none_cap: int, seed: int = SEED) -> pd.DataFrame:
    frame = manifest.copy().reset_index(drop=True)
    frame["manifest_index"] = np.arange(len(frame), dtype=np.int64)
    label_lists = frame["labels"].map(_label_list)
    positive_mask = label_lists.map(lambda labels: any(label in labels for label in DEFECT_LABELS))
    pure_none_mask = label_lists.map(lambda labels: set(labels) == {NONE_LABEL})

    positives = frame[positive_mask].copy()
    positives["selected_reason"] = "defect_positive"

    none_candidates = frame[pure_none_mask].copy()
    none_target = min(int(none_cap), len(none_candidates))
    sampled_none = _sample_stratified_none(none_candidates, none_target=none_target, seed=seed)
    sampled_none["selected_reason"] = "pure_none_subset"

    subset = (
        pd.concat([positives, sampled_none], ignore_index=True)
        .sort_values("manifest_index")
        .reset_index(drop=True)
    )
    subset["labels"] = subset["labels"].map(_label_list)
    return subset


def load_defect_embedding_frame(path: Path = Path("data/processed/defect_embeddings.parquet")) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame["labels"] = frame["labels"].map(_label_list)
    return frame


def labels_to_multihot(labels: object, label_vocab: list[str] | None = None) -> np.ndarray:
    vocab = label_vocab or DEFECT_LABELS
    label_set = set(_label_list(labels))
    return np.array([1.0 if label in label_set else 0.0 for label in vocab], dtype="float32")


def _sample_stratified_none(none_candidates: pd.DataFrame, none_target: int, seed: int) -> pd.DataFrame:
    if none_target <= 0 or none_candidates.empty:
        return none_candidates.head(0).copy()
    if none_target >= len(none_candidates):
        return none_candidates.copy()
    group_cols = ["source_dataset", "domain", "split"]
    group_sizes = none_candidates.groupby(group_cols, observed=True).size().reset_index(name="count")
    total = int(group_sizes["count"].sum())
    quotas = group_sizes["count"].to_numpy(dtype="float64") / float(total) * float(none_target)
    floors = np.floor(quotas).astype(int)
    remainder = none_target - int(floors.sum())
    fractions = quotas - floors
    for index in np.argsort(-fractions)[:remainder]:
        floors[index] += 1
    group_sizes["quota"] = np.minimum(floors, group_sizes["count"].to_numpy(dtype=int))

    sampled: list[pd.DataFrame] = []
    for row in group_sizes.itertuples(index=False):
        quota = int(row.quota)
        if quota <= 0:
            continue
        mask = np.ones(len(none_candidates), dtype=bool)
        for col in group_cols:
            mask &= none_candidates[col].to_numpy() == getattr(row, col)
        group = none_candidates[mask]
        sampled.append(group.sample(n=quota, random_state=seed + len(sampled)))
    result = pd.concat(sampled, ignore_index=True)
    if len(result) < none_target:
        missing = none_target - len(result)
        extras = none_candidates.drop(index=result["manifest_index"].to_numpy(), errors="ignore")
        if len(extras) > 0:
            result = pd.concat([result, extras.sample(n=min(missing, len(extras)), random_state=seed)], ignore_index=True)
    return result.head(none_target).copy()


def _batch_to_table(batch: dict[str, Any], embeddings: np.ndarray, dim: int) -> pa.Table:
    arrays = {
        "manifest_index": pa.array(batch["manifest_index"], type=pa.int64()),
        "image_path": pa.array(batch["image_path"], type=pa.string()),
        "source_dataset": pa.array(batch["source_dataset"], type=pa.string()),
        "domain": pa.array(batch["domain"], type=pa.string()),
        "structure_material": pa.array(batch["structure_material"], type=pa.string()),
        "labels": pa.array(batch["labels"], type=pa.list_(pa.string())),
        "has_crack": pa.array(batch["has_crack"], type=pa.int8()),
        "split": pa.array(batch["split"], type=pa.string()),
        "selected_reason": pa.array(batch["selected_reason"], type=pa.string()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(embeddings.reshape(-1), type=pa.float32()),
            dim,
        ),
    }
    return pa.table(arrays)


def _label_list(labels: object) -> list[str]:
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    if isinstance(labels, tuple):
        labels = list(labels)
    if isinstance(labels, str):
        labels = [labels]
    return [str(label) for label in labels]


def _label_totals(frame: pd.DataFrame) -> dict[str, int]:
    return {
        label: int(frame["labels"].map(lambda labels: label in _label_list(labels)).sum())
        for label in [*DEFECT_LABELS, NONE_LABEL]
    }


def _subset_counts(frame: pd.DataFrame) -> dict[str, int]:
    return {str(k): int(v) for k, v in frame["selected_reason"].value_counts().sort_index().items()}


def _group_counts(frame: pd.DataFrame, columns: list[str]) -> list[dict[str, Any]]:
    grouped = frame.groupby(columns, observed=True).size().reset_index(name="rows")
    return grouped.sort_values(columns).to_dict("records")


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

