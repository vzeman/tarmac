from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from rich.console import Console
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from tqdm.auto import tqdm

from tarmac.defect import DEFECT_LABELS, SEED
from tarmac.defect.embeddings import build_defect_embeddings, labels_to_multihot, load_defect_embedding_frame
from tarmac.defect.model import DefectHead


@dataclass(frozen=True)
class DefectTrainConfig:
    manifest_path: str
    embeddings_path: str
    embedding_metadata_path: str
    output_checkpoint: str
    output_metadata: str
    epochs: int
    batch_size: int
    embed_batch_size: int
    lr: float
    weight_decay: float
    seed: int
    patience: int
    checkpoint_dir: str
    device: str
    hidden_dim: int
    dropout: float
    none_cap: int


def train_defect_head(
    manifest_path: Path = Path("data/processed/defect_manifest.parquet"),
    embeddings_path: Path = Path("data/processed/defect_embeddings.parquet"),
    embedding_metadata_path: Path = Path("data/processed/defect_embeddings.json"),
    output_checkpoint: Path = Path("models/defect_head.pt"),
    output_metadata: Path = Path("models/defect_head.json"),
    epochs: int = 40,
    batch_size: int = 512,
    embed_batch_size: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 5,
    resume: bool = False,
    num_workers: int = 0,
    none_cap: int = 20_000,
    hidden_dim: int = 512,
    dropout: float = 0.3,
) -> dict[str, Any]:
    _seed_everything(SEED)
    if not torch.backends.mps.is_available():
        raise RuntimeError("train-defect requires Apple MPS. CPU fallback is disabled by design.")
    device = torch.device("mps")
    console = Console()

    if not embeddings_path.exists() or not embedding_metadata_path.exists():
        console.print("Defect embedding cache not found; building it on MPS first.")
        build_defect_embeddings(
            manifest_path=manifest_path,
            output_path=embeddings_path,
            metadata_path=embedding_metadata_path,
            none_cap=none_cap,
            batch_size=embed_batch_size,
            num_workers=num_workers,
            seed=SEED,
        )

    frame = load_defect_embedding_frame(embeddings_path)
    embeddings = np.vstack(frame["embedding"].to_numpy()).astype("float32")
    labels = np.vstack(frame["labels"].map(labels_to_multihot).to_numpy()).astype("float32")
    splits = frame["split"].astype(str).to_numpy()
    input_dim = int(embeddings.shape[1])

    train_mask = splits == "train"
    val_mask = splits == "val"
    if int(train_mask.sum()) == 0 or int(val_mask.sum()) == 0:
        raise RuntimeError("Defect embeddings must include non-empty train and val splits.")

    train_x = torch.from_numpy(embeddings[train_mask]).float().to(device)
    train_y = torch.from_numpy(labels[train_mask]).float().to(device)
    val_x = torch.from_numpy(embeddings[val_mask]).float().to(device)
    val_y = labels[val_mask].astype("int64")

    head = DefectHead(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=len(DEFECT_LABELS), dropout=dropout).to(device)
    positives = train_y.sum(dim=0)
    negatives = train_y.shape[0] - positives
    pos_weight = (negatives / torch.clamp(positives, min=1.0)).float().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    checkpoint_dir = Path("models/checkpoints/defect")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    config = DefectTrainConfig(
        manifest_path=str(manifest_path),
        embeddings_path=str(embeddings_path),
        embedding_metadata_path=str(embedding_metadata_path),
        output_checkpoint=str(output_checkpoint),
        output_metadata=str(output_metadata),
        epochs=epochs,
        batch_size=batch_size,
        embed_batch_size=embed_batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=SEED,
        patience=patience,
        checkpoint_dir=str(checkpoint_dir),
        device=device.type,
        hidden_dim=hidden_dim,
        dropout=dropout,
        none_cap=none_cap,
    )

    history: list[dict[str, Any]] = []
    best_val_macro_ap = -1.0
    best_epoch = 0
    start_epoch = 1
    epochs_without_improvement = 0
    if resume:
        latest = _latest_epoch_checkpoint(checkpoint_dir)
        if latest is None:
            raise RuntimeError(f"--resume was requested but no checkpoints exist in {checkpoint_dir}.")
        state = torch.load(latest, map_location=device)
        head.load_state_dict(state["head_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = list(state["history"])
        best_val_macro_ap = float(state["best_val_macro_ap"])
        best_epoch = int(state["best_epoch"])
        start_epoch = int(state["epoch"]) + 1
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    console.print(
        "Training defect head on MPS; "
        f"rows={len(frame)} train={int(train_mask.sum())} val={int(val_mask.sum())} "
        f"pos_weight={dict(zip(DEFECT_LABELS, [round(float(x), 3) for x in pos_weight.detach().cpu()]))}"
    )
    for epoch in range(start_epoch, epochs + 1):
        head.train()
        order = torch.randperm(len(train_x), generator=generator)
        losses: list[float] = []
        for start in tqdm(range(0, len(order), batch_size), desc=f"Defect epoch {epoch}/{epochs}", unit="batch"):
            idx = order[start : start + batch_size].to(device)
            logits = head(train_x[idx])
            loss = criterion(logits, train_y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_probs = _predict_probs(head, val_x)
        val_metrics = metric_summary(val_y, val_probs, thresholds=np.full(len(DEFECT_LABELS), 0.5, dtype="float32"))
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_macro_ap": val_metrics["macro_ap"],
            "val_macro_f1_at_0_5": val_metrics["macro_f1"],
            "val_micro_f1_at_0_5": val_metrics["micro_f1"],
            "val_per_label_ap": val_metrics["per_label_ap"],
        }
        history.append(row)

        improved = row["val_macro_ap"] > best_val_macro_ap
        if improved:
            best_val_macro_ap = float(row["val_macro_ap"])
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_checkpoint = checkpoint_dir / f"epoch_{epoch}.pt"
        state = {
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_dim": input_dim,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "label_vocab": DEFECT_LABELS,
            "config": asdict(config),
            "history": history,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_macro_ap": best_val_macro_ap,
            "epochs_without_improvement": epochs_without_improvement,
            "pos_weight": [float(x) for x in pos_weight.detach().cpu().numpy()],
        }
        torch.save(state, epoch_checkpoint)
        if improved:
            shutil.copy2(epoch_checkpoint, output_checkpoint)
            shutil.copy2(epoch_checkpoint, checkpoint_dir / "best.pt")
        _write_training_metadata(
            output_metadata=output_metadata,
            config=config,
            history=history,
            checkpoint=output_checkpoint,
            latest_epoch_checkpoint=epoch_checkpoint,
            best_epoch=best_epoch,
            best_val_macro_ap=best_val_macro_ap,
            thresholds=None,
            pos_weight=[float(x) for x in pos_weight.detach().cpu().numpy()],
        )
        console.print(
            f"epoch={epoch} loss={row['loss']:.4f} "
            f"val_macro_ap={row['val_macro_ap']:.4f} val_macro_f1@0.5={row['val_macro_f1_at_0_5']:.4f}"
        )
        if epochs_without_improvement >= patience:
            console.print(f"Early stopping at epoch {epoch}; patience={patience}.")
            break

    if not output_checkpoint.exists():
        raise RuntimeError("Training finished without producing a best defect checkpoint.")
    best_state = torch.load(output_checkpoint, map_location=device)
    head.load_state_dict(best_state["head_state_dict"])
    val_probs = _predict_probs(head, val_x)
    thresholds = choose_per_label_thresholds(val_y, val_probs)
    threshold_metrics = metric_summary(val_y, val_probs, thresholds=thresholds)
    _write_training_metadata(
        output_metadata=output_metadata,
        config=config,
        history=history,
        checkpoint=output_checkpoint,
        latest_epoch_checkpoint=checkpoint_dir / f"epoch_{history[-1]['epoch']}.pt",
        best_epoch=best_epoch,
        best_val_macro_ap=best_val_macro_ap,
        thresholds={label: float(thresholds[i]) for i, label in enumerate(DEFECT_LABELS)},
        pos_weight=[float(x) for x in pos_weight.detach().cpu().numpy()],
        val_threshold_metrics=threshold_metrics,
    )
    return {
        "checkpoint": str(output_checkpoint),
        "metadata": str(output_metadata),
        "best_epoch": best_epoch,
        "best_val_macro_ap": best_val_macro_ap,
        "thresholds": {label: float(thresholds[i]) for i, label in enumerate(DEFECT_LABELS)},
        "history": history,
    }


def choose_per_label_thresholds(y_true: np.ndarray, probs: np.ndarray) -> np.ndarray:
    thresholds = np.full(probs.shape[1], 0.5, dtype="float32")
    for index in range(probs.shape[1]):
        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in np.linspace(0.05, 0.95, 181):
            pred = (probs[:, index] >= threshold).astype("int64")
            score = f1_score(y_true[:, index], pred, zero_division=0)
            if score > best_f1:
                best_f1 = float(score)
                best_threshold = float(threshold)
        thresholds[index] = best_threshold
    return thresholds


def metric_summary(y_true: np.ndarray, probs: np.ndarray, thresholds: np.ndarray) -> dict[str, Any]:
    pred = (probs >= thresholds.reshape(1, -1)).astype("int64")
    per_label_ap: dict[str, float] = {}
    per_label_f1: dict[str, float] = {}
    for index, label in enumerate(DEFECT_LABELS):
        per_label_ap[label] = _average_precision(y_true[:, index], probs[:, index])
        per_label_f1[label] = float(f1_score(y_true[:, index], pred[:, index], zero_division=0))
    return {
        "macro_ap": float(np.nanmean(list(per_label_ap.values()))),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "macro_precision": float(precision_score(y_true, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, pred, average="macro", zero_division=0)),
        "per_label_ap": per_label_ap,
        "per_label_f1": per_label_f1,
    }


@torch.inference_mode()
def _predict_probs(head: DefectHead, x: torch.Tensor) -> np.ndarray:
    head.eval()
    return torch.sigmoid(head(x)).detach().cpu().numpy().astype("float32")


def _average_precision(y_true: np.ndarray, probs: np.ndarray) -> float:
    if len(np.unique(y_true.astype("int64"))) < 2:
        return float("nan")
    return float(average_precision_score(y_true.astype("int64"), probs))


def _write_training_metadata(
    output_metadata: Path,
    config: DefectTrainConfig,
    history: list[dict[str, Any]],
    checkpoint: Path,
    latest_epoch_checkpoint: Path,
    best_epoch: int,
    best_val_macro_ap: float,
    thresholds: dict[str, float] | None,
    pos_weight: list[float],
    val_threshold_metrics: dict[str, Any] | None = None,
) -> None:
    payload: dict[str, Any] = {
        "config": asdict(config),
        "history": history,
        "best_epoch": best_epoch,
        "best_val_macro_ap": best_val_macro_ap,
        "checkpoint": str(checkpoint),
        "latest_epoch_checkpoint": str(latest_epoch_checkpoint),
        "label_vocab": DEFECT_LABELS,
        "pos_weight": dict(zip(DEFECT_LABELS, pos_weight)),
        "thresholds": thresholds,
    }
    if val_threshold_metrics is not None:
        payload["val_threshold_metrics"] = val_threshold_metrics
    output_metadata.write_text(json.dumps(payload, indent=2) + "\n")


def _latest_epoch_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else -1,
    )
    return checkpoints[-1] if checkpoints else None


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

