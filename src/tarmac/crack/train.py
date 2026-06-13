from __future__ import annotations

import json
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image, ImageFile
from rich.console import Console
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from tarmac.crack.model import CrackHead
from tarmac.embedding.embedder import DINOV3_MODEL, HFBackboneEmbedder
from tarmac.embedding.tiling import tile_boxes
from tarmac.inference.analyze import load_active_artifacts

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED = 42


@dataclass(frozen=True)
class CrackTrainConfig:
    manifest_path: str
    model_name: str
    backbone_checkpoint: str
    output_checkpoint: str
    output_metadata: str
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    seed: int
    patience: int
    checkpoint_dir: str
    device: str


class CrackImageDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame, processor: Any) -> None:
        self.frame = frame.reset_index(drop=True)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            cropped = crop_manifest_image(image, str(row.get("tile", "full")))
            pixel_values = self.processor(images=cropped, return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixel_values,
            "label": float(row["has_crack"]),
            "split": str(row["split"]),
            "image_path": str(row["image_path"]),
        }


def train_crack_head(
    manifest_path: Path = Path("data/processed/crack_manifest.parquet"),
    output_checkpoint: Path = Path("models/crack_head.pt"),
    output_metadata: Path = Path("models/crack_head.json"),
    epochs: int = 8,
    batch_size: int = 128,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    patience: int = 3,
    resume: bool = False,
    num_workers: int = 0,
) -> dict[str, Any]:
    _seed_everything(SEED)
    console = Console()
    if not torch.backends.mps.is_available():
        raise RuntimeError("train-crack requires Apple MPS. CPU fallback is disabled by design.")
    device = torch.device("mps")

    active = load_active_artifacts()
    manifest = pd.read_parquet(manifest_path)
    embedder = HFBackboneEmbedder(
        model_name=active.model_name or DINOV3_MODEL,
        checkpoint_path=active.checkpoint_path,
        allow_fallback=False,
        attn_implementation="eager",
        device_name="mps",
    )
    if embedder.device.type != "mps":
        raise RuntimeError(f"train-crack requires MPS; selected device was {embedder.device.type}.")

    embeddings, labels, splits = embed_crack_manifest(
        manifest=manifest,
        embedder=embedder,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    input_dim = embeddings.shape[1]
    head = CrackHead(input_dim=input_dim).to(device)
    train_mask = splits == "train"
    val_mask = splits == "val"
    train_x = torch.from_numpy(embeddings[train_mask]).float().to(device)
    train_y = torch.from_numpy(labels[train_mask]).float().to(device)
    val_x = torch.from_numpy(embeddings[val_mask]).float().to(device)
    val_y_np = labels[val_mask].astype("int64")

    pos = float(train_y.sum().item())
    neg = float(len(train_y) - pos)
    pos_weight = torch.tensor([neg / max(pos, 1.0)], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    checkpoint_dir = Path("models/checkpoints/crack")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    config = CrackTrainConfig(
        manifest_path=str(manifest_path),
        model_name=embedder.model_name,
        backbone_checkpoint=str(active.checkpoint_path),
        output_checkpoint=str(output_checkpoint),
        output_metadata=str(output_metadata),
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        seed=SEED,
        patience=patience,
        checkpoint_dir=str(checkpoint_dir),
        device=device.type,
    )

    history: list[dict[str, float | int]] = []
    best_f1 = -1.0
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
        history = state["history"]
        best_f1 = float(state["best_val_f1"])
        best_epoch = int(state["best_epoch"])
        start_epoch = int(state["epoch"]) + 1
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    console.print(
        f"Training crack head on MPS; rows={len(manifest)} train={int(train_mask.sum())} "
        f"val={int(val_mask.sum())} pos_weight={float(pos_weight.item()):.3f}"
    )
    for epoch in range(start_epoch, epochs + 1):
        head.train()
        order = torch.randperm(len(train_x), generator=generator)
        losses: list[float] = []
        for start in tqdm(range(0, len(order), batch_size), desc=f"Crack epoch {epoch}/{epochs}", unit="batch"):
            idx = order[start : start + batch_size].to(device)
            logits = head(train_x[idx])
            loss = criterion(logits, train_y[idx])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))

        val_metrics = _metrics_for_logits(head, val_x, val_y_np, threshold=0.5)
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)),
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_accuracy": val_metrics["accuracy"],
        }
        history.append(row)
        improved = row["val_f1"] > best_f1
        if improved:
            best_f1 = float(row["val_f1"])
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_checkpoint = checkpoint_dir / f"epoch_{epoch}.pt"
        state = {
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "input_dim": input_dim,
            "model_name": embedder.model_name,
            "backbone_checkpoint": str(active.checkpoint_path),
            "config": asdict(config),
            "history": history,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_f1": best_f1,
            "epochs_without_improvement": epochs_without_improvement,
        }
        torch.save(state, epoch_checkpoint)
        if improved:
            shutil.copy2(epoch_checkpoint, output_checkpoint)
            shutil.copy2(epoch_checkpoint, checkpoint_dir / "best.pt")
        output_metadata.write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_f1": best_f1,
                    "checkpoint": str(output_checkpoint),
                    "latest_epoch_checkpoint": str(epoch_checkpoint),
                },
                indent=2,
            )
            + "\n"
        )
        console.print(f"epoch={epoch} loss={row['loss']:.4f} val_f1={row['val_f1']:.4f}")
        if epochs_without_improvement >= patience:
            console.print(f"Early stopping at epoch {epoch}; patience={patience}.")
            break

    return {
        "checkpoint": str(output_checkpoint),
        "metadata": str(output_metadata),
        "best_epoch": best_epoch,
        "best_val_f1": best_f1,
        "history": history,
    }


@torch.inference_mode()
def embed_crack_manifest(
    manifest: pd.DataFrame,
    embedder: HFBackboneEmbedder,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dataset = CrackImageDataset(manifest, embedder.processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    vectors: list[np.ndarray] = []
    labels: list[float] = []
    splits: list[str] = []
    embedder.model.eval()
    for batch in tqdm(loader, desc="Embedding crack images", unit="batch"):
        emb = embedder.embed_pixel_values(batch["pixel_values"]).numpy().astype("float32")
        vectors.append(emb)
        labels.extend(float(x) for x in batch["label"])
        splits.extend(str(x) for x in batch["split"])
    return np.vstack(vectors), np.array(labels, dtype="float32"), np.array(splits)


def crop_manifest_image(image: Image.Image, tile: str) -> Image.Image:
    rgb = image.convert("RGB")
    if not tile.startswith("tile_"):
        return rgb
    try:
        tile_index = int(tile.split("_")[-1])
    except ValueError:
        return rgb
    boxes = tile_boxes(*rgb.size)
    if tile_index < 0 or tile_index >= len(boxes):
        return rgb
    return rgb.crop(boxes[tile_index])


@torch.inference_mode()
def _metrics_for_logits(
    head: CrackHead,
    x: torch.Tensor,
    y_true: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    head.eval()
    probs = torch.sigmoid(head(x)).detach().cpu().numpy()
    pred = (probs >= threshold).astype("int64")
    return {
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, pred)),
    }


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
