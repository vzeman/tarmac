from __future__ import annotations

import json
import math
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageFile
from rich.console import Console
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from tqdm.auto import tqdm

from tarmac.embedding.embedder import DINOV3_MODEL, HFBackboneEmbedder
from tarmac.train.supcon import (
    ProjectionHead,
    _assert_finite_gradients,
    _find_first_nonfinite_layer,
    _freeze_except_last_blocks,
    _latest_epoch_checkpoint,
    _move_optimizer_state,
    _restore_trainable_state,
    _run_mps_sanity_check,
    _save_best_copy,
    _seed_everything,
    _select_device,
    _snapshot_trainable_state,
    _train_transform,
    supervised_contrastive_loss,
)

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED = 42


@dataclass(frozen=True)
class CrackSupConConfig:
    manifest_path: str
    model_name: str
    initial_checkpoint: str | None
    output_checkpoint: str
    output_metadata: str
    epochs: int
    batch_size: int
    effective_batch_size: int
    accumulation_steps: int
    backbone_lr: float
    head_lr: float
    weight_decay: float
    unfrozen_blocks: int
    seed: int
    run_name: str
    checkpoint_dir: str
    patience: int
    device: str
    attn_implementation: str


class CrackSupConDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose, input_size: int = 224) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform
        self._placeholder = torch.zeros(3, input_size, input_size)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        try:
            with Image.open(row["image_path"]) as image:
                pixels = self.transform(image.convert("RGB"))
        except Exception:
            pixels = self._placeholder
        return {
            "pixel_values": pixels,
            "label": int(row["has_crack"]),
        }


class CrackEvalDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame, processor: Any) -> None:
        self.frame = frame.reset_index(drop=True)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt")["pixel_values"][0]
        return {
            "pixel_values": pixel_values,
            "split": str(row["split"]),
            "has_crack": int(row["has_crack"]),
        }


def train_crack_supcon(
    manifest_path: Path = Path("data/processed/crack_manifest.parquet"),
    output_checkpoint: Path = Path("models/crack_finetuned_backbone.pt"),
    output_metadata: Path = Path("models/crack_finetuned_backbone.json"),
    model_name: str = DINOV3_MODEL,
    initial_checkpoint: Path | None = None,
    epochs: int = 10,
    batch_size: int = 32,
    effective_batch_size: int = 128,
    backbone_lr: float = 5e-5,
    head_lr: float = 5e-4,
    weight_decay: float = 1e-4,
    unfrozen_blocks: int = 4,
    num_workers: int = 0,
    device_name: str = "auto",
    run_name: str = "crack_supcon",
    resume: bool = False,
    patience: int = 3,
    attn_implementation: str = "eager",
    max_train_rows: int = 0,
) -> dict[str, Any]:
    """SupCon fine-tune the DINOv3 backbone on binary crack/no-crack labels.

    Uses crack_manifest.parquet as training data. A balanced sampler ensures
    equal crack/no-crack representation within each batch regardless of
    per-dataset class imbalance. Validates with kNN binary crack F1.
    """
    console = Console()
    _seed_everything(SEED)

    manifest = pd.read_parquet(manifest_path)
    train_frame = manifest[manifest["split"] == "train"].copy()
    if train_frame.empty:
        raise RuntimeError("Train split is empty; cannot run crack SupCon fine-tuning.")

    if max_train_rows > 0 and len(train_frame) > max_train_rows:
        # Stratified subsample to keep class balance
        train_frame = (
            train_frame.groupby("has_crack", group_keys=False)
            .apply(lambda g: g.sample(n=min(len(g), max_train_rows // 2), random_state=SEED))
            .reset_index(drop=True)
        )
        console.print(f"Subsampled to {len(train_frame)} train rows (max_train_rows={max_train_rows})")

    embedder = HFBackboneEmbedder(
        model_name=model_name,
        checkpoint_path=initial_checkpoint,
        allow_fallback=False,
        attn_implementation=attn_implementation,
        move_to_device=False,
    )
    model = embedder.model
    device = _select_device(embedder.device, device_name)
    if device.type != "mps":
        raise RuntimeError(
            f"Crack backbone fine-tuning requires MPS; selected device was {device.type}."
        )
    embedder.device = device
    model.to(device)
    hidden_size = int(getattr(model.config, "hidden_size", 768))
    head = ProjectionHead(hidden_size).to(device)
    _freeze_except_last_blocks(model, last_n=unfrozen_blocks)
    model.train()

    transform = _train_transform(embedder.processor, embedder.input_size)
    train_dataset = CrackSupConDataset(train_frame, transform, input_size=embedder.input_size)
    sampler = _balanced_crack_sampler(train_frame)
    accumulation_steps = max(1, math.ceil(effective_batch_size / batch_size))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        drop_last=True,
    )
    total_steps = max(1, math.ceil(len(train_loader) / accumulation_steps) * epochs)

    trainable_backbone = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        [
            {"params": trainable_backbone, "lr": backbone_lr},
            {"params": head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    checkpoint_dir = Path("models/checkpoints") / run_name

    config = CrackSupConConfig(
        manifest_path=str(manifest_path),
        model_name=embedder.model_name,
        initial_checkpoint=str(initial_checkpoint) if initial_checkpoint is not None else None,
        output_checkpoint=str(output_checkpoint),
        output_metadata=str(output_metadata),
        epochs=epochs,
        batch_size=batch_size,
        effective_batch_size=effective_batch_size,
        accumulation_steps=accumulation_steps,
        backbone_lr=backbone_lr,
        head_lr=head_lr,
        weight_decay=weight_decay,
        unfrozen_blocks=unfrozen_blocks,
        seed=SEED,
        run_name=run_name,
        checkpoint_dir=str(checkpoint_dir),
        patience=patience,
        device=device.type,
        attn_implementation=attn_implementation,
    )

    history: list[dict[str, float | int | str]] = []
    best_score = -1.0
    best_epoch = 0
    start_epoch = 1
    epochs_without_improvement = 0
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    if resume:
        latest_checkpoint = _latest_epoch_checkpoint(checkpoint_dir)
        if latest_checkpoint is None:
            raise RuntimeError(f"--resume requested but no epoch checkpoints exist in {checkpoint_dir}.")
        loaded = _load_crack_checkpoint(latest_checkpoint, model, head, optimizer, scheduler, device)
        history = loaded["history"]
        best_score = float(loaded["best_val_crack_f1"])
        best_epoch = int(loaded["best_epoch"])
        start_epoch = int(loaded["epoch"]) + 1
        epochs_without_improvement = int(loaded.get("epochs_without_improvement", 0))
        console.print(f"Resuming {run_name} from {latest_checkpoint} at epoch {start_epoch}.")
    else:
        backbone_snapshot = _snapshot_trainable_state(model)
        head_snapshot = _snapshot_trainable_state(head)
        _run_mps_sanity_check(
            model=model,
            head=head,
            optimizer=optimizer,
            pixel_batches=[batch for _, batch in zip(range(2), train_loader)],
            device=device,
        )
        _restore_trainable_state(model, backbone_snapshot, device)
        _restore_trainable_state(head, head_snapshot, device)
        optimizer = AdamW(
            [
                {"params": trainable_backbone, "lr": backbone_lr},
                {"params": head.parameters(), "lr": head_lr},
            ],
            weight_decay=weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    crack_count = int(train_frame["has_crack"].sum())
    no_crack_count = len(train_frame) - crack_count
    console.print(
        f"Crack SupCon on {device.type}; backbone={embedder.backbone} ({embedder.model_name}); "
        f"train rows={len(train_frame)} crack={crack_count} no_crack={no_crack_count}"
    )
    global_step = max(0, start_epoch - 1) * max(1, math.ceil(len(train_loader) / accumulation_steps))
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        head.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"Crack SupCon epoch {epoch}/{epochs}", unit="batch")
        for batch_index, batch in enumerate(progress, start=1):
            loss = _crack_forward_loss(model, head, batch["pixel_values"], batch["label"], device)
            scaled = loss / accumulation_steps
            scaled.backward()
            running_loss += float(loss.detach().cpu())

            if batch_index % accumulation_steps == 0 or batch_index == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            progress.set_postfix(loss=f"{running_loss / batch_index:.4f}")

        epoch_loss = running_loss / max(1, len(train_loader))
        val_f1 = _val_crack_f1(model, embedder.processor, manifest, device, max(128, batch_size), num_workers)
        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "loss": epoch_loss,
            "val_crack_f1": val_f1,
            "device": device.type,
        }
        history.append(row)
        console.print(f"epoch={epoch} loss={epoch_loss:.4f} val_crack_f1={val_f1:.4f}")

        improved = val_f1 > best_score
        if improved:
            best_score = val_f1
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_checkpoint = checkpoint_dir / f"epoch_{epoch}.pt"
        _save_crack_checkpoint(
            path=epoch_checkpoint,
            model=model,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            embedder=embedder,
            config=config,
            history=history,
            epoch=epoch,
            best_epoch=best_epoch,
            best_score=best_score,
            epochs_without_improvement=epochs_without_improvement,
        )
        if improved:
            shutil.copy2(epoch_checkpoint, output_checkpoint)
            _save_best_copy(checkpoint_dir, epoch_checkpoint)
        output_metadata.write_text(
            json.dumps(
                {
                    "config": asdict(config),
                    "history": history,
                    "best_epoch": best_epoch,
                    "best_val_crack_f1": best_score,
                    "checkpoint": str(output_checkpoint),
                    "latest_epoch_checkpoint": str(epoch_checkpoint),
                    "epochs_without_improvement": epochs_without_improvement,
                },
                indent=2,
            )
            + "\n"
        )
        if epochs_without_improvement >= patience:
            console.print(f"Early stopping at epoch {epoch}; patience={patience}.")
            break

    return {
        "backbone": embedder.backbone,
        "model_name": embedder.model_name,
        "epochs_trained": len(history),
        "best_epoch": best_epoch,
        "best_val_crack_f1": best_score,
        "checkpoint": str(output_checkpoint),
        "metadata": str(output_metadata),
        "history": history,
    }


def _balanced_crack_sampler(frame: pd.DataFrame) -> WeightedRandomSampler:
    labels = frame["has_crack"].to_numpy()
    class_counts = np.bincount(labels, minlength=2).astype(float)
    class_counts = np.where(class_counts == 0, 1.0, class_counts)
    weights = 1.0 / class_counts[labels]
    return WeightedRandomSampler(
        weights=torch.from_numpy(weights).float(),
        num_samples=len(frame),
        replacement=True,
    )


def _crack_forward_loss(
    model: nn.Module,
    head: nn.Module,
    pixel_values: torch.Tensor,
    labels: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    values = pixel_values.to(device)
    label_tensor = labels.to(device)
    try:
        outputs = model(pixel_values=values)
    except RuntimeError as exc:
        raise RuntimeError(f"MPS forward failed: {exc}") from exc
    cls = outputs.last_hidden_state[:, 0, :].float()
    if not torch.isfinite(cls).all():
        layer_name = _find_first_nonfinite_layer(model, values)
        raise RuntimeError(f"Non-finite CLS on {device.type}; first bad layer: {layer_name}.")
    projections = head(cls)
    if not torch.isfinite(projections).all():
        raise RuntimeError(f"Non-finite projection-head output on {device.type}.")
    loss = supervised_contrastive_loss(projections, label_tensor)
    if not torch.isfinite(loss):
        raise RuntimeError(f"NaN/Inf SupCon loss on {device.type}.")
    return loss


@torch.inference_mode()
def _val_crack_f1(
    model: nn.Module,
    processor: Any,
    manifest: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    eval_frame = manifest[manifest["split"].isin(["train", "val"])].copy()
    dataset = CrackEvalDataset(eval_frame, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    vectors: list[np.ndarray] = []
    splits: list[str] = []
    labels: list[int] = []
    for batch in tqdm(loader, desc="Val kNN embeddings", unit="batch", leave=False):
        outputs = model(pixel_values=batch["pixel_values"].to(device))
        embeddings = F.normalize(outputs.last_hidden_state[:, 0, :].float(), p=2, dim=1)
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("Validation embeddings became non-finite on MPS.")
        vectors.append(embeddings.cpu().numpy().astype("float32"))
        splits.extend(str(s) for s in batch["split"])
        labels.extend(int(lbl) for lbl in batch["has_crack"])

    matrix = np.vstack(vectors).astype("float32")
    split_array = np.array(splits)
    label_array = np.array(labels)
    train_mask = split_array == "train"
    val_mask = split_array == "val"
    if not np.any(val_mask):
        return 0.0
    classifier = KNeighborsClassifier(n_neighbors=10, metric="cosine", weights="distance")
    classifier.fit(matrix[train_mask], label_array[train_mask])
    predictions = classifier.predict(matrix[val_mask])
    return float(f1_score(label_array[val_mask], predictions, zero_division=0))


def _save_crack_checkpoint(
    path: Path,
    model: nn.Module,
    head: nn.Module,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    embedder: HFBackboneEmbedder,
    config: CrackSupConConfig,
    history: list[dict[str, float | int | str]],
    epoch: int,
    best_epoch: int,
    best_score: float,
    epochs_without_improvement: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "projection_head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "requested_model": embedder.requested_model,
            "model_name": embedder.model_name,
            "backbone": embedder.backbone,
            "embedding_dim": int(getattr(model.config, "hidden_size", 768)),
            "config": asdict(config),
            "history": history,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_crack_f1": best_score,
            "epochs_without_improvement": epochs_without_improvement,
        },
        path,
    )


def _load_crack_checkpoint(
    path: Path,
    model: nn.Module,
    head: nn.Module,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device,
) -> dict[str, Any]:
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    head.load_state_dict(state["projection_head_state_dict"])
    optimizer.load_state_dict(state["optimizer_state_dict"])
    scheduler.load_state_dict(state["scheduler_state_dict"])
    _move_optimizer_state(optimizer, device)
    return state
