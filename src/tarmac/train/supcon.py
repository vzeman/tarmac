from __future__ import annotations

import json
import math
import random
import re
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm.auto import tqdm

from tarmac.embedding.embedder import DINOV3_MODEL, HFBackboneEmbedder

ImageFile.LOAD_TRUNCATED_IMAGES = True
SEED = 42
TEMPERATURE = 0.07


@dataclass(frozen=True)
class SupConConfig:
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
    temperature: float
    unfrozen_blocks: int
    seed: int
    run_name: str
    checkpoint_dir: str
    patience: int
    device: str
    attn_implementation: str


class SupConImageDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame, transform: transforms.Compose) -> None:
        self.frame = frame.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            pixels = self.transform(image.convert("RGB"))
        return {
            "pixel_values": pixels,
            "label": int(row["supcon_label"]),
        }


class EvalImageDataset(Dataset[dict[str, Any]]):
    def __init__(self, frame: pd.DataFrame, processor: Any) -> None:
        self.frame = frame.reset_index(drop=True)
        self.processor = processor

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        with Image.open(row["image_path"]) as image:
            pixel_values = self.processor(images=image.convert("RGB"), return_tensors="pt")[
                "pixel_values"
            ][0]
        return {
            "pixel_values": pixel_values,
            "split": row["split"],
            "quality": int(row["quality"]),
        }


class ProjectionHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, 128),
        )

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(embeddings), p=2, dim=1)


def train_supcon(
    manifest_path: Path = Path("data/processed/manifest.parquet"),
    output_checkpoint: Path = Path("models/finetuned_backbone.pt"),
    output_metadata: Path = Path("models/finetuned_backbone.json"),
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
    run_name: str = "supcon",
    resume: bool = False,
    patience: int = 3,
    attn_implementation: str = "eager",
) -> dict[str, Any]:
    console = Console()
    _seed_everything(SEED)

    manifest = pd.read_parquet(manifest_path)
    train_frame = manifest[manifest["split"] == "train"].copy()
    if train_frame.empty:
        raise RuntimeError("Train split is empty; cannot run supervised contrastive fine-tuning.")
    train_frame["composite_label"] = (
        train_frame["surface_type"].astype(str) + "__q" + train_frame["quality"].astype(str)
    )
    train_frame["supcon_label"] = pd.factorize(train_frame["composite_label"])[0].astype("int64")

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
            f"Training requires MPS and will not fall back to CPU; selected device was {device.type}."
        )
    embedder.device = device
    model.to(device)
    hidden_size = int(getattr(model.config, "hidden_size", 768))
    head = ProjectionHead(hidden_size).to(device)
    _freeze_except_last_blocks(model, last_n=unfrozen_blocks)
    model.train()

    transform = _train_transform(embedder.processor, embedder.input_size)
    train_dataset = SupConImageDataset(train_frame, transform)
    generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        generator=generator,
    )

    trainable_backbone = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(
        [
            {"params": trainable_backbone, "lr": backbone_lr},
            {"params": head.parameters(), "lr": head_lr},
        ],
        weight_decay=weight_decay,
    )
    accumulation_steps = max(1, math.ceil(effective_batch_size / batch_size))
    total_steps = max(1, math.ceil(len(train_loader) / accumulation_steps) * epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)
    checkpoint_dir = Path("models/checkpoints") / run_name

    config = SupConConfig(
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
        temperature=TEMPERATURE,
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
            raise RuntimeError(f"--resume was requested but no epoch checkpoints exist in {checkpoint_dir}.")
        loaded = _load_training_checkpoint(
            latest_checkpoint,
            model=model,
            head=head,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
        )
        history = loaded["history"]
        best_score = float(loaded["best_val_quality_macro_f1"])
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

    console.print(
        f"Training SupCon on {device.type}; backbone={embedder.backbone} "
        f"({embedder.model_name}); train rows={len(train_frame)}"
    )
    global_step = max(0, start_epoch - 1) * max(1, math.ceil(len(train_loader) / accumulation_steps))
    for epoch in range(start_epoch, epochs + 1):
        model.train()
        head.train()
        running_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        progress = tqdm(train_loader, desc=f"SupCon epoch {epoch}/{epochs}", unit="batch")
        for batch_index, batch in enumerate(progress, start=1):
            loss = _forward_loss(
                model=model,
                head=head,
                pixel_values=batch["pixel_values"],
                labels=batch["label"],
                device=device,
            )
            scaled_loss = loss / accumulation_steps
            scaled_loss.backward()
            running_loss += float(loss.detach().cpu())

            if batch_index % accumulation_steps == 0 or batch_index == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
            progress.set_postfix(loss=f"{running_loss / batch_index:.4f}")

        epoch_loss = running_loss / max(1, len(train_loader))
        val_macro_f1 = _val_quality_macro_f1(
            model=model,
            processor=embedder.processor,
            manifest=manifest,
            device=device,
            batch_size=max(128, batch_size),
            num_workers=num_workers,
        )
        row: dict[str, float | int | str] = {
            "epoch": epoch,
            "loss": epoch_loss,
            "val_quality_macro_f1": val_macro_f1,
            "device": device.type,
        }
        history.append(row)
        console.print(
            f"epoch={epoch} loss={epoch_loss:.4f} val_quality_macro_f1={val_macro_f1:.4f}"
        )

        improved = val_macro_f1 > best_score
        if improved:
            best_score = val_macro_f1
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        epoch_checkpoint = checkpoint_dir / f"epoch_{epoch}.pt"
        _save_checkpoint(
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
                    "best_val_quality_macro_f1": best_score,
                    "checkpoint": str(output_checkpoint),
                    "latest_epoch_checkpoint": str(epoch_checkpoint),
                    "epochs_without_improvement": epochs_without_improvement,
                    "mps_nan_root_cause": (
                        "Previous trainer caught MPS RuntimeError/NaN and silently moved training "
                        "to CPU. This trainer uses float32 eager attention on MPS, no AMP/autocast, "
                        "a two-batch finite sanity step, and aborts with NaN diagnostics instead."
                    ),
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
        "best_val_quality_macro_f1": best_score,
        "checkpoint": str(output_checkpoint),
        "metadata": str(output_metadata),
        "history": history,
    }


def supervised_contrastive_loss(
    projections: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = TEMPERATURE,
) -> torch.Tensor:
    labels = labels.view(-1, 1)
    positive_mask = torch.eq(labels, labels.T).float().to(projections.device)
    logits = torch.div(torch.matmul(projections, projections.T), temperature)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    logits_mask = torch.ones_like(positive_mask) - torch.eye(
        positive_mask.shape[0], device=projections.device
    )
    positive_mask = positive_mask * logits_mask
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    positive_counts = positive_mask.sum(dim=1)
    valid = positive_counts > 0
    if not torch.any(valid):
        return projections.sum() * 0.0
    mean_log_prob_pos = (positive_mask * log_prob).sum(dim=1)[valid] / positive_counts[valid]
    return -mean_log_prob_pos.mean()


def _run_mps_sanity_check(
    model: nn.Module,
    head: nn.Module,
    optimizer: AdamW,
    pixel_batches: list[dict[str, Any]],
    device: torch.device,
) -> None:
    if device.type != "mps":
        raise RuntimeError("MPS sanity check requires an MPS device.")
    if len(pixel_batches) < 2:
        raise RuntimeError("Need at least two batches for the MPS sanity train step.")
    model.train()
    head.train()
    optimizer.zero_grad(set_to_none=True)
    for batch_index, batch in enumerate(pixel_batches, start=1):
        loss = _forward_loss(
            model=model,
            head=head,
            pixel_values=batch["pixel_values"],
            labels=batch["label"],
            device=device,
        )
        loss.backward()
        _assert_finite_gradients(model, "backbone")
        _assert_finite_gradients(head, "projection_head")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if not math.isfinite(float(loss.detach().cpu())):
            raise RuntimeError(f"MPS sanity batch {batch_index} produced non-finite loss.")


def _forward_loss(
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
        raise RuntimeError(f"MPS forward failed without CPU fallback: {exc}") from exc
    cls = outputs.last_hidden_state[:, 0, :].float()
    if not torch.isfinite(cls).all():
        layer_name = _find_first_nonfinite_layer(model, values)
        raise RuntimeError(f"Non-finite backbone CLS embedding on {device.type}; first bad layer: {layer_name}.")
    projections = head(cls)
    if not torch.isfinite(projections).all():
        raise RuntimeError(f"Non-finite projection-head output on {device.type}.")
    loss = supervised_contrastive_loss(projections, label_tensor)
    if not torch.isfinite(loss):
        layer_name = _find_first_nonfinite_layer(model, values)
        raise RuntimeError(f"NaN/Inf SupCon loss on {device.type}; first bad layer: {layer_name}.")
    return loss


@torch.inference_mode()
def _val_quality_macro_f1(
    model: nn.Module,
    processor: Any,
    manifest: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    num_workers: int,
) -> float:
    eval_frame = manifest[manifest["split"].isin(["train", "val"])].copy()
    dataset = EvalImageDataset(eval_frame, processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model.eval()
    vectors: list[np.ndarray] = []
    splits: list[str] = []
    qualities: list[int] = []
    for batch in tqdm(loader, desc="Val kNN embeddings", unit="batch", leave=False):
        outputs = model(pixel_values=batch["pixel_values"].to(device))
        embeddings = F.normalize(outputs.last_hidden_state[:, 0, :].float(), p=2, dim=1)
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("Validation embeddings became non-finite on MPS.")
        vectors.append(embeddings.cpu().numpy().astype("float32"))
        splits.extend(batch["split"])
        qualities.extend([int(q) for q in batch["quality"]])

    matrix = np.vstack(vectors).astype("float32")
    split_array = np.array(splits)
    quality_array = np.array(qualities)
    train_mask = split_array == "train"
    val_mask = split_array == "val"
    classifier = KNeighborsClassifier(n_neighbors=10, metric="cosine", weights="distance")
    classifier.fit(matrix[train_mask], quality_array[train_mask])
    predictions = classifier.predict(matrix[val_mask])
    return float(f1_score(quality_array[val_mask], predictions, average="macro"))


def _save_checkpoint(
    path: Path,
    model: nn.Module,
    head: nn.Module,
    optimizer: AdamW,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    embedder: HFBackboneEmbedder,
    config: SupConConfig,
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
            "best_val_quality_macro_f1": best_score,
            "epochs_without_improvement": epochs_without_improvement,
        },
        path,
    )


def _load_training_checkpoint(
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


def _latest_epoch_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = sorted(
        checkpoint_dir.glob("epoch_*.pt"),
        key=lambda path: int(path.stem.split("_")[-1]) if path.stem.split("_")[-1].isdigit() else -1,
    )
    return checkpoints[-1] if checkpoints else None


def _save_best_copy(checkpoint_dir: Path, epoch_checkpoint: Path) -> None:
    shutil.copy2(epoch_checkpoint, checkpoint_dir / "best.pt")


def _freeze_except_last_blocks(model: nn.Module, last_n: int) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    layer_indexes = sorted(
        {
            int(match.group(1))
            for name, _ in model.named_parameters()
            if (match := _layer_match(name))
        }
    )
    if not layer_indexes:
        raise RuntimeError("Could not identify transformer encoder layers to unfreeze.")
    trainable_indexes = set(layer_indexes[-last_n:])
    for name, parameter in model.named_parameters():
        match = _layer_match(name)
        if match and int(match.group(1)) in trainable_indexes:
            parameter.requires_grad = True


def _layer_match(name: str) -> re.Match[str] | None:
    return re.search(r"(?:encoder|model)\.layer\.(\d+)", name)


def _train_transform(processor: Any, input_size: int) -> transforms.Compose:
    mean = getattr(processor, "image_mean", [0.485, 0.456, 0.406])
    std = getattr(processor, "image_std", [0.229, 0.224, 0.225])
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(input_size, scale=(0.6, 1.0), antialias=True),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(5),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _snapshot_trainable_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }


def _restore_trainable_state(
    model: nn.Module,
    snapshot: dict[str, torch.Tensor],
    device: torch.device,
) -> None:
    named_parameters = dict(model.named_parameters())
    for name, value in snapshot.items():
        named_parameters[name].data.copy_(value.to(device))


def _assert_finite_gradients(model: nn.Module, module_name: str) -> None:
    for name, parameter in model.named_parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"Non-finite gradient in {module_name}.{name} on MPS.")


@torch.inference_mode()
def _find_first_nonfinite_layer(model: nn.Module, pixel_values: torch.Tensor) -> str:
    first_bad = "unknown"
    hooks = []

    def hook(name: str) -> Any:
        def _hook(_module: nn.Module, _inputs: tuple[Any, ...], output: Any) -> None:
            nonlocal first_bad
            if first_bad != "unknown":
                return
            tensors: list[torch.Tensor] = []
            if isinstance(output, torch.Tensor):
                tensors = [output]
            elif isinstance(output, (tuple, list)):
                tensors = [item for item in output if isinstance(item, torch.Tensor)]
            elif hasattr(output, "last_hidden_state"):
                tensors = [output.last_hidden_state]
            for tensor in tensors:
                if tensor.is_floating_point() and not torch.isfinite(tensor).all():
                    first_bad = name
                    return

        return _hook

    for name, module in model.named_modules():
        if name:
            hooks.append(module.register_forward_hook(hook(name)))
    try:
        model(pixel_values=pixel_values)
    except Exception as exc:
        if first_bad == "unknown":
            first_bad = f"forward_exception:{type(exc).__name__}"
    finally:
        for handle in hooks:
            handle.remove()
    return first_bad


def _move_optimizer_state(optimizer: AdamW, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def _select_device(default_device: torch.device, device_name: str) -> torch.device:
    if device_name == "auto":
        return default_device
    if device_name == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return torch.device("mps")
    if device_name == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported device: {device_name}")
