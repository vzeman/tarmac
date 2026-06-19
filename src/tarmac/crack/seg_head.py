from __future__ import annotations

import json
import math
import random
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageEnhance, ImageFile
from rich.console import Console
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from tarmac.datasets.crack500_seg import find_crack500_seg_pairs
from tarmac.datasets.crackairport import find_crackairport_pairs
from tarmac.datasets.crackforest import find_crackforest_pairs
from tarmac.datasets.cracktree260 import find_cracktree260_pairs, find_crkwh100_pairs
from tarmac.datasets.cssc import find_cssc_pairs
from tarmac.datasets.deepcrack_liu import find_deepcrack_liu_pairs
from tarmac.datasets.khanh11k import find_khanh11k_pairs
from tarmac.embedding.embedder import DINOV3_MODEL, HFBackboneEmbedder
from tarmac.inference.analyze import load_active_artifacts

ImageFile.LOAD_TRUNCATED_IMAGES = True

SEED = 42
DEFAULT_IMAGE_SIZE = 512
DEFAULT_PATCH_SIZE = 16
DEFAULT_CHECKPOINT = Path("models/crack_seg_head.pt")
DEFAULT_METADATA = Path("models/crack_seg_head.json")
DEFAULT_METRICS = Path("reports/crack_seg_head_metrics.json")
DEFAULT_REPORT = Path("reports/CRACK_SEGMENTATION.md")
DEFAULT_CHECKPOINT_DIR = Path("models/checkpoints/seg_head")
DEFAULT_MANIFEST = Path("data/processed/crack_seg_expanded/manifest.jsonl")


def seed_everything(seed: int = SEED) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def require_mps(device: str = "mps") -> str:
    normalized = device.lower()
    if normalized != "mps":
        raise RuntimeError("Dense crack segmentation training/evaluation requires MPS. Pass --device mps.")
    if not torch.backends.mps.is_available():
        raise RuntimeError("Apple MPS is not available; refusing to silently fall back to CPU.")
    return "mps"


@dataclass(frozen=True)
class SegRecord:
    image_path: Path
    mask_path: Path
    split: str
    source_dataset: str


@dataclass(frozen=True)
class SegHeadTrainConfig:
    manifest_path: str
    output_checkpoint: str
    output_metadata: str
    checkpoint_dir: str
    image_size: int
    patch_size: int
    epochs: int
    batch_size: int
    lr: float
    weight_decay: float
    patience: int
    seed: int
    device: str
    hidden_dim: int
    dropout: float
    pos_weight: float
    backbone_model_name: str
    backbone_checkpoint: str


@dataclass
class PixelStats:
    tp: int = 0
    fp: int = 0
    fn: int = 0
    tn: int = 0

    def update(self, pred: torch.Tensor, target: torch.Tensor) -> None:
        pred_bool = pred.bool()
        target_bool = target.bool()
        self.tp += int((pred_bool & target_bool).sum().item())
        self.fp += int((pred_bool & ~target_bool).sum().item())
        self.fn += int((~pred_bool & target_bool).sum().item())
        self.tn += int((~pred_bool & ~target_bool).sum().item())

    def metrics(self) -> dict[str, float | int]:
        tp = float(self.tp)
        fp = float(self.fp)
        fn = float(self.fn)
        tn = float(self.tn)
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        iou = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
        dice = 2.0 * tp / (2.0 * tp + fp + fn) if 2.0 * tp + fp + fn > 0 else 0.0
        accuracy = (tp + tn) / (tp + fp + fn + tn) if tp + fp + fn + tn > 0 else 0.0
        return {
            "iou": float(iou),
            "dice": float(dice),
            "f1": float(dice),
            "precision": float(precision),
            "recall": float(recall),
            "accuracy": float(accuracy),
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
        }


class DenseCrackSegHead(nn.Module):
    """Lightweight decoder over frozen DINO patch-token feature maps."""

    def __init__(self, input_dim: int = 768, hidden_dim: int = 256, dropout: float = 0.05) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.proj = nn.Sequential(
            nn.Conv2d(input_dim, hidden_dim, kernel_size=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.GELU(),
        )
        self.decoder = nn.Sequential(
            _upsample_block(hidden_dim, 128),
            _upsample_block(128, 64),
            _upsample_block(64, 32),
            _upsample_block(32, 16),
            nn.Conv2d(16, 1, kernel_size=1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.proj(features))


class CrackSegMaskDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        records: list[SegRecord],
        processor: Any,
        *,
        image_size: int = DEFAULT_IMAGE_SIZE,
        augment: bool = False,
    ) -> None:
        self.records = records
        self.processor = processor
        self.image_size = image_size
        self.augment = augment

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        with Image.open(record.image_path) as image:
            rgb = image.convert("RGB").resize((self.image_size, self.image_size), Image.Resampling.BILINEAR)
        with Image.open(record.mask_path) as mask_image:
            mask = mask_image.convert("L").resize((self.image_size, self.image_size), Image.Resampling.NEAREST)

        if self.augment:
            rgb, mask = _augment_pair(rgb, mask)

        pixel_values = self.processor(images=rgb, return_tensors="pt")["pixel_values"][0]
        mask_arr = (np.asarray(mask, dtype=np.uint8) > 0).astype("float32")
        return {
            "pixel_values": pixel_values,
            "mask": torch.from_numpy(mask_arr).unsqueeze(0),
            "image_path": str(record.image_path),
            "mask_path": str(record.mask_path),
            "split": record.split,
            "source_dataset": record.source_dataset,
        }


@dataclass
class LearnedSegHeadBundle:
    checkpoint_path: Path
    metadata: dict[str, Any]
    threshold: float
    image_size: int
    patch_size: int
    patch_grid: int
    embedder: HFBackboneEmbedder
    head: DenseCrackSegHead
    device: torch.device


_BUNDLE_CACHE: dict[tuple[str, float | None], LearnedSegHeadBundle] = {}


def train_seg_head(
    manifest_path: Path = DEFAULT_MANIFEST,
    output_checkpoint: Path = DEFAULT_CHECKPOINT,
    output_metadata: Path = DEFAULT_METADATA,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT_DIR,
    epochs: int = 60,
    batch_size: int = 4,
    lr: float = 2e-4,
    weight_decay: float = 1e-4,
    patience: int = 8,
    seed: int = SEED,
    resume: bool = False,
    num_workers: int = 0,
    device: str = "mps",
    image_size: int = DEFAULT_IMAGE_SIZE,
    hidden_dim: int = 256,
    dropout: float = 0.05,
) -> dict[str, Any]:
    require_mps(device)
    seed_everything(seed)
    torch.set_num_threads(1)
    console = Console()
    device_obj = torch.device("mps")
    records = load_seg_records(manifest_path, seed=seed)
    train_records = [record for record in records if record.split == "train"]
    val_records = [record for record in records if record.split == "val"]
    if not train_records or not val_records:
        raise RuntimeError("Segmentation training requires non-empty train and val splits.")

    embedder = _load_active_backbone(image_size=image_size, device=device)
    input_dim = int(getattr(embedder.model.config, "hidden_size", 768))
    patch_size = int(getattr(embedder.model.config, "patch_size", DEFAULT_PATCH_SIZE))
    patch_grid = image_size // patch_size
    head = DenseCrackSegHead(input_dim=input_dim, hidden_dim=hidden_dim, dropout=dropout).to(device_obj)
    optimizer = AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)

    pos_weight_value = _estimate_pos_weight(train_records, image_size=image_size)
    pos_weight = torch.tensor([pos_weight_value], dtype=torch.float32, device=device_obj)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    output_metadata.parent.mkdir(parents=True, exist_ok=True)
    config = SegHeadTrainConfig(
        manifest_path=str(manifest_path),
        output_checkpoint=str(output_checkpoint),
        output_metadata=str(output_metadata),
        checkpoint_dir=str(checkpoint_dir),
        image_size=image_size,
        patch_size=patch_size,
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        seed=seed,
        device=device,
        hidden_dim=hidden_dim,
        dropout=dropout,
        pos_weight=pos_weight_value,
        backbone_model_name=embedder.model_name,
        backbone_checkpoint=str(load_active_artifacts().checkpoint_path),
    )

    start_epoch = 1
    best_epoch = 0
    best_val_dice = -1.0
    best_threshold = 0.5
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    if resume:
        latest = _latest_epoch_checkpoint(checkpoint_dir)
        if latest is None:
            raise RuntimeError(f"--resume was requested but no checkpoints exist in {checkpoint_dir}.")
        state = torch.load(latest, map_location=device_obj)
        head.load_state_dict(state["head_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        history = list(state.get("history", []))
        best_epoch = int(state.get("best_epoch", 0))
        best_val_dice = float(state.get("best_val_dice", -1.0))
        best_threshold = float(state.get("best_threshold", state.get("threshold", 0.5)))
        epochs_without_improvement = int(state.get("epochs_without_improvement", 0))
        start_epoch = int(state["epoch"]) + 1

    train_loader = DataLoader(
        CrackSegMaskDataset(train_records, embedder.processor, image_size=image_size, augment=True),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    console.print(
        f"Training DINOv3 dense crack segmenter on MPS; records={len(records)} "
        f"train={len(train_records)} val={len(val_records)} patch_grid={patch_grid}x{patch_grid} "
        f"pos_weight={pos_weight_value:.3f}"
    )
    for epoch in range(start_epoch, epochs + 1):
        head.train()
        losses: list[float] = []
        progress = tqdm(train_loader, desc=f"Seg head epoch {epoch}/{epochs}", unit="batch")
        for batch in progress:
            masks = batch["mask"].to(device_obj, non_blocking=False)
            with torch.no_grad():
                features = _dense_patch_features(
                    embedder,
                    batch["pixel_values"],
                    patch_grid=patch_grid,
                    device=device_obj,
                )
            logits = head(features)
            loss = bce(logits, masks) + dice_loss(logits, masks)
            if torch.isnan(loss).any():
                raise RuntimeError("NaN loss while training segmentation head on MPS.")
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            losses.append(loss_value)
            progress.set_postfix(loss=f"{loss_value:.4f}")

        val_threshold, val_metrics = _choose_threshold(
            head=head,
            embedder=embedder,
            records=val_records,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            patch_grid=patch_grid,
        )
        row = {
            "epoch": epoch,
            "loss": float(np.mean(losses)) if losses else 0.0,
            "val_iou": float(val_metrics["iou"]),
            "val_dice": float(val_metrics["dice"]),
            "val_precision": float(val_metrics["precision"]),
            "val_recall": float(val_metrics["recall"]),
            "val_threshold": float(val_threshold),
        }
        history.append(row)
        improved = float(row["val_dice"]) > best_val_dice
        if improved:
            best_epoch = epoch
            best_val_dice = float(row["val_dice"])
            best_threshold = float(val_threshold)
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
            "image_size": image_size,
            "patch_size": patch_size,
            "patch_grid": patch_grid,
            "model_name": embedder.model_name,
            "backbone_checkpoint": str(load_active_artifacts().checkpoint_path),
            "config": asdict(config),
            "history": history,
            "epoch": epoch,
            "best_epoch": best_epoch,
            "best_val_dice": best_val_dice,
            "best_threshold": best_threshold,
            "threshold": float(val_threshold),
            "epochs_without_improvement": epochs_without_improvement,
        }
        torch.save(state, epoch_checkpoint)
        if improved:
            shutil.copy2(epoch_checkpoint, output_checkpoint)
            shutil.copy2(epoch_checkpoint, checkpoint_dir / "best.pt")
        metadata = {
            "config": asdict(config),
            "history": history,
            "best_epoch": best_epoch,
            "best_val_dice": best_val_dice,
            "threshold": best_threshold,
            "checkpoint": str(output_checkpoint),
            "latest_epoch_checkpoint": str(epoch_checkpoint),
        }
        output_metadata.write_text(json.dumps(metadata, indent=2) + "\n")
        console.print(
            f"epoch={epoch} loss={row['loss']:.4f} val_iou={row['val_iou']:.4f} "
            f"val_dice={row['val_dice']:.4f} val_threshold={row['val_threshold']:.3f}"
        )
        if epochs_without_improvement >= patience:
            console.print(f"Early stopping at epoch {epoch}; patience={patience}.")
            break

    return {
        "checkpoint": str(output_checkpoint),
        "metadata": str(output_metadata),
        "best_epoch": best_epoch,
        "best_val_dice": best_val_dice,
        "history": history,
    }


def evaluate_seg_head(
    manifest_path: Path = DEFAULT_MANIFEST,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    metadata_path: Path = DEFAULT_METADATA,
    metrics_path: Path = DEFAULT_METRICS,
    report_path: Path = DEFAULT_REPORT,
    batch_size: int = 4,
    num_workers: int = 0,
    device: str = "mps",
    render_examples: bool = True,
    examples_dir: Path = Path("reports/examples"),
    compare_classical: bool = True,
) -> dict[str, Any]:
    require_mps(device)
    torch.set_num_threads(1)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing segmentation head checkpoint: {checkpoint_path}")
    records = load_seg_records(manifest_path)
    eval_records = [record for record in records if record.split in {"val", "test"}]
    if not eval_records:
        raise RuntimeError("No val/test records available for segmentation evaluation.")

    state = torch.load(checkpoint_path, map_location=torch.device("mps"))
    image_size = int(state.get("image_size", DEFAULT_IMAGE_SIZE))
    patch_grid = int(state.get("patch_grid", image_size // int(state.get("patch_size", DEFAULT_PATCH_SIZE))))
    embedder = _load_active_backbone(image_size=image_size, device=device)
    head = DenseCrackSegHead(
        input_dim=int(state.get("input_dim", getattr(embedder.model.config, "hidden_size", 768))),
        hidden_dim=int(state.get("hidden_dim", 256)),
        dropout=float(state.get("dropout", 0.05)),
    ).to(torch.device("mps"))
    head.load_state_dict(state["head_state_dict"])
    head.eval()

    val_records = [record for record in eval_records if record.split == "val"]
    val_threshold, val_metrics = _choose_threshold(
        head=head,
        embedder=embedder,
        records=val_records,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
        patch_grid=patch_grid,
    )
    threshold = val_threshold
    split_metrics: dict[str, dict[str, dict[str, float | int]]] = {}
    for split in ("val", "test"):
        split_records = [record for record in eval_records if record.split == split]
        split_metrics[split] = _metrics_for_records_by_source(
            head=head,
            embedder=embedder,
            records=split_records,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
            patch_grid=patch_grid,
            threshold=threshold,
        )
    split_metrics["val"]["overall"] = val_metrics

    comparison: dict[str, Any] = {
        "dinov3_dense_head": split_metrics["test"].get("overall", {}),
    }
    test_records = [record for record in eval_records if record.split == "test"]
    if compare_classical and test_records:
        comparison["classical"] = evaluate_classical_segmenter(test_records, image_size=image_size)

    example_paths: list[str] = []
    if render_examples:
        example_paths = render_crackairport_examples(
            checkpoint_path=checkpoint_path,
            records=test_records,
            output_dir=examples_dir,
            count=3,
        )

    result = {
        "checkpoint": str(checkpoint_path),
        "metadata": str(metadata_path),
        "manifest": str(manifest_path),
        "threshold": threshold,
        "val": split_metrics["val"],
        "test": split_metrics["test"],
        "comparison": comparison,
        "example_paths": example_paths,
    }
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(result, indent=2) + "\n")
    _update_checkpoint_threshold(checkpoint_path, metadata_path, threshold)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(_markdown_report(result) + "\n")
    return result


def load_seg_records(manifest_path: Path = DEFAULT_MANIFEST, *, seed: int = SEED) -> list[SegRecord]:
    if manifest_path.exists():
        rows = [json.loads(line) for line in manifest_path.read_text().splitlines() if line.strip()]
        records = []
        for row in rows:
            image_path = Path(str(row.get("source_image") or row["image_path"])).expanduser()
            mask_path = Path(str(row.get("source_mask") or row.get("mask_path") or row["label_path"])).expanduser()
            records.append(
                SegRecord(
                    image_path=image_path,
                    mask_path=mask_path,
                    split=str(row["split"]),
                    source_dataset=str(row.get("source_dataset", "unknown")),
                )
            )
        return records
    return _fallback_records(seed=seed)


@torch.inference_mode()
def predict_crack_mask(
    image: Image.Image,
    *,
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    threshold: float | None = None,
    device_name: str = "auto",
) -> tuple[np.ndarray, np.ndarray]:
    bundle = load_learned_segmenter(
        checkpoint_path=checkpoint_path,
        threshold=threshold,
        device_name=device_name,
    )
    width, height = image.size
    rgb = image.convert("RGB").resize((bundle.image_size, bundle.image_size), Image.Resampling.BILINEAR)
    pixel_values = bundle.embedder.processor(images=rgb, return_tensors="pt")["pixel_values"]
    features = _dense_patch_features(
        bundle.embedder,
        pixel_values,
        patch_grid=bundle.patch_grid,
        device=bundle.device,
    )
    logits = bundle.head(features)
    probs = torch.sigmoid(logits)
    probs = F.interpolate(probs, size=(height, width), mode="bilinear", align_corners=False)
    heatmap = probs.squeeze(0).squeeze(0).detach().cpu().numpy().astype("float32")
    mask = heatmap >= bundle.threshold
    return mask.astype(bool), heatmap


def load_learned_segmenter(
    checkpoint_path: Path = DEFAULT_CHECKPOINT,
    *,
    threshold: float | None = None,
    device_name: str = "auto",
) -> LearnedSegHeadBundle:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing learned crack segmenter: {checkpoint_path}")
    device = _prediction_device(device_name)
    cache_key = (f"{checkpoint_path.resolve()}::{device.type}", threshold)
    cached = _BUNDLE_CACHE.get(cache_key)
    if cached is not None:
        return cached

    state = torch.load(checkpoint_path, map_location=device)
    metadata_path = DEFAULT_METADATA if checkpoint_path == DEFAULT_CHECKPOINT else checkpoint_path.with_suffix(".json")
    metadata = json.loads(metadata_path.read_text()) if metadata_path.exists() else {}
    image_size = int(state.get("image_size", metadata.get("image_size", DEFAULT_IMAGE_SIZE)))
    patch_size = int(state.get("patch_size", DEFAULT_PATCH_SIZE))
    patch_grid = int(state.get("patch_grid", image_size // patch_size))
    embedder = _load_active_backbone(image_size=image_size, device=device.type, require_accelerator=False)
    head = DenseCrackSegHead(
        input_dim=int(state.get("input_dim", getattr(embedder.model.config, "hidden_size", 768))),
        hidden_dim=int(state.get("hidden_dim", 256)),
        dropout=float(state.get("dropout", 0.05)),
    ).to(device)
    head.load_state_dict(state["head_state_dict"])
    head.eval()
    chosen_threshold = float(threshold if threshold is not None else state.get("threshold", metadata.get("threshold", 0.5)))
    bundle = LearnedSegHeadBundle(
        checkpoint_path=checkpoint_path,
        metadata=metadata,
        threshold=chosen_threshold,
        image_size=image_size,
        patch_size=patch_size,
        patch_grid=patch_grid,
        embedder=embedder,
        head=head,
        device=device,
    )
    _BUNDLE_CACHE[cache_key] = bundle
    return bundle


def render_crackairport_examples(
    checkpoint_path: Path,
    records: list[SegRecord],
    output_dir: Path = Path("reports/examples"),
    count: int = 3,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    crackairport_records = [
        record for record in records if record.source_dataset == "crackairport" and record.mask_path.exists()
    ]
    selected = crackairport_records[:count]
    paths: list[str] = []
    for index, record in enumerate(selected, start=1):
        with Image.open(record.image_path) as image:
            rgb = image.convert("RGB")
        with Image.open(record.mask_path) as mask_image:
            gt = mask_image.convert("L").resize(rgb.size, Image.Resampling.NEAREST)
        pred_mask, _heatmap = predict_crack_mask(rgb, checkpoint_path=checkpoint_path)
        panel = _example_panel(rgb, gt, pred_mask)
        name = "08_crack_seg_learned.png" if index == 1 else f"08_crack_seg_learned_{index}.png"
        out_path = output_dir / name
        panel.save(out_path, quality=92)
        paths.append(str(out_path))
    return paths


def evaluate_classical_segmenter(records: list[SegRecord], *, image_size: int = DEFAULT_IMAGE_SIZE) -> dict[str, float | int]:
    from tarmac.crack.segment import extract_crack_mask

    stats = PixelStats()
    heatmap = np.ones((image_size, image_size), dtype=np.float32)
    for record in tqdm(records, desc="Classical crack eval", unit="image"):
        with Image.open(record.image_path) as image:
            rgb_image = image.convert("RGB").resize((image_size, image_size), Image.Resampling.BILINEAR)
        with Image.open(record.mask_path) as mask_image:
            target = np.asarray(mask_image.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)) > 0
        pred = extract_crack_mask(np.asarray(rgb_image, dtype=np.uint8), heatmap=heatmap)
        stats.update(torch.from_numpy(pred), torch.from_numpy(target))
    metrics = stats.metrics()
    metrics["count"] = len(records)
    return metrics


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1.0) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    dims = tuple(range(1, probs.ndim))
    intersection = (probs * targets).sum(dim=dims)
    denominator = probs.sum(dim=dims) + targets.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def _upsample_block(in_channels: int, out_channels: int) -> nn.Sequential:
    groups = 8 if out_channels >= 32 else 4
    return nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
        nn.GroupNorm(groups, out_channels),
        nn.GELU(),
    )


def _prediction_device(device_name: str) -> torch.device:
    normalized = device_name.lower()
    if normalized == "auto":
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    if normalized == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("Apple MPS is not available; pass --device cpu for crack segmentation inference.")
        return torch.device("mps")
    if normalized == "cpu":
        return torch.device("cpu")
    raise ValueError(f"Unsupported crack segmentation device: {device_name}")


def _load_active_backbone(*, image_size: int, device: str, require_accelerator: bool = True) -> HFBackboneEmbedder:
    if require_accelerator:
        require_mps(device)
        device_name = "mps"
    else:
        device_name = _prediction_device(device).type
    active = load_active_artifacts()
    embedder = HFBackboneEmbedder(
        model_name=active.model_name or DINOV3_MODEL,
        checkpoint_path=active.checkpoint_path,
        allow_fallback=False,
        attn_implementation="eager",
        device_name=device_name,
    )
    if require_accelerator and embedder.device.type != "mps":
        raise RuntimeError(f"Dense crack segmentation requires MPS; selected device was {embedder.device.type}.")
    _set_processor_image_size(embedder.processor, image_size)
    embedder.model.eval()
    for parameter in embedder.model.parameters():
        parameter.requires_grad_(False)
    return embedder


def _set_processor_image_size(processor: Any, image_size: int) -> None:
    processor.size = {"height": image_size, "width": image_size}
    if hasattr(processor, "crop_size"):
        processor.crop_size = {"height": image_size, "width": image_size}


def _dense_patch_features(
    embedder: HFBackboneEmbedder,
    pixel_values: torch.Tensor,
    *,
    patch_grid: int,
    device: torch.device,
) -> torch.Tensor:
    values = pixel_values.to(device)
    outputs = embedder.model(pixel_values=values)
    hidden = outputs.last_hidden_state.float()
    patch_count = patch_grid * patch_grid
    if hidden.shape[1] < patch_count:
        raise RuntimeError(f"Backbone returned {hidden.shape[1]} tokens, fewer than {patch_count} patch tokens.")
    patch_tokens = hidden[:, -patch_count:, :]
    if torch.isnan(patch_tokens).any():
        raise RuntimeError("DINOv3 patch tokens contain NaNs on MPS.")
    return patch_tokens.transpose(1, 2).reshape(hidden.shape[0], hidden.shape[2], patch_grid, patch_grid).contiguous()


def _estimate_pos_weight(records: list[SegRecord], *, image_size: int, cap: float = 30.0) -> float:
    positive = 0
    total = 0
    for record in tqdm(records, desc="Scanning crack mask balance", unit="mask"):
        with Image.open(record.mask_path) as mask_image:
            mask = np.asarray(mask_image.convert("L").resize((image_size, image_size), Image.Resampling.NEAREST)) > 0
        positive += int(mask.sum())
        total += int(mask.size)
    negative = max(0, total - positive)
    if positive <= 0:
        return 1.0
    return float(min(cap, max(1.0, negative / positive)))


@torch.inference_mode()
def _evaluate_head_loader(
    *,
    head: DenseCrackSegHead,
    embedder: HFBackboneEmbedder,
    loader: DataLoader,
    device: torch.device,
    patch_grid: int,
    threshold: float,
    desc: str,
) -> dict[str, float | int]:
    head.eval()
    stats = PixelStats()
    logit_threshold = math.log(threshold / (1.0 - threshold))
    for batch in tqdm(loader, desc=desc, unit="batch"):
        masks = batch["mask"].to(device)
        features = _dense_patch_features(embedder, batch["pixel_values"], patch_grid=patch_grid, device=device)
        logits = head(features)
        pred = logits >= logit_threshold
        stats.update(pred.detach().cpu(), masks.detach().cpu() > 0.5)
    return stats.metrics()


def _metrics_for_records_by_source(
    *,
    head: DenseCrackSegHead,
    embedder: HFBackboneEmbedder,
    records: list[SegRecord],
    image_size: int,
    batch_size: int,
    num_workers: int,
    patch_grid: int,
    threshold: float,
) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    if not records:
        return result
    groups = {"overall": records}
    for source in sorted({record.source_dataset for record in records}):
        groups[source] = [record for record in records if record.source_dataset == source]
    for name, group_records in groups.items():
        loader = DataLoader(
            CrackSegMaskDataset(group_records, embedder.processor, image_size=image_size, augment=False),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
        metrics = _evaluate_head_loader(
            head=head,
            embedder=embedder,
            loader=loader,
            device=torch.device("mps"),
            patch_grid=patch_grid,
            threshold=threshold,
            desc=f"Seg eval {name}",
        )
        metrics["count"] = len(group_records)
        result[name] = metrics
    return result


def _choose_threshold(
    *,
    head: DenseCrackSegHead,
    embedder: HFBackboneEmbedder,
    records: list[SegRecord],
    image_size: int,
    batch_size: int,
    num_workers: int,
    patch_grid: int,
) -> tuple[float, dict[str, float | int]]:
    if not records:
        return 0.5, {}
    thresholds = [round(float(x), 3) for x in np.linspace(0.05, 0.95, 37)]
    stats_by_threshold = {threshold: PixelStats() for threshold in thresholds}
    loader = DataLoader(
        CrackSegMaskDataset(records, embedder.processor, image_size=image_size, augment=False),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    head.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Choosing seg threshold", unit="batch"):
            masks = batch["mask"].to("mps") > 0.5
            features = _dense_patch_features(embedder, batch["pixel_values"], patch_grid=patch_grid, device=torch.device("mps"))
            probs = torch.sigmoid(head(features))
            for threshold, stats in stats_by_threshold.items():
                stats.update((probs >= threshold).detach().cpu(), masks.detach().cpu())
    best_threshold = 0.5
    best_metrics: dict[str, float | int] = {}
    best_dice = -1.0
    for threshold, stats in stats_by_threshold.items():
        metrics = stats.metrics()
        if float(metrics["dice"]) > best_dice:
            best_dice = float(metrics["dice"])
            best_threshold = threshold
            best_metrics = metrics
    best_metrics["count"] = len(records)
    return best_threshold, best_metrics


def _augment_pair(image: Image.Image, mask: Image.Image) -> tuple[Image.Image, Image.Image]:
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if random.random() < 0.25:
        image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    if random.random() < 0.35:
        image = ImageEnhance.Contrast(image).enhance(random.uniform(0.85, 1.25))
    if random.random() < 0.35:
        image = ImageEnhance.Brightness(image).enhance(random.uniform(0.9, 1.1))
    return image, mask


def _fallback_records(seed: int = SEED) -> list[SegRecord]:
    records: list[tuple[str, Path, Path]] = []
    records.extend(("crackairport", image, mask) for image, mask in _crackairport_pairs(Path("data/raw/crackairport")))
    records.extend(("crackforest", image, mask) for image, mask in find_crackforest_pairs(Path("data/raw/crackforest")))
    records.extend(("deepcrack_liu", image, mask) for image, mask in find_deepcrack_liu_pairs(Path("data/raw/deepcrack_liu")))
    records.extend(("crack500", image, mask) for image, mask in find_crack500_seg_pairs(Path("data/raw/crack500_seg")))
    records.extend(("cracktree260", image, mask) for image, mask in find_cracktree260_pairs(Path("data/raw/cracktree260")))
    records.extend(("crkwh100", image, mask) for image, mask in find_crkwh100_pairs(Path("data/raw/cracktree260")))
    records.extend(("khanh11k", image, mask) for image, mask in find_khanh11k_pairs(Path("data/raw/khanh11k")))
    records.extend(("cssc", image, mask) for image, mask in find_cssc_pairs(Path("data/raw/cssc")))
    by_source: dict[str, list[tuple[str, Path, Path]]] = {}
    for source, image, mask in records:
        by_source.setdefault(source, []).append((source, image, mask))
    rng = np.random.default_rng(seed)
    split_records: list[SegRecord] = []
    for source in sorted(by_source):
        items = sorted(by_source[source], key=lambda item: str(item[1]))
        indexes = np.arange(len(items))
        rng.shuffle(indexes)
        train_end, val_end = _split_bounds(len(indexes))
        for split, split_indexes in (
            ("train", indexes[:train_end]),
            ("val", indexes[train_end:val_end]),
            ("test", indexes[val_end:]),
        ):
            for index in split_indexes:
                _source, image, mask = items[int(index)]
                split_records.append(SegRecord(image, mask, split, source))
    return sorted(split_records, key=lambda record: (record.split, record.source_dataset, str(record.image_path)))


def _crackairport_pairs(raw_dir: Path) -> list[tuple[Path, Path]]:
    pairs_path = raw_dir / "pairs.jsonl"
    if pairs_path.exists():
        return [
            (Path(row["image_path"]), Path(row["mask_path"]))
            for row in (json.loads(line) for line in pairs_path.read_text().splitlines() if line.strip())
        ]
    return find_crackairport_pairs(raw_dir / "_extracted")


def _split_bounds(count: int) -> tuple[int, int]:
    if count <= 0:
        return 0, 0
    if count == 1:
        return 1, 1
    if count == 2:
        return 1, 2
    train_count = max(1, int(round(0.70 * count)))
    val_count = max(1, int(round(0.15 * count)))
    if train_count + val_count >= count:
        train_count = max(1, count - 2)
        val_count = 1
    return train_count, train_count + val_count


def _latest_epoch_checkpoint(checkpoint_dir: Path) -> Path | None:
    checkpoints = []
    for path in checkpoint_dir.glob("epoch_*.pt"):
        try:
            epoch = int(path.stem.split("_", 1)[1])
        except (IndexError, ValueError):
            continue
        checkpoints.append((epoch, path))
    if not checkpoints:
        return None
    return sorted(checkpoints)[-1][1]


def _update_checkpoint_threshold(checkpoint_path: Path, metadata_path: Path, threshold: float) -> None:
    state = torch.load(checkpoint_path, map_location=torch.device("mps"))
    state["threshold"] = float(threshold)
    torch.save(state, checkpoint_path)
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
    else:
        metadata = {}
    metadata["threshold"] = float(threshold)
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")


def _example_panel(image: Image.Image, gt_mask: Image.Image, pred_mask: np.ndarray) -> Image.Image:
    width, height = image.size
    gt_rgb = Image.new("RGB", image.size, (0, 0, 0))
    gt_arr = np.zeros((height, width, 3), dtype=np.uint8)
    gt_arr[np.asarray(gt_mask, dtype=np.uint8) > 0] = (235, 22, 35)
    gt_rgb = Image.fromarray(gt_arr, mode="RGB")
    pred_overlay = image.convert("RGBA")
    red = np.zeros((height, width, 4), dtype=np.uint8)
    red[pred_mask.astype(bool)] = (235, 22, 35, 145)
    pred_overlay = Image.alpha_composite(pred_overlay, Image.fromarray(red, mode="RGBA")).convert("RGB")
    panel = Image.new("RGB", (width * 3, height), (255, 255, 255))
    panel.paste(image.convert("RGB"), (0, 0))
    panel.paste(gt_rgb, (width, 0))
    panel.paste(pred_overlay, (width * 2, 0))
    return panel


def _markdown_report(result: dict[str, Any]) -> str:
    rows = []
    for split in ("val", "test"):
        for source, metrics in result.get(split, {}).items():
            rows.append(
                "| {split} | {source} | {iou:.4f} | {dice:.4f} | {precision:.4f} | "
                "{recall:.4f} | {accuracy:.4f} | {count} |".format(split=split, source=source, **metrics)
            )
    comparison_rows = []
    for name, metrics in result.get("comparison", {}).items():
        if not metrics:
            continue
        comparison_rows.append(
            "| {name} | {iou:.4f} | {dice:.4f} | {precision:.4f} | {recall:.4f} | {count} |".format(
                name=name,
                **metrics,
            )
        )
    example_lines = "\n".join(f"- `{path}`" for path in result.get("example_paths", [])) or "- Not rendered."
    return f"""# Crack Segmentation and Measurement

Phase 7c added full-frame runway/pavement analysis and pixel-level crack geometry. Track 2 replaces the default pixel mask with a learned frozen-DINOv3 dense-token segmentation head when `models/crack_seg_head.pt` is present.

## Default Segmenter

Default for `tarmac crack-measure` and `tarmac analyze --crack-segmentation`: `dinov3_dense_head`.

The learned model uses the active fine-tuned DINOv3 ViT-B/16 backbone frozen at 512 px input resolution. The 32x32 patch-token grid is decoded by a lightweight convolutional upsampler to a full-resolution crack logit map. The classical Frangi/Sato/black-hat method remains the fallback only when the learned checkpoint is absent.

Chosen threshold: `{result["threshold"]:.3f}` (max Dice on validation).

## Dense-Head Metrics

| Split | Source | IoU | Dice/F1 | Precision | Recall | Pixel accuracy | Images |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(rows)}

## Common Test Comparison

Pixel metrics below use the held-out split from `{result["manifest"]}` when present, otherwise the deterministic CrackAirport + CrackForest raw-data split, with masks resized to 512 px.

| Segmenter | IoU | Dice/F1 | Precision | Recall | Images |
| --- | ---: | ---: | ---: | ---: | ---: |
{chr(10).join(comparison_rows) if comparison_rows else "| Not run | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |"}

Verdict: the DINOv3 dense head is the selected default because it is trained directly on pixel masks from CrackAirport and CrackForest and produces a full-resolution mask that keeps the existing area, length, and width measurement path intact. The classical method remains the no-checkpoint fallback.

## Example Overlays

License-safe CrackAirport examples, left to right: original, ground-truth mask, learned prediction.

{example_lines}

## Outputs

- `tarmac crack-measure <image|dir> --out DIR`: `<name>_crackseg.png`, `crack_measurements.csv`, `crack_measurements.parquet`.
- `tarmac analyze --region full --crack-segmentation`: `crackseg/frame_*_crackseg.png` plus geometry columns in `results.parquet`.
- `tarmac report`: crack geometry overlay gallery and measurement table.
"""
