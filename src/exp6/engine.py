from __future__ import annotations

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

from data_utils import decode_mask
from evaluator import SegmentationMetric


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> str:
    if device_name == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_scheduler(optimizer, scheduler_type: str, epochs: int, min_lr_ratio: float = 0.1):
    if scheduler_type == "none":
        return None
    if scheduler_type != "cosine":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    epochs = max(1, int(epochs))
    min_lr_ratio = float(max(0.0, min(1.0, min_lr_ratio)))

    def _lr_lambda(epoch_idx: int) -> float:
        progress = min(max(epoch_idx / max(1, epochs - 1), 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)


class SegmentationLoss(torch.nn.Module):
    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        class_weights: torch.Tensor | None = None,
        ce_weight: float = 1.0,
        dice_weight: float = 0.3,
    ):
        super().__init__()
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights.float())
        else:
            self.class_weights = None
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.ce_weight = float(max(0.0, ce_weight))
        self.dice_weight = float(max(0.0, dice_weight))

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        total_loss = logits.new_tensor(0.0)

        if self.ce_weight > 0:
            ce = F.cross_entropy(
                logits,
                targets,
                ignore_index=self.ignore_index,
                weight=self.class_weights,
            )
            total_loss = total_loss + self.ce_weight * ce

        if self.dice_weight > 0:
            dice = self._dice_loss(logits, targets)
            total_loss = total_loss + self.dice_weight * dice

        return total_loss

    def _dice_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=1)
        valid_mask = targets.ne(self.ignore_index)
        safe_targets = torch.where(valid_mask, targets, torch.zeros_like(targets))

        one_hot = F.one_hot(safe_targets, num_classes=self.num_classes).permute(0, 3, 1, 2).float()
        valid = valid_mask.unsqueeze(1)
        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)
        denominator = probs.sum(dim=dims) + one_hot.sum(dim=dims)
        dice = (2.0 * intersection + 1e-6) / (denominator + 1e-6)

        present = one_hot.sum(dim=dims) > 0
        if not torch.any(present):
            return logits.new_tensor(0.0)

        if self.class_weights is not None:
            weights = self.class_weights[present]
            return 1.0 - (dice[present] * weights).sum() / weights.sum().clamp_min(1e-6)
        return 1.0 - dice[present].mean()


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device: str,
    criterion,
    grad_clip: float,
    use_amp: bool,
    scaler,
    log_interval: int,
) -> float:
    model.train()
    running_loss = 0.0

    for step, batch in enumerate(dataloader, start=1):
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(images)
            loss = criterion(logits, masks)

        if use_amp and scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        running_loss += float(loss.item())
        if step % log_interval == 0:
            print(f"[train] step={step} avg_loss={running_loss / step:.4f}")

    return running_loss / max(1, len(dataloader))


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device: str,
    criterion,
    num_classes: int,
    ignore_index: int,
) -> dict[str, float | list[float]]:
    model.eval()
    metric = SegmentationMetric(num_classes=num_classes, ignore_index=ignore_index)
    total_loss = 0.0

    for batch in dataloader:
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)

        logits = model(images)
        loss = criterion(logits, masks)
        total_loss += float(loss.item())

        preds = logits.argmax(dim=1)
        metric.update(preds=preds, targets=masks)

    stats = metric.compute()
    stats["loss"] = total_loss / max(1, len(dataloader))
    return stats


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    best_miou: float,
    ckpt_path: Path,
    scheduler=None,
    scaler=None,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": int(epoch),
            "best_miou": float(best_miou),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        },
        ckpt_path,
    )


def load_checkpoint(model, optimizer, ckpt_path: Path, device: str, scheduler=None, scaler=None) -> tuple[int, float]:
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])
    return int(state.get("epoch", 0)), float(state.get("best_miou", 0.0))


def append_jsonl(log_path: Path, record: dict) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


@torch.inference_mode()
def predict_and_save(
    model,
    dataloader,
    device: str,
    colors: tuple[tuple[int, int, int], ...],
    ignore_index: int,
    output_dir: Path,
) -> None:
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    for batch in dataloader:
        images = batch["image"].to(device)
        image_paths = batch["image_path"]
        logits = model(images)
        preds = logits.argmax(dim=1).detach().cpu().numpy()

        for image_path, pred in zip(image_paths, preds):
            color_mask = decode_mask(pred, colors=colors, ignore_index=ignore_index)
            out_path = output_dir / f"{Path(image_path).stem}_pred.png"
            Image.fromarray(color_mask).save(out_path)
