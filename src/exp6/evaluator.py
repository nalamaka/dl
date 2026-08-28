from __future__ import annotations

import numpy as np
import torch


class SegmentationMetric:
    def __init__(self, num_classes: int, ignore_index: int = 255):
        self.num_classes = int(num_classes)
        self.ignore_index = int(ignore_index)
        self.confusion_matrix = np.zeros((self.num_classes, self.num_classes), dtype=np.int64)

    def reset(self) -> None:
        self.confusion_matrix.fill(0)

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        preds_np = preds.detach().cpu().numpy().astype(np.int64)
        targets_np = targets.detach().cpu().numpy().astype(np.int64)

        mask = (targets_np != self.ignore_index) & (targets_np >= 0) & (targets_np < self.num_classes)
        labels = self.num_classes * targets_np[mask] + preds_np[mask]
        counts = np.bincount(labels, minlength=self.num_classes ** 2)
        self.confusion_matrix += counts.reshape(self.num_classes, self.num_classes)

    def compute(self) -> dict[str, float | list[float]]:
        cm = self.confusion_matrix.astype(np.float64)
        total = cm.sum()
        correct = np.diag(cm).sum()

        pixel_acc = float(correct / total) if total > 0 else 0.0

        class_total = cm.sum(axis=1)
        class_acc = np.divide(
            np.diag(cm),
            class_total,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=class_total > 0,
        )
        mean_pixel_acc = float(class_acc[class_total > 0].mean()) if np.any(class_total > 0) else 0.0

        union = cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm)
        iou = np.divide(
            np.diag(cm),
            union,
            out=np.zeros(self.num_classes, dtype=np.float64),
            where=union > 0,
        )
        mean_iou = float(iou[union > 0].mean()) if np.any(union > 0) else 0.0

        freq = np.divide(class_total, total, out=np.zeros(self.num_classes, dtype=np.float64), where=total > 0)
        fw_iou = float((freq * iou).sum()) if total > 0 else 0.0

        return {
            "pixel_accuracy": pixel_acc,
            "mean_pixel_accuracy": mean_pixel_acc,
            "mean_iou": mean_iou,
            "frequency_weighted_iou": fw_iou,
            "per_class_iou": [float(x) for x in iou.tolist()],
        }

