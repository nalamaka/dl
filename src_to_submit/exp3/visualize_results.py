#!/usr/bin/env python3
"""
Training and evaluation visualization utilities for Transformer poetry generation.
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as fm

try:
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None


class TransformerTrainingVisualizer:
    """Visualizer for Transformer training metrics with English labels."""

    def __init__(self, save_dir: str = "visualizations"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        plt.style.use("seaborn-v0_8-whitegrid")
        if sns is not None:
            sns.set_palette("husl")
        self.font_name = self._configure_chinese_font()

        self.colors = {
            "train_loss": "#2E86AB",
            "val_loss": "#A23B72",
            "learning_rate": "#F18F01",
            "perplexity": "#C73E1D",
            "gradient_norm": "#52B788",
            "quality_score": "#3A5A40",
            "rhyme_score": "#6D597A",
            "format_score": "#BC6C25",
        }

    @staticmethod
    def _configure_chinese_font() -> str | None:
        candidates = [
            "Microsoft YaHei",
            "SimHei",
            "Noto Sans CJK SC",
            "WenQuanYi Zen Hei",
            "PingFang SC",
            "Heiti SC",
            "Arial Unicode MS",
        ]
        available = {f.name for f in fm.fontManager.ttflist}
        chosen = next((name for name in candidates if name in available), None)
        if chosen is not None:
            plt.rcParams["font.sans-serif"] = [chosen] + list(plt.rcParams.get("font.sans-serif", []))
        plt.rcParams["axes.unicode_minus"] = False
        return chosen

    def plot_training_dashboard(
        self,
        metrics: Dict[str, List[float]],
        epochs: List[int],
        title: str = "Transformer Poetry Training Dashboard",
        save_name: str = "training_curve.png",
    ) -> str:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(title, fontsize=18, fontweight="bold", y=0.96)

        ax1 = axes[0, 0]
        if metrics.get("train_loss"):
            ax1.plot(
                epochs,
                metrics["train_loss"],
                color=self.colors["train_loss"],
                linewidth=2.4,
                marker="o",
                markersize=4,
                label="Train Loss",
            )
        if metrics.get("val_loss"):
            ax1.plot(
                epochs,
                metrics["val_loss"],
                color=self.colors["val_loss"],
                linewidth=2.4,
                marker="s",
                markersize=4,
                label="Validation Loss",
            )
        ax1.set_title("Loss Curves", fontsize=13, fontweight="bold")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax1.set_yscale("log")
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)

        ax2 = axes[0, 1]
        if metrics.get("learning_rate"):
            ax2.plot(
                epochs,
                metrics["learning_rate"],
                color=self.colors["learning_rate"],
                linewidth=2.4,
                marker="d",
                markersize=4,
            )
            ax2.set_yscale("log")
        else:
            ax2.text(
                0.5,
                0.5,
                "Learning rate is unavailable",
                transform=ax2.transAxes,
                ha="center",
                va="center",
                color="gray",
            )
        ax2.set_title("Learning Rate", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Learning Rate")
        ax2.grid(True, alpha=0.3)

        ax3 = axes[1, 0]
        if metrics.get("perplexity"):
            ax3.plot(
                epochs,
                metrics["perplexity"],
                color=self.colors["perplexity"],
                linewidth=2.4,
                marker="^",
                markersize=4,
            )
            ax3.set_yscale("log")
        elif metrics.get("train_loss"):
            ppl = [float(np.exp(v)) for v in metrics["train_loss"]]
            ax3.plot(
                epochs,
                ppl,
                color=self.colors["perplexity"],
                linewidth=2.4,
                marker="^",
                markersize=4,
            )
            ax3.set_yscale("log")
        else:
            ax3.text(
                0.5,
                0.5,
                "Perplexity is unavailable",
                transform=ax3.transAxes,
                ha="center",
                va="center",
                color="gray",
            )
        ax3.set_title("Perplexity", fontsize=13, fontweight="bold")
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("PPL")
        ax3.grid(True, alpha=0.3)

        ax4 = axes[1, 1]
        if metrics.get("gradient_norm"):
            ax4.plot(
                epochs,
                metrics["gradient_norm"],
                color=self.colors["gradient_norm"],
                linewidth=2.4,
                marker="*",
                markersize=6,
            )
        else:
            ax4.text(
                0.5,
                0.5,
                "Gradient norm is unavailable",
                transform=ax4.transAxes,
                ha="center",
                va="center",
                color="gray",
            )
        ax4.set_title("Gradient Norm", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("Norm")
        ax4.grid(True, alpha=0.3)

        if metrics.get("train_loss"):
            best_idx = int(np.argmin(metrics["train_loss"]))
            ax1.scatter(epochs[best_idx], metrics["train_loss"][best_idx], color="red", s=60, zorder=6)
            ax1.annotate(
                f"Best Train Loss\nEpoch={epochs[best_idx]}\nLoss={metrics['train_loss'][best_idx]:.4f}",
                xy=(epochs[best_idx], metrics["train_loss"][best_idx]),
                xytext=(10, -40),
                textcoords="offset points",
                color="red",
                fontsize=8,
                va="top",
                bbox={"facecolor": "white", "edgecolor": "red", "linewidth": 1.2, "alpha": 0.9},
                arrowprops={"arrowstyle": "->", "color": "red", "linewidth": 1.0},
            )

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return str(save_path)

    def plot_eval_quality(
        self,
        eval_rows: List[dict],
        save_name: str = "eval_quality.png",
    ) -> str:
        quality = [float(_quality_score(x.get("metrics", {}))) for x in eval_rows]
        rhyme = [float(x.get("metrics", {}).get("rhyme_score", 0.0)) for x in eval_rows]
        fmt = [float(x.get("metrics", {}).get("format_score", 0.0)) for x in eval_rows]
        tone = [float(x.get("metrics", {}).get("zeqi_pingshou_score", 0.0)) for x in eval_rows]
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        bins = min(12, max(5, len(quality) // 2))
        axes[0].hist(
            quality,
            bins=bins,
            color=self.colors["quality_score"],
            alpha=0.88,
            edgecolor="white",
            linewidth=0.7,
        )
        axes[0].axvline(float(np.mean(quality)), color="black", linestyle="--", linewidth=1.5, label="Mean")
        axes[0].set_title("Quality Score Distribution", fontsize=13, fontweight="bold")
        axes[0].set_xlabel("Quality Score")
        axes[0].set_ylabel("Count")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        metric_names = ["Quality", "Rhyme", "Format", "Tone"]
        metric_values = [float(np.mean(quality)), float(np.mean(rhyme)), float(np.mean(fmt)), float(np.mean(tone))]
        metric_colors = [
            self.colors["quality_score"],
            self.colors["rhyme_score"],
            self.colors["format_score"],
            "#4E79A7",
        ]
        axes[1].bar(metric_names, metric_values, color=metric_colors, alpha=0.9)
        axes[1].set_title("Mean Metric Scores", fontsize=13, fontweight="bold")
        axes[1].set_ylabel("Score")
        axes[1].set_ylim(0.0, 1.05)
        for i, v in enumerate(metric_values):
            axes[1].text(i, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        axes[1].grid(True, alpha=0.3)

        axes[2].axis("off")
        summary_rows = [
            ["Samples", f"{len(quality)}"],
            ["Quality Mean", f"{np.mean(quality):.4f}"],
            ["Quality Median", f"{np.median(quality):.4f}"],
            ["Quality Min", f"{np.min(quality):.4f}"],
            ["Quality Max", f"{np.max(quality):.4f}"],
            ["Rhyme Mean", f"{np.mean(rhyme):.4f}"],
            ["Format Mean", f"{np.mean(fmt):.4f}"],
            ["Tone Mean", f"{np.mean(tone):.4f}"],
        ]
        table = axes[2].table(
            cellText=summary_rows,
            colLabels=["Metric", "Value"],
            loc="center",
            cellLoc="center",
            colLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.1, 1.5)
        axes[2].set_title("Summary Table", fontsize=13, fontweight="bold")

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return str(save_path)

    def plot_case_showcase(self, eval_rows: List[dict], save_name: str = "case_showcase.png", k: int = 6) -> str:
        if not eval_rows:
            raise ValueError("No evaluation rows provided.")

        pick_n = min(k, len(eval_rows))
        sampled = eval_rows[:pick_n] if len(eval_rows) <= pick_n else list(np.random.choice(eval_rows, size=pick_n, replace=False))

        rows = int(np.ceil(pick_n / 2))
        cols = 2
        fig, axes = plt.subplots(rows, cols, figsize=(14, 3.8 * rows))
        if rows == 1:
            axes = np.array([axes])
        flat_axes = axes.flatten()

        for i, ax in enumerate(flat_axes):
            if i >= pick_n:
                ax.axis("off")
                continue
            item = sampled[i]
            prompt = str(item.get("prompt", ""))
            poem = str(item.get("poem", ""))
            metrics = item.get("metrics", {})
            quality = _quality_score(metrics)
            rhyme = float(metrics.get("rhyme_score", 0.0))
            fmt = float(metrics.get("format_score", 0.0))
            tone = float(metrics.get("zeqi_pingshou_score", 0.0))

            title = f"Prompt: {prompt[:14]} | Q={quality:.3f}"
            text = f"{poem}\n\nformat={fmt:.3f}, rhyme={rhyme:.3f}, tone={tone:.3f}"
            ax.axis("off")
            ax.set_title(title, fontsize=10, pad=8)
            text_kwargs = {"va": "top", "ha": "left", "fontsize": 10, "wrap": True}
            if self.font_name:
                text_kwargs["fontfamily"] = self.font_name
            ax.text(0.01, 0.96, text, **text_kwargs)

        plt.tight_layout()
        save_path = self.save_dir / save_name
        plt.savefig(save_path, dpi=220, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        return str(save_path)


def _read_history(path: Path) -> Dict[str, List[float]]:
    metrics = {
        "epoch": [],
        "train_loss": [],
        "val_loss": [],
        "learning_rate": [],
        "perplexity": [],
        "gradient_norm": [],
    }
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics["epoch"].append(int(float(row["epoch"])))
            metrics["train_loss"].append(float(row.get("train_loss", row.get("avg_loss", "nan"))))
            val_loss_raw = row.get("val_loss", "")
            metrics["val_loss"].append(float(val_loss_raw) if val_loss_raw else np.nan)
            metrics["learning_rate"].append(float(row.get("learning_rate", row.get("lr", "nan"))))
            metrics["perplexity"].append(float(row.get("perplexity", row.get("ppl", "nan"))))
            metrics["gradient_norm"].append(float(row.get("gradient_norm", row.get("grad_norm", "nan"))))

    for key in ("val_loss", "learning_rate", "perplexity", "gradient_norm"):
        vals = metrics[key]
        metrics[key] = [x for x in vals if not np.isnan(x)] if vals and all(np.isnan(v) for v in vals) else vals
    return metrics


def _read_eval_rows(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _quality_score(metrics: dict) -> float:
    if "quality_score" in metrics:
        return float(metrics["quality_score"])
    return (
        0.2 * float(metrics.get("format_score", 0.0))
        + 0.2 * float(metrics.get("rhyme_score", 0.0))
        + 0.15 * float(metrics.get("zeqi_pingshou_score", 0.0))
        + 0.1 * float(metrics.get("tone_valid_ratio", 0.0))
        + 0.05 * float(metrics.get("acrostic_score", 0.0))
        + 0.15 * float(metrics.get("distinct1", 0.0))
        + 0.1 * (1.0 - float(metrics.get("repetition2", 0.0)))
        + 0.05 * (1.0 / max(1.0, float(metrics.get("ppl", 1.0))))
    )


def _parse_args():
    parser = argparse.ArgumentParser(description="Visualize Exp3 training and evaluation logs.")
    parser.add_argument("--history_file", type=str, default="log/history.csv")
    parser.add_argument("--eval_file", type=str, default="log/eval_results.jsonl")
    parser.add_argument("--output_dir", type=str, default="visualizations")
    parser.add_argument("--title", type=str, default="Transformer Poetry Generation Training")
    parser.add_argument("--save_prefix", type=str, default="")
    return parser.parse_args()


def main():
    args = _parse_args()
    history_file = Path(args.history_file)
    eval_file = Path(args.eval_file)
    visualizer = TransformerTrainingVisualizer(save_dir=args.output_dir)

    if not history_file.exists():
        raise FileNotFoundError(f"{history_file} not found. Run training first.")

    metrics = _read_history(history_file)
    epochs = metrics.pop("epoch")
    prefix = args.save_prefix if args.save_prefix else datetime.now().strftime("%Y%m%d_%H%M%S")

    dashboard_path = visualizer.plot_training_dashboard(
        metrics=metrics,
        epochs=epochs,
        title=args.title,
        save_name=f"{prefix}_training_curve.png",
    )
    print(f"Saved: {dashboard_path}")

    if eval_file.exists():
        eval_rows = _read_eval_rows(eval_file)
        quality_path = visualizer.plot_eval_quality(eval_rows, save_name=f"{prefix}_eval_quality.png")
        showcase_path = visualizer.plot_case_showcase(eval_rows, save_name=f"{prefix}_case_showcase.png")
        print(f"Saved: {quality_path}")
        print(f"Saved: {showcase_path}")
    else:
        print(f"Skip evaluation plots because {eval_file} does not exist.")


if __name__ == "__main__":
    main()
