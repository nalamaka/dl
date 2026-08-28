from __future__ import annotations

import argparse
import json
from pathlib import Path


def _infer_initial_lr(steps: list[int], values: list[float], fit_points: int = 5) -> float | None:
    n = min(len(steps), len(values), max(2, fit_points))
    if n < 2:
        return None

    xs = [float(x) for x in steps[:n]]
    ys = [float(y) for y in values[:n]]
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    denom = sum((x - x_mean) ** 2 for x in xs)
    if denom == 0:
        return None

    slope = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denom
    intercept = y_mean - slope * x_mean
    return max(0.0, float(intercept))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize exp4 training logs")
    parser.add_argument("--log_file", type=str, required=True, help="Path to jsonl log file")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for plots")
    parser.add_argument("--show", action="store_true", help="Show figures interactively")
    return parser.parse_args()


def _load_records(log_file: Path) -> list[dict]:
    records: list[dict] = []
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                records.append(json.loads(s))
            except json.JSONDecodeError:
                continue
    return records


def _select_bleu_series(epoch_recs: list[dict]) -> tuple[list[float], str]:
    if any("val_bleu4_raw" in r for r in epoch_recs):
        return [float(r.get("val_bleu4", 0.0)) for r in epoch_recs], "BLEU4"
    return [float(r.get("val_bleu4", 0.0)) * 100.0 for r in epoch_recs], "BLEU4"


def main() -> None:
    args = parse_args()
    log_file = Path(args.log_file)
    if not log_file.exists():
        raise FileNotFoundError(f"log file not found: {log_file}")

    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "matplotlib is required for visualization. Install via: pip install matplotlib"
        ) from exc

    records = _load_records(log_file)
    step_recs = [r for r in records if r.get("type") == "step"]
    epoch_recs = [r for r in records if r.get("type") == "epoch"]

    out_dir = Path(args.out_dir) if args.out_dir else (log_file.parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    if step_recs:
        steps = [int(r.get("global_step", 0)) for r in step_recs]
        losses = [float(r.get("loss", 0.0)) for r in step_recs]
        lr_step_recs = [r for r in step_recs if bool(r.get("optimizer_step", False))]
        lr_steps = [int(r.get("global_step", 0)) for r in lr_step_recs]
        lrs = [float(r.get("lr", 0.0)) for r in lr_step_recs]

        plt.figure(figsize=(9, 4.5))
        plt.plot(steps, losses, linewidth=1.2)
        plt.title("Step Loss Curve")
        plt.xlabel("Global Step")
        plt.ylabel("Loss")
        plt.grid(alpha=0.25)
        if steps:
            plt.xlim(left=min(steps))
        plt.tight_layout()
        step_loss_path = out_dir / "step_loss.png"
        plt.savefig(step_loss_path, dpi=150)

        if lr_steps and lrs:
            inferred_lr0 = _infer_initial_lr(lr_steps, lrs, fit_points=5)
            if lr_steps[0] != 0 and inferred_lr0 is not None:
                lr_steps = [0] + lr_steps
                lrs = [inferred_lr0] + lrs

            plt.figure(figsize=(9, 4.5))
            plt.plot(lr_steps, lrs, linewidth=1.2)
            plt.title("Learning Rate Curve")
            plt.xlabel("Global Step (Optimizer Update)")
            plt.ylabel("LR")
            plt.grid(alpha=0.25)
            plt.xlim(left=0)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.tight_layout()
            lr_path = out_dir / "lr_curve.png"
            plt.savefig(lr_path, dpi=150)
        else:
            lr_path = None
    else:
        step_loss_path = None
        lr_path = None

    if epoch_recs:
        epochs = [int(r.get("epoch", 0)) for r in epoch_recs]
        train_losses = [float(r.get("train_loss", 0.0)) for r in epoch_recs]
        val_losses = [float(r.get("val_loss", 0.0)) for r in epoch_recs]
        val_bleu, bleu_ylabel = _select_bleu_series(epoch_recs)

        plt.figure(figsize=(9, 4.5))
        plt.plot(epochs, train_losses, marker="o", label="train_loss")
        plt.plot(epochs, val_losses, marker="o", label="val_loss")
        plt.title("Epoch Loss Curves")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(alpha=0.25)
        plt.tight_layout()
        epoch_loss_path = out_dir / "epoch_loss.png"
        plt.savefig(epoch_loss_path, dpi=150)

        plt.figure(figsize=(9, 4.5))
        plt.plot(epochs, val_bleu, marker="o", color="tab:green")
        plt.title("Validation BLEU4 Curve")
        plt.xlabel("Epoch")
        plt.ylabel(bleu_ylabel)
        plt.grid(alpha=0.25)
        plt.tight_layout()
        bleu_path = out_dir / "val_bleu4.png"
        plt.savefig(bleu_path, dpi=150)
    else:
        epoch_loss_path = None
        bleu_path = None

    if args.show:
        plt.show()

    print(f"[viz] log file: {log_file}")
    print(f"[viz] output dir: {out_dir}")
    if step_loss_path:
        print(f"[viz] step loss: {step_loss_path}")
    if lr_path:
        print(f"[viz] lr curve: {lr_path}")
    if epoch_loss_path:
        print(f"[viz] epoch loss: {epoch_loss_path}")
    if bleu_path:
        print(f"[viz] val BLEU4: {bleu_path}")


if __name__ == "__main__":
    main()
