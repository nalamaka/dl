import os
import shutil
from pathlib import Path

import torch

import trainer
from data import build_mnist_loaders
from models import CNN
from trainer import train_model
from visualize_results import plot_prediction_cases, plot_training_curve


def run_one(optimizer_name, loss_name, label_smoothing=0.0):
    tag = f"{optimizer_name}_{loss_name}"
    print(f"\n=== Running: {tag} ===")

    # Update trainer-level config without modifying source files.
    trainer.OPTIMIZER = optimizer_name
    trainer.LOSS_FN = loss_name
    trainer.LABEL_SMOOTHING = label_smoothing

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = build_mnist_loaders()
    model = CNN().to(device)

    train_model(model, train_loader, test_loader, device)

    out_dir = Path("visualizations")
    ckpt_file = Path("checkpoint") / "best_cnn.pth"
    log_file = Path("log") / "train_log.csv"
    plot_training_curve(str(log_file), str(out_dir))
    plot_prediction_cases(str(ckpt_file), str(out_dir), device)

    # Archive each run to avoid being overwritten by the next run.
    run_dir = Path("runs") / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(log_file, run_dir / "train_log.csv")
    shutil.copy2(ckpt_file, run_dir / "best_cnn.pth")
    shutil.copy2(out_dir / "training_curve.png", run_dir / "training_curve.png")
    shutil.copy2(out_dir / "case_showcase.png", run_dir / "case_showcase.png")
    print(f"Saved run artifacts to: {run_dir}")


def main():
    os.makedirs("runs", exist_ok=True)
    combos = [
        ("adam", "ce", 0.0),
        ("adam", "ce_ls", 0.1),
        ("sgd", "ce", 0.0),
        ("sgd", "ce_ls", 0.1),
    ]
    for optimizer_name, loss_name, label_smoothing in combos:
        run_one(optimizer_name, loss_name, label_smoothing=label_smoothing)

    print("\nAll combinations finished.")


if __name__ == "__main__":
    main()
