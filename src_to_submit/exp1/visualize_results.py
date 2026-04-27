import csv
import os
import random

import matplotlib.pyplot as plt
import torch

from data import build_mnist_loaders
from models import CNN


def load_history(log_file):
    epochs = []
    losses = []
    train_accs = []
    test_accs = []
    with open(log_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            losses.append(float(row["loss"]))
            if "train_accuracy" in row:
                train_accs.append(float(row["train_accuracy"]))
                test_accs.append(float(row["test_accuracy"]))
            else:
                train_accs.append(float("nan"))
                test_accs.append(float(row["accuracy"]))
    return epochs, losses, train_accs, test_accs


def plot_training_curve(log_file, output_dir):
    epochs, losses, train_accs, test_accs = load_history(log_file)
    if not epochs:
        return

    os.makedirs(output_dir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(losses, label="loss")
    axes[0].set_title("Exp1 Loss Curve")
    axes[0].set_xlabel("log step")
    axes[0].set_ylabel("loss")
    axes[0].grid(alpha=0.3)

    axes[1].plot(train_accs, label="train_acc", color="tab:blue")
    axes[1].plot(test_accs, label="test_acc", color="tab:green")
    axes[1].axhline(0.98, color="tab:red", linestyle="--", linewidth=1.5, label="baseline=98%")
    axes[1].set_title("Exp1 Train/Test Accuracy")
    axes[1].set_xlabel("log step")
    axes[1].set_ylabel("accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    # axes[1].text(
    #     0.02,
    #     0.08,
    #     "Baseline = 98.0%",
    #     transform=axes[1].transAxes,
    #     color="tab:red",
    #     fontsize=9,
    #     bbox={"facecolor": "white", "edgecolor": "tab:red", "linewidth": 1.0, "alpha": 0.9},
    # )

    best_idx = max(range(len(test_accs)), key=lambda i: test_accs[i])
    axes[1].scatter(best_idx, test_accs[best_idx], color="red", s=60, zorder=5)
    axes[1].annotate(
        "BEST(TEST)\n"
        f"step={best_idx}\n"
        f"train_acc={train_accs[best_idx]:.4f}\n"
        f"test_acc={test_accs[best_idx]:.4f}\n"
        f"loss={losses[best_idx]:.4f}",
        xy=(best_idx, test_accs[best_idx]),
        xytext=(10, -55),
        textcoords="offset points",
        color="red",
        fontsize=8,
        va="top",
        bbox={"facecolor": "white", "edgecolor": "red", "linewidth": 1.5, "alpha": 0.9},
        arrowprops={"arrowstyle": "->", "color": "red", "linewidth": 1.2},
    )

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(output_dir, "training_curve.png"), dpi=150)
    plt.close(fig)


def plot_prediction_cases(checkpoint_file, output_dir, device):
    _, test_loader = build_mnist_loaders()
    model = CNN().to(device)
    model.load_state_dict(torch.load(checkpoint_file, map_location=device))
    model.eval()

    images = []
    labels = []
    preds = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x)
            batch_pred = torch.argmax(logits, dim=1).cpu()

            for i in range(batch_x.size(0)):
                images.append(batch_x[i].cpu())
                labels.append(int(batch_y[i]))
                preds.append(int(batch_pred[i]))
            if len(images) >= 64:
                break

    picked = random.sample(range(len(images)), k=min(9, len(images)))
    fig = plt.figure(figsize=(8, 8))
    for idx, pick in enumerate(picked, start=1):
        ax = fig.add_subplot(3, 3, idx)
        ax.imshow(images[pick].squeeze(0), cmap="gray")
        ax.set_title(f"gt={labels[pick]}, pred={preds[pick]}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "case_showcase.png"), dpi=150)
    plt.close(fig)


def main():
    output_dir = "visualizations"
    log_file = os.path.join("log", "train_log.csv")
    ckpt_file = os.path.join("checkpoint", "best_cnn.pth")

    if not os.path.exists(log_file):
        raise FileNotFoundError("log/train_log.csv not found. Run training first: python main.py")
    if not os.path.exists(ckpt_file):
        raise FileNotFoundError("checkpoint/best_cnn.pth not found. Run training first: python main.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_training_curve(log_file, output_dir)
    plot_prediction_cases(ckpt_file, output_dir, device)
    print("Saved: visualizations/training_curve.png")
    print("Saved: visualizations/case_showcase.png")


if __name__ == "__main__":
    main()
