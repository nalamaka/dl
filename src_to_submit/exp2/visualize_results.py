import csv
import glob
import os
import random

import matplotlib.pyplot as plt
import torch

from data import build_dataloader
from main import build_model, parse_args


CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")


def load_history(history_file):
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    with open(history_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            if "train_acc" in row:
                train_acc.append(float(row["train_acc"]))
            else:
                train_acc.append(float("nan"))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))
    return epochs, train_loss, train_acc, val_loss, val_acc


def plot_training(history_file, output_dir):
    epochs, train_loss, train_acc, val_loss, val_acc = load_history(history_file)
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, train_loss, marker="o", label="train_loss")
    axes[0].plot(epochs, val_loss, marker="s", label="val_loss")
    axes[0].set_title("Exp2 Loss Curve")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, train_acc, marker="o", color="tab:blue", label="train_acc")
    axes[1].plot(epochs, val_acc, marker="^", color="tab:green", label="val_acc")
    axes[1].axhline(75.0, color="tab:red", linestyle="--", linewidth=1.5, label="baseline=75%")
    axes[1].set_title("Exp2 Train/Validation Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy(%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    # axes[1].text(
    #     0.02,
    #     0.08,
    #     "Baseline = 75.0%",
    #     transform=axes[1].transAxes,
    #     color="tab:red",
    #     fontsize=9,
    #     bbox={"facecolor": "white", "edgecolor": "tab:red", "linewidth": 1.0, "alpha": 0.9},
    # )

    best_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
    best_epoch = epochs[best_idx]
    axes[1].scatter(best_epoch, val_acc[best_idx], color="red", s=60, zorder=5)
    axes[1].annotate(
        "BEST\n"
        f"epoch={best_epoch}\n"
        f"train_acc={train_acc[best_idx]:.3f}%\n"
        f"val_acc={val_acc[best_idx]:.3f}%\n"
        f"val_loss={val_loss[best_idx]:.4f}",
        xy=(best_epoch, val_acc[best_idx]),
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


def get_latest_checkpoint():
    candidates = glob.glob(os.path.join("checkpoint", "*-ckpt.t7"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def plot_cases(ckpt_file, output_dir, device):
    args = parse_args()
    _, testloader = build_dataloader(args.batch_size, args.image_size)
    model = build_model(args, device)

    state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state["net"])
    model.eval()

    images, labels, preds = [], [], []
    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            logits = model(x)
            p = torch.argmax(logits, dim=1).cpu()
            for i in range(x.size(0)):
                images.append(x[i].cpu())
                labels.append(int(y[i]))
                preds.append(int(p[i]))
            if len(images) >= 64:
                break

    picked = random.sample(range(len(images)), k=min(9, len(images)))
    fig = plt.figure(figsize=(10, 8))
    for idx, pick in enumerate(picked, start=1):
        ax = fig.add_subplot(3, 3, idx)
        img = images[pick].permute(1, 2, 0)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        ax.imshow(img)
        ax.set_title(f"gt={CLASSES[labels[pick]]}, pred={CLASSES[preds[pick]]}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "case_showcase.png"), dpi=150)
    plt.close(fig)


def main():
    history_file = os.path.join("log", "history.csv")
    if not os.path.exists(history_file):
        raise FileNotFoundError("log/history.csv not found. Run training first: python main.py")

    ckpt_file = get_latest_checkpoint()
    if ckpt_file is None:
        raise FileNotFoundError("checkpoint/*.t7 not found. Run training first: python main.py")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "visualizations"

    plot_training(history_file, output_dir)
    plot_cases(ckpt_file, output_dir, device)

    print("Saved: visualizations/training_curve.png")
    print("Saved: visualizations/case_showcase.png")


if __name__ == "__main__":
    main()
