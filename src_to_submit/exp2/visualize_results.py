import csv
import glob
import os
import random

import matplotlib.pyplot as plt
import torch
import torchvision
import torchvision.transforms as transforms

from config import DATA_ROOT
from main import build_model, parse_args


CLASSES = ("plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck")
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
PREDICTION_FILE = os.path.join("visualizations", "predictions.pt")


def load_history(history_file):
    epochs, train_loss, train_acc, val_loss, val_acc = [], [], [], [], []
    with open(history_file, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            epochs.append(int(row["epoch"]))
            train_loss.append(float(row["train_loss"]))
            train_acc.append(float(row.get("train_acc", "nan")))
            val_loss.append(float(row["val_loss"]))
            val_acc.append(float(row["val_acc"]))
    return epochs, train_loss, train_acc, val_loss, val_acc


def plot_training(history_file, output_dir):
    epochs, train_loss, train_acc, val_loss, val_acc = load_history(history_file)
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
    best_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
    axes[1].scatter(epochs[best_idx], val_acc[best_idx], color="red", s=60, zorder=5, label="best_val")
    axes[1].set_title("Exp2 Train/Validation Accuracy")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy(%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "training_curve.png"), dpi=150)
    plt.close(fig)


def get_latest_checkpoint():
    candidates = glob.glob(os.path.join("checkpoint", "*-ckpt.t7"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def load_model(ckpt_file, args, device):
    model = build_model(args, device)
    state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state["net"])
    model.eval()
    return model


def build_testset(args):
    valid_resize = int(round(args.image_size * 256 / 224))
    trans_valid = transforms.Compose(
        [
            transforms.Resize(valid_resize),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ]
    )
    try:
        testset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=False, download=False, transform=trans_valid
        )
    except RuntimeError:
        testset = torchvision.datasets.CIFAR10(
            root=DATA_ROOT, train=False, download=True, transform=trans_valid
        )
    return testset


def load_predictions(pred_file):
    state = torch.load(pred_file, map_location="cpu")
    labels = state["labels"]
    preds = state["preds"]
    return labels, preds


def denormalize_image(img):
    mean = torch.tensor(MEAN).view(3, 1, 1)
    std = torch.tensor(STD).view(3, 1, 1)
    return torch.clamp(img * std + mean, 0.0, 1.0)


def plot_cases_from_predictions(testset, labels, preds, output_dir, num_cases=9):
    valid_ids = list(range(min(len(testset), len(preds))))
    picked = random.sample(valid_ids, k=min(num_cases, len(valid_ids)))

    fig = plt.figure(figsize=(10, 8))
    for idx, pick in enumerate(picked, start=1):
        img, gt = testset[pick]
        pred = preds[pick]
        ax = fig.add_subplot(3, 3, idx)
        img = denormalize_image(img).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"gt={CLASSES[gt]}, pred={CLASSES[pred]}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "case_showcase.png"), dpi=150)
    plt.close(fig)


def plot_report(history_file, testset, labels, preds, output_dir):
    epochs, train_loss, train_acc, val_loss, val_acc = load_history(history_file)
    valid_ids = list(range(min(len(testset), len(preds))))
    picked = random.sample(valid_ids, k=min(8, len(valid_ids)))

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 4, hspace=0.45, wspace=0.35)

    ax_loss = fig.add_subplot(gs[0, 0:2])
    ax_loss.plot(epochs, train_loss, marker="o", label="train_loss")
    ax_loss.plot(epochs, val_loss, marker="s", label="val_loss")
    ax_loss.set_title("Loss Curve")
    ax_loss.set_xlabel("epoch")
    ax_loss.set_ylabel("loss")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc = fig.add_subplot(gs[0, 2:4])
    ax_acc.plot(epochs, train_acc, marker="o", color="tab:blue", label="train_acc")
    ax_acc.plot(epochs, val_acc, marker="^", color="tab:green", label="val_acc")
    best_idx = max(range(len(val_acc)), key=lambda i: val_acc[i])
    ax_acc.scatter(epochs[best_idx], val_acc[best_idx], color="red", s=60, zorder=5, label="best_val")
    ax_acc.set_title("Accuracy Curve")
    ax_acc.set_xlabel("epoch")
    ax_acc.set_ylabel("accuracy(%)")
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    for i, pick in enumerate(picked):
        r = 1 + i // 4
        c = i % 4
        img, gt = testset[pick]
        pred = preds[pick]
        ax = fig.add_subplot(gs[r, c])
        img = denormalize_image(img).permute(1, 2, 0).numpy()
        ax.imshow(img)
        ax.set_title(f"gt={CLASSES[gt]}\npred={CLASSES[pred]}", fontsize=9)
        ax.axis("off")

    fig.suptitle("Training Curves + Prediction Cases", fontsize=14)
    fig.savefig(os.path.join(output_dir, "visual_report.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def compute_class_heatmap(model, image_tensor, class_idx):
    x = image_tensor.unsqueeze(0).clone().detach().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(x)
    score = logits[0, class_idx]
    score.backward()
    grad = x.grad.detach().abs()[0]
    heatmap = grad.mean(dim=0)
    heatmap = heatmap - heatmap.min()
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap.cpu().numpy()


def plot_class_heatmaps(model, testset, labels, preds, output_dir, device, num_samples=8):
    wrong_ids = [i for i, (y, p) in enumerate(zip(labels, preds)) if y != p]
    if not wrong_ids:
        return
    picked = wrong_ids[: min(num_samples, len(wrong_ids))]

    fig, axes = plt.subplots(len(picked), 3, figsize=(12, 4 * len(picked)))
    if len(picked) == 1:
        axes = [axes]

    for row, idx in enumerate(picked):
        img, gt = testset[idx]
        pred = preds[idx]
        img = img.to(device)

        gt_heat = compute_class_heatmap(model, img, gt)
        pred_heat = compute_class_heatmap(model, img, pred)
        misfocus = pred_heat - gt_heat
        misfocus = misfocus - misfocus.min()
        misfocus = misfocus / (misfocus.max() + 1e-8)
        base = denormalize_image(img.cpu()).permute(1, 2, 0).numpy()

        ax0, ax1, ax2 = axes[row]
        ax0.imshow(base)
        ax0.set_title(f"Input\nGT={CLASSES[gt]}, Pred={CLASSES[pred]}")
        ax0.axis("off")

        ax1.imshow(base)
        ax1.imshow(gt_heat, cmap="jet", alpha=0.45)
        ax1.set_title(f"GT Attention: {CLASSES[gt]}")
        ax1.axis("off")

        ax2.imshow(base)
        ax2.imshow(misfocus, cmap="jet", alpha=0.5)
        ax2.set_title(f"Misclassified Focus: {CLASSES[pred]} > {CLASSES[gt]}")
        ax2.axis("off")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "misclassified_heatmaps.png"), dpi=150)
    plt.close(fig)


def build_confusion_matrix(labels, preds, num_classes):
    conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    for y, p in zip(labels, preds):
        conf[y, p] += 1
    return conf


def plot_class_statistics_bar_and_confusion(labels, preds, output_dir):
    num_classes = len(CLASSES)
    conf = build_confusion_matrix(labels, preds, num_classes)
    class_total = conf.sum(dim=1).float().clamp(min=1.0)
    class_acc = (conf.diag().float() / class_total).tolist()
    avg_acc = sum(class_acc) / len(class_acc)

    x = list(range(num_classes))
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(x, class_acc, color="#8bbcd4", edgecolor="#3b4cc0", linewidth=1.2)
    ax.axhline(avg_acc, color="red", linestyle="--", linewidth=1.2, label=f"Average Accuracy: {avg_acc:.3f}")
    ax.set_ylim(0, 1.1)
    ax.set_xticks(x, CLASSES, rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("Accuracy")
    ax.set_title("Classification Accuracy by Class")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    for i, b in enumerate(bars):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01, f"{class_acc[i]:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "class_accuracy_by_class.png"), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 8))
    im = ax.imshow(conf.numpy(), cmap="Blues")
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted Class")
    ax.set_ylabel("True Class")
    ax.set_xticks(x, CLASSES, rotation=45, ha="right")
    ax.set_yticks(x, CLASSES)
    threshold = conf.max().item() * 0.6 if conf.numel() > 0 else 0
    for i in range(num_classes):
        for j in range(num_classes):
            val = int(conf[i, j].item())
            color = "white" if val > threshold else "black"
            ax.text(j, i, f"{val}", ha="center", va="center", color=color, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    class_total_int = conf.sum(dim=1).tolist()
    class_correct_int = conf.diag().tolist()
    class_wrong_int = (conf.sum(dim=1) - conf.diag()).tolist()

    fig, ax = plt.subplots(figsize=(12, 5))
    w = 0.28
    ax.bar([i - w for i in x], class_total_int, width=w, label="total", color="tab:blue")
    ax.bar(x, class_correct_int, width=w, label="correct", color="tab:green")
    ax.bar([i + w for i in x], class_wrong_int, width=w, label="wrong", color="tab:red")
    ax.set_xticks(x, CLASSES, rotation=45, ha="right")
    ax.set_xlabel("Class")
    ax.set_ylabel("count")
    ax.set_title("Per-class Statistics")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "class_statistics_bar.png"), dpi=150)
    plt.close(fig)


def main():
    history_file = os.path.join("log", "history.csv")
    if not os.path.exists(history_file):
        raise FileNotFoundError("log/history.csv not found. Run training first: python main.py")
    if not os.path.exists(PREDICTION_FILE):
        raise FileNotFoundError(
            "visualizations/predictions.pt not found. Run prediction first: python predict_results.py"
        )

    ckpt_file = get_latest_checkpoint()
    if ckpt_file is None:
        raise FileNotFoundError("checkpoint/*.t7 not found. Run training first: python main.py")

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = "visualizations"
    os.makedirs(output_dir, exist_ok=True)

    labels, preds = load_predictions(PREDICTION_FILE)
    testset = build_testset(args)
    if len(preds) != len(testset) or len(labels) != len(testset):
        raise ValueError("Prediction length does not match dataset length. Please rerun: python predict_results.py")

    plot_training(history_file, output_dir)
    plot_cases_from_predictions(testset, labels, preds, output_dir)
    plot_report(history_file, testset, labels, preds, output_dir)
    plot_class_statistics_bar_and_confusion(labels, preds, output_dir)

    model = load_model(ckpt_file, args, device)
    plot_class_heatmaps(model, testset, labels, preds, output_dir, device, num_samples=8)

    print("Saved: visualizations/training_curve.png")
    print("Saved: visualizations/case_showcase.png")
    print("Saved: visualizations/visual_report.png")
    print("Saved: visualizations/misclassified_heatmaps.png")
    print("Saved: visualizations/class_accuracy_by_class.png")
    print("Saved: visualizations/confusion_matrix.png")
    print("Saved: visualizations/class_statistics_bar.png")


if __name__ == "__main__":
    main()
