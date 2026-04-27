import argparse
import csv
import os

import torch
import torch.nn as nn

from config import (
    BATCH_SIZE,
    EPOCHS,
    IMAGE_SIZE,
    LABEL_SMOOTHING,
    LOSS_FN,
    LR,
    MOMENTUM,
    NET,
    OPTIMIZER,
    PATCH_SIZE,
    USE_COSINE,
    WEIGHT_DECAY,
)
from data import build_dataloader
from engine import evaluate, train_one_epoch
from model import Vit


def parse_args():
    parser = argparse.ArgumentParser(description="ViT CIFAR10 Training")
    parser.add_argument("--net", default=NET)
    parser.add_argument("--patch", type=int, default=PATCH_SIZE)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE)
    parser.add_argument("--cos", action="store_true", default=USE_COSINE, help="Use cosine scheduler")
    parser.add_argument("--optimizer", type=str, default=OPTIMIZER)
    parser.add_argument("--loss", type=str, default=LOSS_FN)
    parser.add_argument("--weight_decay", type=float, default=WEIGHT_DECAY)
    parser.add_argument("--momentum", type=float, default=MOMENTUM)
    parser.add_argument("--label_smoothing", type=float, default=LABEL_SMOOTHING)
    return parser.parse_args()


def build_model(args, device):
    return Vit(
        image_size=args.image_size,
        patch_size=args.patch,
        num_classes=10,
        dim=256,
        depth=6,
        heads=8,
        mlp_dim=512,
        pool="cls",
        channels=3,
        dim_head=32,
        dropout=0.2,
        emb_dropout=0.2,
    ).to(device)


def build_optimizer(args, net):
    name = args.optimizer.lower()
    if name == "adamw":
        return torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            net.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}. Use 'adamw' or 'sgd'.")


def build_criterion(args):
    name = args.loss.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "ce_ls":
        return nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    raise ValueError(f"Unsupported loss: {args.loss}. Use 'ce' or 'ce_ls'.")


if __name__ == "__main__":
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trainloader, testloader = build_dataloader(args.batch_size, args.image_size)
    net = build_model(args, device)

    criterion = build_criterion(args)
    optimizer = build_optimizer(args, net)
    if args.cos:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=2
        )

    os.makedirs("log", exist_ok=True)
    history_path = os.path.join("log", "history.csv")
    with open(history_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        best_acc = 0.0
        for epoch in range(1, args.epochs + 1):
            train_loss, train_acc = train_one_epoch(
                epoch, net, trainloader, optimizer, criterion, device
            )
            val_loss, val_acc, best_acc = evaluate(
                epoch,
                net,
                testloader,
                optimizer,
                criterion,
                scheduler,
                args,
                best_acc,
                device,
            )
            if args.cos:
                scheduler.step()

            writer.writerow(
                [
                    epoch,
                    f"{train_loss:.6f}",
                    f"{train_acc:.6f}",
                    f"{val_loss:.6f}",
                    f"{val_acc:.6f}",
                ]
            )
            f.flush()

            print(
                f"Epoch {epoch}/{args.epochs} | train_loss: {train_loss:.4f} | "
                f"train_acc: {train_acc:.2f}% | val_loss: {val_loss:.4f} | val_acc: {val_acc:.2f}%"
            )

    print(f"Training log saved to {history_path}")
