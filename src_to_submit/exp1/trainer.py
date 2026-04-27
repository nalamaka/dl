import csv
import os

import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    EPOCHS,
    LABEL_SMOOTHING,
    LOSS_FN,
    LR,
    MOMENTUM,
    OPTIMIZER,
    WEIGHT_DECAY,
)


def evaluate(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            output = model(x)
            pred_y = torch.max(output, 1)[1]

            correct += (pred_y == y).sum().item()
            total += y.size(0)

    return correct / total if total > 0 else 0.0


def build_optimizer(model, lr):
    name = OPTIMIZER.lower()
    if name == "adam":
        return optim.Adam(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    if name == "sgd":
        return optim.SGD(
            model.parameters(), lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY
        )
    raise ValueError(f"Unsupported optimizer: {OPTIMIZER}. Use 'adam' or 'sgd'.")


def build_loss():
    name = LOSS_FN.lower()
    if name == "ce":
        return nn.CrossEntropyLoss()
    if name == "ce_ls":
        return nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    raise ValueError(f"Unsupported loss: {LOSS_FN}. Use 'ce' or 'ce_ls'.")


def train_model(model, train_loader, test_loader, device, lr=LR, epochs=EPOCHS):
    optimizer = build_optimizer(model, lr)
    loss_func = build_loss()

    os.makedirs("log", exist_ok=True)
    os.makedirs("checkpoint", exist_ok=True)

    log_path = os.path.join("log", "train_log.csv")
    history = []
    best_acc = 0.0

    with open(log_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "step", "loss", "train_accuracy", "test_accuracy"])

        for epoch in range(epochs):
            model.train()
            train_correct = 0
            train_total = 0
            for step, (x, y) in enumerate(train_loader):
                x = x.to(device)
                y = y.to(device)

                output = model(x)
                loss = loss_func(output, y)
                pred = torch.argmax(output, dim=1)
                train_correct += (pred == y).sum().item()
                train_total += y.size(0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if step % 50 == 0:
                    train_accuracy = train_correct / train_total if train_total > 0 else 0.0
                    test_accuracy = evaluate(model, test_loader, device)
                    print(
                        f"Epoch: {epoch}, Step: {step}, "
                        f"Loss: {loss.item():.4f}, "
                        f"TrainAcc: {train_accuracy:.4f}, TestAcc: {test_accuracy:.4f}"
                    )

                    writer.writerow(
                        [
                            epoch,
                            step,
                            f"{loss.item():.6f}",
                            f"{train_accuracy:.6f}",
                            f"{test_accuracy:.6f}",
                        ]
                    )
                    f.flush()
                    history.append(
                        {
                            "epoch": epoch,
                            "step": step,
                            "loss": float(loss.item()),
                            "train_accuracy": float(train_accuracy),
                            "test_accuracy": float(test_accuracy),
                        }
                    )

                    if test_accuracy > best_acc:
                        best_acc = test_accuracy
                        torch.save(model.state_dict(), os.path.join("checkpoint", "best_cnn.pth"))

                    model.train()

    return history
