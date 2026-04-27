import torch

from data import build_mnist_loaders
from models import CNN
from trainer import train_model


if __name__ == "__main__":
    if torch.cuda.is_available():
        print("GPU is available. Using CUDA.")
        device = torch.device("cuda")
    else:
        print("GPU is not available. Using CPU.")
        device = torch.device("cpu")

    train_loader, test_loader = build_mnist_loaders()
    cnn = CNN().to(device)
    history = train_model(cnn, train_loader, test_loader, device)
    if history:
        last = history[-1]
        acc = last.get("test_accuracy", last.get("accuracy"))
        if acc is None:
            acc = float("nan")
        train_acc = last.get("train_accuracy")
        train_acc_text = f", train_acc={train_acc:.4f}" if train_acc is not None else ""
        print(
            f"Training finished. Last record -> epoch={last['epoch']}, "
            f"step={last['step']}, loss={last['loss']:.4f}, test_acc={acc:.4f}{train_acc_text}"
        )
