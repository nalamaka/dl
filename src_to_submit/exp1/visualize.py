import torch
import matplotlib.pyplot as plt

from data import build_mnist_train_dataset_for_visualization


def visualize_samples():
    dataset = build_mnist_train_dataset_for_visualization(data_root="data")
    figure = plt.figure(figsize=(8, 8))
    cols, rows = 3, 3

    for i in range(1, cols * rows + 1):
        sample_idx = torch.randint(len(dataset), size=(1,)).item()
        img, label = dataset[sample_idx]
        figure.add_subplot(rows, cols, i)
        plt.title(label)
        plt.axis("off")
        plt.imshow(img.squeeze(), cmap="gray")

    plt.show()


if __name__ == "__main__":
    visualize_samples()
