import torch
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

# Data preprocessing
transform = transforms.Compose([
    transforms.ToTensor()
])

dataset = datasets.MNIST(root="data", train=True, transform=transform, download=True)

def visualize_samples():
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