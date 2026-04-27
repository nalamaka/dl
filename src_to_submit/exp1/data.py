from torch.utils.data import DataLoader
from torchvision import datasets, transforms


from config import BATCH_SIZE, DATA_ROOT


def build_transform(with_normalize=True):
    transform_steps = [transforms.ToTensor()]
    if with_normalize:
        transform_steps.append(transforms.Normalize((0.5,), (0.5,)))
    return transforms.Compose(transform_steps)


def build_mnist_loaders(batch_size=BATCH_SIZE, data_root=DATA_ROOT):
    transform = build_transform(with_normalize=True)

    train_data = datasets.MNIST(
        root=data_root,
        train=True,
        transform=transform,
        download=True,
    )
    test_data = datasets.MNIST(
        root=data_root,
        train=False,
        transform=transform,
        download=True,
    )

    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


def build_mnist_train_dataset_for_visualization(data_root="data"):
    return datasets.MNIST(
        root=data_root,
        train=True,
        transform=build_transform(with_normalize=False),
        download=True,
    )
