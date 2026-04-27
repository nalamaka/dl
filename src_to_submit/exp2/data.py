import torch
import torchvision
import torchvision.transforms as transforms

from config import DATA_ROOT, NUM_WORKERS


def build_dataloader(batch_size, image_size, data_root=DATA_ROOT, num_workers=NUM_WORKERS):
    trans_train = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

    trans_valid = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2470, 0.2435, 0.2616]),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=trans_train
    )
    train_loader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=trans_valid
    )
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader
