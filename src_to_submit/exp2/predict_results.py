import glob
import os

import torch
import torchvision
import torchvision.transforms as transforms

from config import DATA_ROOT, NUM_WORKERS
from main import build_model, parse_args


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
OUTPUT_FILE = os.path.join("visualizations", "predictions.pt")


def get_latest_checkpoint():
    candidates = glob.glob(os.path.join("checkpoint", "*-ckpt.t7"))
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def build_test_loader(args):
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
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=args.batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    return testloader


def main():
    ckpt_file = get_latest_checkpoint()
    if ckpt_file is None:
        raise FileNotFoundError("checkpoint/*.t7 not found. Run training first: python main.py")

    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, device)

    state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state["net"])
    model.eval()

    testloader = build_test_loader(args)
    labels, preds = [], []
    with torch.no_grad():
        for x, y in testloader:
            x = x.to(device)
            logits = model(x)
            p = torch.argmax(logits, dim=1).cpu().tolist()
            preds.extend(p)
            labels.extend(y.tolist())

    total = len(labels)
    correct = sum(int(y == p) for y, p in zip(labels, preds))
    acc = 100.0 * correct / max(total, 1)

    os.makedirs("visualizations", exist_ok=True)
    torch.save(
        {
            "labels": labels,
            "preds": preds,
            "checkpoint": ckpt_file,
            "image_size": args.image_size,
            "patch": args.patch,
        },
        OUTPUT_FILE,
    )

    print(f"Saved: {OUTPUT_FILE}")
    print(f"Total samples: {total}")
    print(f"Correct: {correct}")
    print(f"Accuracy: {acc:.4f}%")


if __name__ == "__main__":
    main()
