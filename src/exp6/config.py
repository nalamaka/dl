from dataclasses import dataclass
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
EXP_DIR = Path(__file__).resolve().parent


CAMVID_CLASSES = [
    "Sky",
    "Building",
    "Pole",
    "Road",
    "Pavement",
    "Tree",
    "SignSymbol",
    "Fence",
    "Car",
    "Pedestrian",
    "Bicyclist",
]

CAMVID_COLORS = [
    (128, 128, 128),
    (128, 0, 0),
    (192, 192, 128),
    (128, 64, 128),
    (60, 40, 222),
    (128, 128, 0),
    (192, 128, 128),
    (64, 64, 128),
    (64, 0, 128),
    (64, 64, 0),
    (0, 128, 192),
]


@dataclass
class Config:
    data_dir: Path = ROOT_DIR / "data" / "CamVid"

    train_image_dir: str = "train"
    train_mask_dir: str = "train_labels"
    val_image_dir: str = "val"
    val_mask_dir: str = "val_labels"
    test_image_dir: str = "test"
    test_mask_dir: str = "test_labels"

    save_dir: Path = EXP_DIR / "checkpoints"
    log_dir: Path = EXP_DIR / "logs"
    pred_dir: Path = EXP_DIR / "predictions"

    class_names: tuple[str, ...] = tuple(CAMVID_CLASSES)
    class_colors: tuple[tuple[int, int, int], ...] = tuple(CAMVID_COLORS)
    num_classes: int = len(CAMVID_CLASSES)
    ignore_index: int = 255

    seed: int = 42
    device: str = "cuda"

    image_height: int = 360
    image_width: int = 480
    batch_size: int = 2
    num_workers: int = 0
    epochs: int = 100
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    use_amp: bool = True

    encoder_channels: tuple[int, ...] = (64, 128, 256, 512, 512)
    log_interval: int = 10
    scheduler: str = "cosine"
    min_lr_ratio: float = 0.1

    # Loss / optimization options for long-tail classes in CamVid.
    use_class_weights: bool = True
    class_weight_power: float = 0.22
    dice_loss_weight: float = 0.0
    ce_loss_weight: float = 1.0

    # Stronger yet still lightweight augmentations.
    min_scale: float = 0.75
    max_scale: float = 1.25
    brightness_jitter: float = 0.15
    contrast_jitter: float = 0.15
