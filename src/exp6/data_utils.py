from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from PIL import ImageEnhance
import torch
from torch.utils.data import DataLoader, Dataset

from config import Config


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class SegmentationSample:
    image_path: Path
    mask_path: Path | None


def _list_image_files(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"folder not found: {folder}")
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    ]
    return sorted(files)


def _resolve_mask_name_candidates(image_path: Path) -> list[str]:
    stem = image_path.stem
    suffix = image_path.suffix
    return [
        f"{stem}{suffix}",
        f"{stem}.png",
        f"{stem}_L{suffix}",
        f"{stem}_L.png",
        f"{stem}_label{suffix}",
        f"{stem}_label.png",
    ]


def _find_mask_path(mask_dir: Path, image_path: Path) -> Path:
    for candidate in _resolve_mask_name_candidates(image_path):
        p = mask_dir / candidate
        if p.exists():
            return p
    raise FileNotFoundError(f"mask not found for image: {image_path}")


def discover_split_samples(image_dir: Path, mask_dir: Path | None) -> list[SegmentationSample]:
    image_files = _list_image_files(image_dir)
    samples: list[SegmentationSample] = []
    for image_path in image_files:
        if mask_dir is None:
            samples.append(SegmentationSample(image_path=image_path, mask_path=None))
        else:
            samples.append(
                SegmentationSample(
                    image_path=image_path,
                    mask_path=_find_mask_path(mask_dir, image_path),
                )
            )
    return samples


def load_camvid_splits(data_dir: Path) -> dict[str, list[SegmentationSample]]:
    split_pairs = {
        "train": (data_dir / Config.train_image_dir, data_dir / Config.train_mask_dir),
        "val": (data_dir / Config.val_image_dir, data_dir / Config.val_mask_dir),
        "test": (data_dir / Config.test_image_dir, data_dir / Config.test_mask_dir),
    }
    return {
        split: discover_split_samples(image_dir=image_dir, mask_dir=mask_dir)
        for split, (image_dir, mask_dir) in split_pairs.items()
    }


def build_color_map(colors: tuple[tuple[int, int, int], ...]) -> dict[tuple[int, int, int], int]:
    return {tuple(color): idx for idx, color in enumerate(colors)}


def encode_mask(mask: Image.Image, color_map: dict[tuple[int, int, int], int], ignore_index: int) -> torch.Tensor:
    mask_np = np.array(mask)
    if mask_np.ndim == 2:
        encoded = mask_np.astype(np.int64)
        valid = (encoded >= 0) & (encoded < len(color_map))
        encoded = np.where(valid, encoded, ignore_index)
        return torch.from_numpy(encoded).long()

    if mask_np.ndim != 3 or mask_np.shape[2] < 3:
        raise ValueError(f"Unsupported mask shape: {mask_np.shape}")

    rgb = mask_np[:, :, :3]
    encoded = np.full((rgb.shape[0], rgb.shape[1]), ignore_index, dtype=np.int64)
    for color, class_idx in color_map.items():
        matches = np.all(rgb == np.array(color, dtype=np.uint8), axis=-1)
        encoded[matches] = class_idx
    return torch.from_numpy(encoded).long()


def decode_mask(mask: np.ndarray, colors: tuple[tuple[int, int, int], ...], ignore_index: int) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    for class_idx, color in enumerate(colors):
        out[mask == class_idx] = np.array(color, dtype=np.uint8)
    out[mask == ignore_index] = np.array([0, 0, 0], dtype=np.uint8)
    return out


class CamVidDataset(Dataset):
    def __init__(
        self,
        samples: list[SegmentationSample],
        image_size: tuple[int, int],
        class_colors: tuple[tuple[int, int, int], ...],
        ignore_index: int,
        augment: bool = False,
        min_scale: float = 0.75,
        max_scale: float = 1.25,
        brightness_jitter: float = 0.15,
        contrast_jitter: float = 0.15,
    ):
        self.samples = samples
        self.image_size = image_size
        self.color_map = build_color_map(class_colors)
        self.class_colors = class_colors
        self.ignore_index = ignore_index
        self.augment = augment
        self.min_scale = float(min_scale)
        self.max_scale = float(max_scale)
        self.brightness_jitter = float(max(0.0, brightness_jitter))
        self.contrast_jitter = float(max(0.0, contrast_jitter))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, image_path: Path) -> Image.Image:
        return Image.open(image_path).convert("RGB")

    def _load_mask(self, mask_path: Path) -> Image.Image:
        return Image.open(mask_path)

    def _resize_pair(self, image: Image.Image, mask: Image.Image | None) -> tuple[Image.Image, Image.Image | None]:
        target_size = (self.image_size[1], self.image_size[0])
        image = image.resize(target_size, Image.BILINEAR)
        if mask is not None:
            mask = mask.resize(target_size, Image.NEAREST)
        return image, mask

    def _random_rescale_and_crop(
        self,
        image: Image.Image,
        mask: Image.Image | None,
    ) -> tuple[Image.Image, Image.Image | None]:
        if not self.augment:
            return image, mask

        target_h, target_w = self.image_size
        scale = random.uniform(self.min_scale, self.max_scale)
        scaled_w = max(target_w, int(round(image.width * scale)))
        scaled_h = max(target_h, int(round(image.height * scale)))

        image = image.resize((scaled_w, scaled_h), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((scaled_w, scaled_h), Image.NEAREST)

        max_left = max(0, scaled_w - target_w)
        max_top = max(0, scaled_h - target_h)
        left = random.randint(0, max_left) if max_left > 0 else 0
        top = random.randint(0, max_top) if max_top > 0 else 0
        crop_box = (left, top, left + target_w, top + target_h)

        image = image.crop(crop_box)
        if mask is not None:
            mask = mask.crop(crop_box)
        return image, mask

    def _maybe_flip(self, image: Image.Image, mask: Image.Image | None) -> tuple[Image.Image, Image.Image | None]:
        if self.augment and random.random() < 0.5:
            image = image.transpose(Image.FLIP_LEFT_RIGHT)
            if mask is not None:
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
        return image, mask

    def _maybe_color_jitter(self, image: Image.Image) -> Image.Image:
        if not self.augment:
            return image

        if self.brightness_jitter > 0:
            factor = random.uniform(1.0 - self.brightness_jitter, 1.0 + self.brightness_jitter)
            image = ImageEnhance.Brightness(image).enhance(factor)
        if self.contrast_jitter > 0:
            factor = random.uniform(1.0 - self.contrast_jitter, 1.0 + self.contrast_jitter)
            image = ImageEnhance.Contrast(image).enhance(factor)
        return image

    def _image_to_tensor(self, image: Image.Image) -> torch.Tensor:
        arr = np.array(image).astype(np.float32) / 255.0
        tensor = torch.from_numpy(arr).permute(2, 0, 1)
        return (tensor - IMAGENET_MEAN) / IMAGENET_STD

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        sample = self.samples[index]
        image = self._load_image(sample.image_path)
        mask = self._load_mask(sample.mask_path) if sample.mask_path is not None else None
        image, mask = self._random_rescale_and_crop(image, mask)
        image, mask = self._maybe_flip(image, mask)
        image = self._maybe_color_jitter(image)

        if not self.augment:
            image, mask = self._resize_pair(image, mask)

        item: dict[str, torch.Tensor | str] = {
            "image": self._image_to_tensor(image),
            "image_path": str(sample.image_path),
        }
        if mask is not None:
            item["mask"] = encode_mask(mask, self.color_map, self.ignore_index)
        return item


def build_dataloader(
    samples: list[SegmentationSample],
    image_size: tuple[int, int],
    batch_size: int,
    num_workers: int,
    class_colors: tuple[tuple[int, int, int], ...],
    ignore_index: int,
    shuffle: bool,
    augment: bool,
    min_scale: float = 0.75,
    max_scale: float = 1.25,
    brightness_jitter: float = 0.15,
    contrast_jitter: float = 0.15,
) -> DataLoader:
    dataset = CamVidDataset(
        samples=samples,
        image_size=image_size,
        class_colors=class_colors,
        ignore_index=ignore_index,
        augment=augment,
        min_scale=min_scale,
        max_scale=max_scale,
        brightness_jitter=brightness_jitter,
        contrast_jitter=contrast_jitter,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def estimate_class_weights(
    samples: list[SegmentationSample],
    image_size: tuple[int, int],
    class_colors: tuple[tuple[int, int, int], ...],
    ignore_index: int,
    power: float = 0.35,
) -> torch.Tensor:
    color_map = build_color_map(class_colors)
    counts = np.zeros(len(class_colors), dtype=np.float64)
    target_size = (image_size[1], image_size[0])

    for sample in samples:
        if sample.mask_path is None:
            continue
        mask = Image.open(sample.mask_path)
        mask = mask.resize(target_size, Image.NEAREST)
        encoded = encode_mask(mask, color_map, ignore_index).numpy()
        valid = (encoded != ignore_index)
        if not np.any(valid):
            continue
        hist = np.bincount(encoded[valid].reshape(-1), minlength=len(class_colors))
        counts += hist[: len(class_colors)]

    counts = np.maximum(counts, 1.0)
    freq = counts / counts.sum()
    weights = np.power(1.0 / freq, float(power))
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)
