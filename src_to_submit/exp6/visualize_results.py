from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import torch

from config import Config
from data_utils import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_color_map,
    decode_mask,
    encode_mask,
    load_camvid_splits,
)
from engine import load_checkpoint, select_device
from main import build_model, resolve_checkpoint_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize exp6 logs and segmentation examples")
    parser.add_argument("--log_file", type=str, default="", help="Path to a training jsonl log file")
    parser.add_argument("--ckpt_path", type=str, default="", help="Checkpoint path. Defaults to best checkpoint.")
    parser.add_argument("--ckpt_name", type=str, default="segnet_camvid.pth", help="Checkpoint base name.")
    parser.add_argument("--data_dir", type=str, default=str(Config.data_dir), help="CamVid data directory.")
    parser.add_argument("--out_dir", type=str, default="", help="Output directory for figures and summary.")
    parser.add_argument("--device", type=str, default=Config.device, help="cuda or cpu")
    parser.add_argument("--image_height", type=int, default=Config.image_height)
    parser.add_argument("--image_width", type=int, default=Config.image_width)
    parser.add_argument("--num_examples", type=int, default=6, help="Number of test-set examples to visualize.")
    parser.add_argument("--show", action="store_true", help="Show matplotlib windows interactively.")
    return parser.parse_args()


def _find_latest_log() -> Path:
    log_files = sorted(Config.log_dir.glob("*.jsonl"))
    if not log_files:
        raise FileNotFoundError(f"no log files found under {Config.log_dir}")
    return log_files[-1]


def _load_records(log_file: Path) -> tuple[list[dict], int]:
    records: list[dict] = []
    skipped = 0
    with log_file.open("r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                records.append(json.loads(s))
            except json.JSONDecodeError:
                skipped += 1
    return records, skipped


def _write_cleaned_log(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _resolve_out_dir(args, log_file: Path) -> Path:
    return Path(args.out_dir) if args.out_dir else (log_file.parent / "figures" / log_file.stem)


def _resolve_checkpoint(args) -> Path:
    if args.ckpt_path:
        ckpt = Path(args.ckpt_path)
        if not ckpt.exists():
            raise FileNotFoundError(f"checkpoint not found: {ckpt}")
        return ckpt
    _, best_path, last_path = resolve_checkpoint_paths(args.ckpt_name)
    if best_path.exists():
        return best_path
    if last_path.exists():
        return last_path
    raise FileNotFoundError(f"no checkpoint found for {args.ckpt_name}")


def _save_line_plot(
    x_values: list[int],
    series: list[tuple[str, list[float], str]],
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: Path,
    show: bool,
) -> None:
    try:
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:
        raise RuntimeError("matplotlib is required for visualization. Install via: pip install matplotlib") from exc

    plt.figure(figsize=(9, 4.8))
    for label, values, color in series:
        plt.plot(x_values, values, marker="o", linewidth=1.4, label=label, color=color)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    if show:
        plt.show()
    plt.close()


def _collect_epoch_series(epoch_records: list[dict]) -> dict[str, list[float] | list[int]]:
    epochs = [int(r.get("epoch", 0)) for r in epoch_records]
    train_loss = [float(r.get("train_loss", 0.0)) for r in epoch_records]
    val_loss = [float(r.get("val_stats", {}).get("loss", 0.0)) for r in epoch_records]
    test_loss = [float(r.get("test_stats", {}).get("loss", 0.0)) for r in epoch_records]
    val_miou = [float(r.get("val_stats", {}).get("mean_iou", 0.0)) for r in epoch_records]
    test_miou = [float(r.get("test_stats", {}).get("mean_iou", 0.0)) for r in epoch_records]
    val_pa = [float(r.get("val_stats", {}).get("pixel_accuracy", 0.0)) for r in epoch_records]
    test_pa = [float(r.get("test_stats", {}).get("pixel_accuracy", 0.0)) for r in epoch_records]
    val_mpa = [float(r.get("val_stats", {}).get("mean_pixel_accuracy", 0.0)) for r in epoch_records]
    test_mpa = [float(r.get("test_stats", {}).get("mean_pixel_accuracy", 0.0)) for r in epoch_records]
    return {
        "epochs": epochs,
        "train_loss": train_loss,
        "val_loss": val_loss,
        "test_loss": test_loss,
        "val_miou": val_miou,
        "test_miou": test_miou,
        "val_pa": val_pa,
        "test_pa": test_pa,
        "val_mpa": val_mpa,
        "test_mpa": test_mpa,
    }


def _normalize_image(image: Image.Image) -> torch.Tensor:
    arr = np.array(image).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(2, 0, 1)
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


def _load_visual_triplet(sample, image_size: tuple[int, int], color_map, ignore_index: int):
    target_size = (image_size[1], image_size[0])
    image = Image.open(sample.image_path).convert("RGB").resize(target_size, Image.BILINEAR)
    mask = Image.open(sample.mask_path).resize(target_size, Image.NEAREST) if sample.mask_path is not None else None
    image_tensor = _normalize_image(image).unsqueeze(0)

    gt_mask = None
    if mask is not None:
        gt_idx = encode_mask(mask, color_map=color_map, ignore_index=ignore_index).numpy()
        gt_mask = decode_mask(gt_idx, colors=Config.class_colors, ignore_index=ignore_index)
    return image, image_tensor, gt_mask


def _compose_example_panel(image: Image.Image, gt_mask: np.ndarray, pred_mask: np.ndarray, title: str) -> Image.Image:
    image_np = np.array(image)
    gt_img = Image.fromarray(gt_mask)
    pred_img = Image.fromarray(pred_mask)

    gap = 10
    label_h = 28
    width = image_np.shape[1]
    height = image_np.shape[0]
    canvas = Image.new("RGB", (width * 3 + gap * 4, height + label_h + gap * 2), color=(255, 255, 255))

    draw = ImageDraw.Draw(canvas)
    draw.text((gap, 6), title, fill=(0, 0, 0))
    items = [
        ("Input", Image.fromarray(image_np)),
        ("GroundTruth", gt_img),
        ("Prediction", pred_img),
    ]
    for idx, (label, panel) in enumerate(items):
        x = gap + idx * (width + gap)
        y = gap + label_h
        canvas.paste(panel, (x, y))
        draw.text((x, gap + 8), label, fill=(0, 0, 0))
    return canvas


def _generate_example_panels(
    ckpt_path: Path,
    data_dir: Path,
    out_dir: Path,
    device_name: str,
    image_size: tuple[int, int],
    num_examples: int,
) -> list[Path]:
    splits = load_camvid_splits(data_dir)
    test_samples = splits["test"][: max(0, num_examples)]
    if not test_samples:
        return []

    device = select_device(device_name)
    model = build_model().to(device)
    load_checkpoint(model=model, optimizer=None, ckpt_path=ckpt_path, device=device)
    model.eval()

    color_map = build_color_map(Config.class_colors)
    saved_paths: list[Path] = []
    examples_dir = out_dir / "examples"
    examples_dir.mkdir(parents=True, exist_ok=True)

    for idx, sample in enumerate(test_samples, start=1):
        image, image_tensor, gt_mask = _load_visual_triplet(
            sample=sample,
            image_size=image_size,
            color_map=color_map,
            ignore_index=Config.ignore_index,
        )
        with torch.inference_mode():
            logits = model(image_tensor.to(device))
            pred_idx = logits.argmax(dim=1).squeeze(0).detach().cpu().numpy()
        pred_mask = decode_mask(pred_idx, colors=Config.class_colors, ignore_index=Config.ignore_index)

        title = Path(sample.image_path).name
        panel = _compose_example_panel(image, gt_mask, pred_mask, title=title)
        out_path = examples_dir / f"example_{idx:02d}_{Path(sample.image_path).stem}.png"
        panel.save(out_path)
        saved_paths.append(out_path)
    return saved_paths


def _build_summary(epoch_records: list[dict]) -> str:
    best_by_val = max(epoch_records, key=lambda r: float(r.get("val_stats", {}).get("mean_iou", -1.0)))
    final_rec = epoch_records[-1]

    best_epoch = int(best_by_val.get("epoch", 0))
    best_val = best_by_val.get("val_stats", {})
    best_test = best_by_val.get("test_stats", {})
    final_epoch = int(final_rec.get("epoch", 0))
    final_test = final_rec.get("test_stats", {})

    per_class = list(zip(Config.class_names, final_test.get("per_class_iou", [])))
    weakest = sorted(per_class, key=lambda x: float(x[1]))[:3]
    strongest = sorted(per_class, key=lambda x: float(x[1]), reverse=True)[:3]

    lines = [
        "# Exp6 Visualization Summary",
        "",
        "## Best Validation Epoch",
        f"- Epoch: {best_epoch}",
        f"- Val mIoU: {float(best_val.get('mean_iou', 0.0)):.4f}",
        f"- Val PA: {float(best_val.get('pixel_accuracy', 0.0)):.4f}",
        f"- Val MPA: {float(best_val.get('mean_pixel_accuracy', 0.0)):.4f}",
        f"- Test mIoU at best val epoch: {float(best_test.get('mean_iou', 0.0)):.4f}",
        "",
        "## Final Epoch",
        f"- Epoch: {final_epoch}",
        f"- Test mIoU: {float(final_test.get('mean_iou', 0.0)):.4f}",
        f"- Test PA: {float(final_test.get('pixel_accuracy', 0.0)):.4f}",
        f"- Test MPA: {float(final_test.get('mean_pixel_accuracy', 0.0)):.4f}",
        f"- Test fwIoU: {float(final_test.get('frequency_weighted_iou', 0.0)):.4f}",
        "",
        "## Strongest Classes",
    ]
    for name, value in strongest:
        lines.append(f"- {name}: {float(value):.4f}")
    lines.extend([
        "",
        "## Weakest Classes",
    ])
    for name, value in weakest:
        lines.append(f"- {name}: {float(value):.4f}")

    if any(name == "Pavement" and float(value) == 0.0 for name, value in weakest):
        lines.extend([
            "",
            "## Observation",
            "- `Pavement` remains the hardest class and is likely confused with `Road`.",
            "- Main scene classes such as `Sky`, `Road`, and `Building` are much more stable than thin or small objects.",
        ])
    return "\n".join(lines) + "\n"


def main() -> None:
    args = parse_args()
    log_file = Path(args.log_file) if args.log_file else _find_latest_log()
    if not log_file.exists():
        raise FileNotFoundError(f"log file not found: {log_file}")

    out_dir = _resolve_out_dir(args, log_file)
    out_dir.mkdir(parents=True, exist_ok=True)

    records, skipped = _load_records(log_file)
    epoch_records = [r for r in records if r.get("type") == "epoch"]
    if not epoch_records:
        raise RuntimeError(f"no epoch records found in {log_file}")

    cleaned_log_path = out_dir / "cleaned_log.jsonl"
    _write_cleaned_log(records, cleaned_log_path)

    series = _collect_epoch_series(epoch_records)
    epochs = series["epochs"]
    _save_line_plot(
        x_values=epochs,
        series=[
            ("train_loss", series["train_loss"], "tab:blue"),
            ("val_loss", series["val_loss"], "tab:orange"),
            ("test_loss", series["test_loss"], "tab:red"),
        ],
        title="Exp6 Loss Curves",
        xlabel="Epoch",
        ylabel="Loss",
        output_path=out_dir / "epoch_loss.png",
        show=args.show,
    )
    _save_line_plot(
        x_values=epochs,
        series=[
            ("val_mIoU", series["val_miou"], "tab:green"),
            ("test_mIoU", series["test_miou"], "tab:purple"),
        ],
        title="Exp6 mIoU Curves",
        xlabel="Epoch",
        ylabel="mIoU",
        output_path=out_dir / "miou_curve.png",
        show=args.show,
    )
    _save_line_plot(
        x_values=epochs,
        series=[
            ("val_PA", series["val_pa"], "tab:cyan"),
            ("test_PA", series["test_pa"], "tab:blue"),
            ("val_MPA", series["val_mpa"], "tab:olive"),
            ("test_MPA", series["test_mpa"], "tab:brown"),
        ],
        title="Exp6 Accuracy Curves",
        xlabel="Epoch",
        ylabel="Metric",
        output_path=out_dir / "accuracy_curve.png",
        show=args.show,
    )

    summary = _build_summary(epoch_records)
    summary_path = out_dir / "summary.md"
    summary_path.write_text(summary, encoding="utf-8")

    ckpt_path = _resolve_checkpoint(args)
    example_paths = _generate_example_panels(
        ckpt_path=ckpt_path,
        data_dir=Path(args.data_dir),
        out_dir=out_dir,
        device_name=args.device,
        image_size=(args.image_height, args.image_width),
        num_examples=args.num_examples,
    )

    print(f"[viz] log file: {log_file}")
    print(f"[viz] skipped malformed lines: {skipped}")
    print(f"[viz] cleaned log: {cleaned_log_path}")
    print(f"[viz] output dir: {out_dir}")
    print(f"[viz] summary: {summary_path}")
    print(f"[viz] checkpoint: {ckpt_path}")
    print(f"[viz] examples: {len(example_paths)}")
    for path in example_paths:
        print(f"[viz] example -> {path}")


if __name__ == "__main__":
    main()
