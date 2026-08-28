from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import torch

from config import Config
from data_utils import build_dataloader, estimate_class_weights, load_camvid_splits
from engine import (
    append_jsonl,
    build_scheduler,
    evaluate,
    load_checkpoint,
    predict_and_save,
    save_checkpoint,
    select_device,
    SegmentationLoss,
    set_seed,
    train_one_epoch,
)
from model import SegNet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp6: SegNet-based street scene segmentation on CamVid")
    parser.add_argument("--mode", choices=["train", "eval", "predict"], required=True)
    parser.add_argument("--data_dir", type=str, default=str(Config.data_dir))
    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--num_workers", type=int, default=Config.num_workers)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    parser.add_argument("--grad_clip", type=float, default=Config.grad_clip)
    parser.add_argument("--image_height", type=int, default=Config.image_height)
    parser.add_argument("--image_width", type=int, default=Config.image_width)
    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--log_interval", type=int, default=Config.log_interval)
    parser.add_argument("--scheduler", choices=["none", "cosine"], default=Config.scheduler)
    parser.add_argument("--min_lr_ratio", type=float, default=Config.min_lr_ratio)
    parser.add_argument("--use_amp", action="store_true", default=Config.use_amp)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--ckpt_name", type=str, default="segnet_camvid.pth")
    parser.add_argument("--pred_dir", type=str, default=str(Config.pred_dir))
    return parser.parse_args()


def build_model() -> SegNet:
    return SegNet(num_classes=Config.num_classes, encoder_channels=Config.encoder_channels)


def build_loaders(args, for_predict: bool = False):
    splits = load_camvid_splits(Path(args.data_dir))
    image_size = (args.image_height, args.image_width)

    train_loader = build_dataloader(
        samples=splits["train"],
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        class_colors=Config.class_colors,
        ignore_index=Config.ignore_index,
        shuffle=True,
        augment=True,
        min_scale=Config.min_scale,
        max_scale=Config.max_scale,
        brightness_jitter=Config.brightness_jitter,
        contrast_jitter=Config.contrast_jitter,
    )
    val_loader = build_dataloader(
        samples=splits["val"],
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        class_colors=Config.class_colors,
        ignore_index=Config.ignore_index,
        shuffle=False,
        augment=False,
        min_scale=Config.min_scale,
        max_scale=Config.max_scale,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )
    test_loader = build_dataloader(
        samples=splits["test"],
        image_size=image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        class_colors=Config.class_colors,
        ignore_index=Config.ignore_index,
        shuffle=False,
        augment=False,
        min_scale=Config.min_scale,
        max_scale=Config.max_scale,
        brightness_jitter=0.0,
        contrast_jitter=0.0,
    )

    if for_predict:
        return test_loader
    return train_loader, val_loader, test_loader


def resolve_checkpoint_paths(ckpt_name: str) -> tuple[Path, Path, Path]:
    base_path = Config.save_dir / ckpt_name
    stem = Path(ckpt_name).stem
    suffix = Path(ckpt_name).suffix or ".pth"
    best_path = Config.save_dir / f"{stem}.best{suffix}"
    last_path = Config.save_dir / f"{stem}.last{suffix}"
    return base_path, best_path, last_path


def run_train(args) -> None:
    set_seed(args.seed)
    if args.no_amp:
        args.use_amp = False

    device = select_device(args.device)
    if device != "cuda":
        args.use_amp = False

    train_loader, val_loader, test_loader = build_loaders(args)
    model = build_model().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        epochs=args.epochs,
        min_lr_ratio=args.min_lr_ratio,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    class_weights = None
    if Config.use_class_weights:
        class_weights = estimate_class_weights(
            samples=train_loader.dataset.samples,
            image_size=(args.image_height, args.image_width),
            class_colors=Config.class_colors,
            ignore_index=Config.ignore_index,
            power=Config.class_weight_power,
        )
        print(f"[loss] class_weights={class_weights.tolist()}")
    criterion = SegmentationLoss(
        num_classes=Config.num_classes,
        ignore_index=Config.ignore_index,
        class_weights=(class_weights.to(device) if class_weights is not None else None),
        ce_weight=Config.ce_loss_weight,
        dice_weight=Config.dice_loss_weight,
    )

    ckpt_path, best_ckpt_path, last_ckpt_path = resolve_checkpoint_paths(args.ckpt_name)

    start_epoch = 1
    best_miou = 0.0
    if args.resume:
        resume_path = Path(args.resume_ckpt) if args.resume_ckpt else last_ckpt_path
        if resume_path.exists():
            epoch_done, best_miou = load_checkpoint(
                model=model,
                optimizer=optimizer,
                ckpt_path=resume_path,
                device=device,
                scheduler=scheduler,
                scaler=scaler,
            )
            start_epoch = epoch_done + 1
            print(f"resume from {resume_path}, start_epoch={start_epoch}, best_mIoU={best_miou:.4f}")
        else:
            print(f"[resume] checkpoint not found: {resume_path}, training from scratch.")

    run_name = f"{Path(args.ckpt_name).stem}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = Config.log_dir / f"{run_name}.jsonl"
    append_jsonl(
        log_path,
        {
            "type": "meta",
            "run_name": run_name,
            "device": device,
            "epochs": int(args.epochs),
            "batch_size": int(args.batch_size),
            "lr": float(args.lr),
            "image_size": [int(args.image_height), int(args.image_width)],
            "use_class_weights": bool(Config.use_class_weights),
            "class_weight_power": float(Config.class_weight_power),
            "dice_loss_weight": float(Config.dice_loss_weight),
        },
    )
    print(f"[data] train={len(train_loader.dataset)} val={len(val_loader.dataset)} test={len(test_loader.dataset)}")
    print(f"[log] writing train logs -> {log_path}")

    for epoch in range(start_epoch, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            criterion=criterion,
            grad_clip=args.grad_clip,
            use_amp=args.use_amp,
            scaler=scaler,
            log_interval=args.log_interval,
        )
        val_stats = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            criterion=criterion,
            num_classes=Config.num_classes,
            ignore_index=Config.ignore_index,
        )
        test_stats = evaluate(
            model=model,
            dataloader=test_loader,
            device=device,
            criterion=criterion,
            num_classes=Config.num_classes,
            ignore_index=Config.ignore_index,
        )

        if scheduler is not None:
            scheduler.step()

        val_miou = float(val_stats["mean_iou"])
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={float(val_stats['loss']):.4f} "
            f"val_PA={float(val_stats['pixel_accuracy']):.4f} "
            f"val_MPA={float(val_stats['mean_pixel_accuracy']):.4f} "
            f"val_mIoU={val_miou:.4f}"
        )
        print(
            f"[epoch {epoch}] test_PA={float(test_stats['pixel_accuracy']):.4f} "
            f"test_MPA={float(test_stats['mean_pixel_accuracy']):.4f} "
            f"test_mIoU={float(test_stats['mean_iou']):.4f}"
        )

        append_jsonl(
            log_path,
            {
                "type": "epoch",
                "epoch": int(epoch),
                "train_loss": float(train_loss),
                "val_stats": val_stats,
                "test_stats": test_stats,
            },
        )

        if val_miou >= best_miou:
            best_miou = val_miou
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_miou=best_miou,
                ckpt_path=best_ckpt_path,
                scheduler=scheduler,
                scaler=scaler,
            )
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_miou=best_miou,
                ckpt_path=ckpt_path,
                scheduler=scheduler,
                scaler=scaler,
            )
            print(f"saved best checkpoint -> {best_ckpt_path}")

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            best_miou=best_miou,
            ckpt_path=last_ckpt_path,
            scheduler=scheduler,
            scaler=scaler,
        )
        print(f"saved last checkpoint -> {last_ckpt_path}")


def _load_eval_model(args):
    set_seed(args.seed)
    if args.no_amp:
        args.use_amp = False
    device = select_device(args.device)

    model = build_model().to(device)
    _, best_ckpt_path, last_ckpt_path = resolve_checkpoint_paths(args.ckpt_name)
    ckpt_path = Path(args.resume_ckpt) if args.resume_ckpt else best_ckpt_path
    if not ckpt_path.exists():
        if last_ckpt_path.exists():
            ckpt_path = last_ckpt_path
        else:
            raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    load_checkpoint(model=model, optimizer=None, ckpt_path=ckpt_path, device=device)
    print(f"loaded checkpoint -> {ckpt_path}")
    return model, device


def run_eval(args) -> None:
    model, device = _load_eval_model(args)
    _, val_loader, test_loader = build_loaders(args)
    criterion = SegmentationLoss(
        num_classes=Config.num_classes,
        ignore_index=Config.ignore_index,
        class_weights=None,
        ce_weight=1.0,
        dice_weight=0.0,
    )

    val_stats = evaluate(
        model=model,
        dataloader=val_loader,
        device=device,
        criterion=criterion,
        num_classes=Config.num_classes,
        ignore_index=Config.ignore_index,
    )
    test_stats = evaluate(
        model=model,
        dataloader=test_loader,
        device=device,
        criterion=criterion,
        num_classes=Config.num_classes,
        ignore_index=Config.ignore_index,
    )

    print(
        f"[val] loss={float(val_stats['loss']):.4f} "
        f"PA={float(val_stats['pixel_accuracy']):.4f} "
        f"MPA={float(val_stats['mean_pixel_accuracy']):.4f} "
        f"mIoU={float(val_stats['mean_iou']):.4f} "
        f"fwIoU={float(val_stats['frequency_weighted_iou']):.4f}"
    )
    print(
        f"[test] loss={float(test_stats['loss']):.4f} "
        f"PA={float(test_stats['pixel_accuracy']):.4f} "
        f"MPA={float(test_stats['mean_pixel_accuracy']):.4f} "
        f"mIoU={float(test_stats['mean_iou']):.4f} "
        f"fwIoU={float(test_stats['frequency_weighted_iou']):.4f}"
    )

    for class_name, class_iou in zip(Config.class_names, test_stats["per_class_iou"]):
        print(f"[test][IoU] {class_name}: {float(class_iou):.4f}")


def run_predict(args) -> None:
    model, device = _load_eval_model(args)
    test_loader = build_loaders(args, for_predict=True)
    predict_and_save(
        model=model,
        dataloader=test_loader,
        device=device,
        colors=Config.class_colors,
        ignore_index=Config.ignore_index,
        output_dir=Path(args.pred_dir),
    )
    print(f"saved prediction masks -> {Path(args.pred_dir)}")


def main():
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    else:
        run_predict(args)


if __name__ == "__main__":
    main()
