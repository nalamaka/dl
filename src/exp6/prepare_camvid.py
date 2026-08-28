from __future__ import annotations

import argparse
import shutil
from dataclasses import dataclass
from pathlib import Path
import tarfile
import urllib.request
import zipfile

from config import Config


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
MASK_CANDIDATE_SUFFIXES = [
    "",
    "_L",
    "_label",
]
DEFAULT_STILLS_URL = "http://www.inf.ethz.ch/personal/gbrostow/temp/701_StillsRaw_full.zip"
DEFAULT_LABELS_URL = "http://mi.eng.cam.ac.uk/research/projects/VideoRec/CamVid/data/LabeledApproved_full.zip"


@dataclass
class SplitLayout:
    train_images: Path
    train_masks: Path
    val_images: Path
    val_masks: Path
    test_images: Path
    test_masks: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deploy CamVid data into exp6 expected layout."
    )
    parser.add_argument(
        "--raw_dir",
        type=str,
        default="",
        help="Raw CamVid root directory. If omitted with --download, data will be downloaded automatically.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(Config.data_dir),
        help="Deployment output directory. Example: E:/hw/deep_learning/data/CamVid",
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "split_dirs", "flat_with_txt"],
        default="auto",
        help="Detection mode.",
    )
    parser.add_argument(
        "--images_dir",
        type=str,
        default="",
        help="Flat-image directory name or path relative to raw_dir for flat_with_txt mode.",
    )
    parser.add_argument(
        "--labels_dir",
        type=str,
        default="",
        help="Flat-label directory name or path relative to raw_dir for flat_with_txt mode.",
    )
    parser.add_argument(
        "--train_list",
        type=str,
        default="",
        help="Train split txt path relative to raw_dir for flat_with_txt mode.",
    )
    parser.add_argument(
        "--val_list",
        type=str,
        default="",
        help="Val split txt path relative to raw_dir for flat_with_txt mode.",
    )
    parser.add_argument(
        "--test_list",
        type=str,
        default="",
        help="Test split txt path relative to raw_dir for flat_with_txt mode.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files in output directory.",
    )
    parser.add_argument("--download", action="store_true", help="Download official CamVid stills and labels first.")
    parser.add_argument(
        "--extract_dir",
        type=str,
        default="",
        help="Optional extraction directory. Defaults to <output_dir>/_raw_download.",
    )
    parser.add_argument(
        "--download_only",
        action="store_true",
        help="Only download and extract, do not deploy into train/val/test layout.",
    )
    return parser.parse_args()


def _resolve_under(root: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return root / path


def _find_existing_dir(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.exists() and path.is_dir():
            return path
    return None


def _find_existing_file(root: Path, candidates: list[str]) -> Path | None:
    for name in candidates:
        path = root / name
        if path.exists() and path.is_file():
            return path
    return None


def detect_split_layout(raw_dir: Path) -> SplitLayout | None:
    train_images = _find_existing_dir(raw_dir, ["train"])
    val_images = _find_existing_dir(raw_dir, ["val", "valid", "validation"])
    test_images = _find_existing_dir(raw_dir, ["test"])
    train_masks = _find_existing_dir(raw_dir, ["train_labels", "trainannot", "train_labels_with_ignored"])
    val_masks = _find_existing_dir(raw_dir, ["val_labels", "valannot", "valid_labels", "validation_labels"])
    test_masks = _find_existing_dir(raw_dir, ["test_labels", "testannot"])

    if all([train_images, val_images, test_images, train_masks, val_masks, test_masks]):
        return SplitLayout(
            train_images=train_images,
            train_masks=train_masks,
            val_images=val_images,
            val_masks=val_masks,
            test_images=test_images,
            test_masks=test_masks,
        )
    return None


def detect_flat_layout(raw_dir: Path, args: argparse.Namespace) -> tuple[Path, Path, Path, Path, Path] | None:
    if args.images_dir:
        images_dir = _resolve_under(raw_dir, args.images_dir)
    else:
        images_dir = _find_existing_dir(raw_dir, ["701_StillsRaw_full", "images", "imgs"])

    if args.labels_dir:
        labels_dir = _resolve_under(raw_dir, args.labels_dir)
    else:
        labels_dir = _find_existing_dir(raw_dir, ["LabeledApproved_full", "labels", "label", "masks", "annot"])

    if args.train_list:
        train_list = _resolve_under(raw_dir, args.train_list)
    else:
        train_list = _find_existing_file(raw_dir, ["train.txt", "train.lst", "train_files.txt"])

    if args.val_list:
        val_list = _resolve_under(raw_dir, args.val_list)
    else:
        val_list = _find_existing_file(raw_dir, ["val.txt", "valid.txt", "validation.txt", "val.lst"])

    if args.test_list:
        test_list = _resolve_under(raw_dir, args.test_list)
    else:
        test_list = _find_existing_file(raw_dir, ["test.txt", "test.lst", "test_files.txt"])

    if all([images_dir, labels_dir, train_list, val_list, test_list]):
        return images_dir, labels_dir, train_list, val_list, test_list
    return None


def _iter_image_files(folder: Path) -> list[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS]
    )


def _resolve_mask_path(labels_dir: Path, image_name: str) -> Path:
    image_path = Path(image_name)
    stem = image_path.stem
    suffix = image_path.suffix or ".png"

    candidates = []
    for extra in MASK_CANDIDATE_SUFFIXES:
        candidates.append(labels_dir / f"{stem}{extra}{suffix}")
        candidates.append(labels_dir / f"{stem}{extra}.png")

    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Mask not found for image '{image_name}' in {labels_dir}")


def _ensure_clean_file_copy(src: Path, dst: Path, overwrite: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        return
    shutil.copy2(src, dst)


def _infer_archive_name_from_url(url: str) -> str:
    name = Path(url.split("?")[0]).name
    return name or "camvid_download.zip"


def _download_file(url: str, dst: Path, overwrite: bool) -> Path:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() and not overwrite:
        print(f"[download] skip existing archive: {dst}")
        return dst

    print(f"[download] downloading: {url}")
    print(f"[download] saving to: {dst}")
    with urllib.request.urlopen(url) as response, dst.open("wb") as f:
        shutil.copyfileobj(response, f)
    return dst


def _read_file_prefix(path: Path, limit: int = 256) -> bytes:
    with path.open("rb") as f:
        return f.read(limit)


def _looks_like_html(data: bytes) -> bool:
    prefix = data.lstrip().lower()
    return prefix.startswith(b"<!doctype html") or prefix.startswith(b"<html") or prefix.startswith(b"<?xml")


def _validate_archive_file(archive_path: Path) -> None:
    if not archive_path.exists():
        raise FileNotFoundError(f"archive not found: {archive_path}")

    prefix = _read_file_prefix(archive_path, limit=512)
    suffixes = "".join(archive_path.suffixes).lower()

    if _looks_like_html(prefix):
        preview = prefix[:160].decode("utf-8", errors="ignore").replace("\n", " ")
        raise RuntimeError(
            "Downloaded file is HTML instead of an archive. "
            f"This usually means the source link has expired or returned an error page.\n"
            f"archive={archive_path}\n"
            f"preview={preview}"
        )

    if suffixes.endswith(".zip") and not zipfile.is_zipfile(archive_path):
        raise RuntimeError(
            f"Downloaded file is not a valid zip archive: {archive_path}. "
            "Please check whether the upstream link is still valid."
        )

    if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        if not tarfile.is_tarfile(archive_path):
            raise RuntimeError(
                f"Downloaded file is not a valid tar archive: {archive_path}. "
                "Please check whether the upstream link is still valid."
            )


def _extract_archive(archive_path: Path, extract_dir: Path, overwrite: bool) -> Path:
    _validate_archive_file(archive_path)
    if extract_dir.exists() and overwrite:
        shutil.rmtree(extract_dir)
    extract_dir.mkdir(parents=True, exist_ok=True)

    suffixes = "".join(archive_path.suffixes).lower()
    print(f"[extract] archive={archive_path}")
    print(f"[extract] target={extract_dir}")

    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "r") as zf:
            zf.extractall(extract_dir)
        return extract_dir

    if suffixes.endswith(".tar") or suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        with tarfile.open(archive_path, "r:*") as tf:
            tf.extractall(extract_dir)
        return extract_dir

    raise ValueError(f"Unsupported archive format: {archive_path}")


def _pick_single_child_dir(root: Path) -> Path:
    children = [p for p in root.iterdir() if p.is_dir()]
    if len(children) == 1:
        return children[0]
    return root


def _copy_downloaded_flat_layout(
    stills_dir: Path,
    labels_dir: Path,
    merged_raw_dir: Path,
    overwrite: bool,
) -> Path:
    merged_raw_dir.mkdir(parents=True, exist_ok=True)
    merged_images_dir = merged_raw_dir / "701_StillsRaw_full"
    merged_labels_dir = merged_raw_dir / "LabeledApproved_full"

    for src_dir, dst_dir in (
        (stills_dir, merged_images_dir),
        (labels_dir, merged_labels_dir),
    ):
        dst_dir.mkdir(parents=True, exist_ok=True)
        for src_path in _iter_image_files(src_dir):
            _ensure_clean_file_copy(src_path, dst_dir / src_path.name, overwrite=overwrite)
        print(f"[download] merged {src_dir} -> {dst_dir}")
    return merged_raw_dir


def prepare_raw_dir(args: argparse.Namespace, output_dir: Path) -> Path:
    raw_dir = Path(args.raw_dir).resolve() if args.raw_dir else Path()
    if args.download:
        extract_dir = (
            Path(args.extract_dir).resolve()
            if args.extract_dir
            else (output_dir / "_raw_download").resolve()
        )
        archives_dir = output_dir / "_archives"
        stills_archive = archives_dir / _infer_archive_name_from_url(DEFAULT_STILLS_URL)
        labels_archive = archives_dir / _infer_archive_name_from_url(DEFAULT_LABELS_URL)

        _download_file(DEFAULT_STILLS_URL, stills_archive, overwrite=args.overwrite)
        _download_file(DEFAULT_LABELS_URL, labels_archive, overwrite=args.overwrite)

        stills_extract_dir = extract_dir / "stills"
        labels_extract_dir = extract_dir / "labels"
        _extract_archive(stills_archive, stills_extract_dir, overwrite=args.overwrite)
        _extract_archive(labels_archive, labels_extract_dir, overwrite=args.overwrite)

        stills_dir = _pick_single_child_dir(stills_extract_dir)
        labels_dir = _pick_single_child_dir(labels_extract_dir)
        raw_dir = _copy_downloaded_flat_layout(
            stills_dir=stills_dir,
            labels_dir=labels_dir,
            merged_raw_dir=extract_dir / "merged_raw",
            overwrite=args.overwrite,
        )
        print(f"[download] prepared raw_dir: {raw_dir}")

    if not raw_dir or not raw_dir.exists():
        raise FileNotFoundError(f"raw_dir not found: {raw_dir}")
    return raw_dir


def deploy_pre_split(raw_dir: Path, output_dir: Path, overwrite: bool) -> None:
    layout = detect_split_layout(raw_dir)
    if layout is None:
        raise FileNotFoundError("Could not detect pre-split CamVid directories under raw_dir.")

    mapping = [
        (layout.train_images, output_dir / "train"),
        (layout.train_masks, output_dir / "train_labels"),
        (layout.val_images, output_dir / "val"),
        (layout.val_masks, output_dir / "val_labels"),
        (layout.test_images, output_dir / "test"),
        (layout.test_masks, output_dir / "test_labels"),
    ]

    for src_dir, dst_dir in mapping:
        files = _iter_image_files(src_dir)
        for src_path in files:
            _ensure_clean_file_copy(src_path, dst_dir / src_path.name, overwrite=overwrite)
        print(f"[deploy] {src_dir} -> {dst_dir} ({len(files)} files)")


def _read_split_list(path: Path) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    names = []
    for line in lines:
        value = line.strip().replace("\\", "/")
        if not value:
            continue
        names.append(Path(value).name)
    return names


def _deploy_named_split(
    image_names: list[str],
    images_dir: Path,
    labels_dir: Path,
    out_image_dir: Path,
    out_mask_dir: Path,
    overwrite: bool,
) -> int:
    count = 0
    for image_name in image_names:
        image_src = images_dir / image_name
        if not image_src.exists():
            raise FileNotFoundError(f"Image not found: {image_src}")
        mask_src = _resolve_mask_path(labels_dir, image_name)
        _ensure_clean_file_copy(image_src, out_image_dir / image_src.name, overwrite=overwrite)
        _ensure_clean_file_copy(mask_src, out_mask_dir / image_src.name, overwrite=overwrite)
        count += 1
    return count


def deploy_flat_with_txt(raw_dir: Path, output_dir: Path, args: argparse.Namespace) -> None:
    layout = detect_flat_layout(raw_dir, args)
    if layout is None:
        raise FileNotFoundError(
            "Could not detect flat CamVid layout with split txt files. "
            "Please pass --images_dir --labels_dir --train_list --val_list --test_list explicitly."
        )

    images_dir, labels_dir, train_list, val_list, test_list = layout
    split_specs = [
        ("train", train_list, output_dir / "train", output_dir / "train_labels"),
        ("val", val_list, output_dir / "val", output_dir / "val_labels"),
        ("test", test_list, output_dir / "test", output_dir / "test_labels"),
    ]

    for split_name, list_path, out_image_dir, out_mask_dir in split_specs:
        names = _read_split_list(list_path)
        count = _deploy_named_split(
            image_names=names,
            images_dir=images_dir,
            labels_dir=labels_dir,
            out_image_dir=out_image_dir,
            out_mask_dir=out_mask_dir,
            overwrite=args.overwrite,
        )
        print(f"[deploy] {split_name}: {count} samples")


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = prepare_raw_dir(args, output_dir)

    if args.download_only:
        print(f"[done] CamVid downloaded/extracted to: {raw_dir}")
        return

    if args.mode == "split_dirs":
        deploy_pre_split(raw_dir=raw_dir, output_dir=output_dir, overwrite=args.overwrite)
    elif args.mode == "flat_with_txt":
        deploy_flat_with_txt(raw_dir=raw_dir, output_dir=output_dir, args=args)
    else:
        split_layout = detect_split_layout(raw_dir)
        if split_layout is not None:
            deploy_pre_split(raw_dir=raw_dir, output_dir=output_dir, overwrite=args.overwrite)
        else:
            deploy_flat_with_txt(raw_dir=raw_dir, output_dir=output_dir, args=args)

    print(f"[done] CamVid deployed to: {output_dir}")


if __name__ == "__main__":
    main()
