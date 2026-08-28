from __future__ import annotations

import argparse
import csv
import subprocess
from dataclasses import dataclass
from pathlib import Path


DEFAULT_VIDEO_NAMES = {
    "0001TP": "01TP_extract.avi",
    "0006R0": "0006R0.MXF",
    "Seq05VD": "0005VD.MXF",
    "0016E5": "0016E5.MXF",
    "0016E5_15Hz": "0016E5.MXF",
}

DEFAULT_FRAME_OFFSETS = {
    "0001TP": 6690,
    "0006R0": 930,
    "Seq05VD": 0,
    "0016E5": 390,
    "0016E5_15Hz": 8001,
}


@dataclass
class FrameTask:
    sequence: str
    label_name: str
    raw_name: str
    global_frame: int
    local_frame: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract CamVid raw frames from local video files according to label names."
    )
    parser.add_argument("--labels_dir", type=str, required=True, help="Directory containing *_L.png label files.")
    parser.add_argument("--videos_dir", type=str, default="", help="Directory containing local CamVid video files.")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for extracted raw frames.")
    parser.add_argument("--manifest_csv", type=str, default="", help="Optional CSV manifest output path.")
    parser.add_argument("--ffmpeg", type=str, default="ffmpeg", help="ffmpeg executable path.")
    parser.add_argument("--extract", action="store_true", help="Actually run ffmpeg extraction.")
    parser.add_argument(
        "--video_map",
        action="append",
        default=[],
        help="Override video mapping, e.g. 0001TP=E:/videos/01TP_extract.avi",
    )
    parser.add_argument(
        "--frame_offset",
        action="append",
        default=[],
        help="Override per-sequence starting frame number, e.g. 0001TP=6690",
    )
    return parser.parse_args()


def expected_raw_name(label_name: str) -> str:
    return label_name.replace("_L.", ".")


def parse_frame_number(label_name: str) -> int:
    stem = Path(label_name).stem
    if stem.endswith("_L"):
        stem = stem[:-2]
    suffix = stem.split("_")[-1]
    if suffix.startswith("f"):
        suffix = suffix[1:]
    return int(suffix)


def detect_sequence(label_name: str) -> str:
    stem = Path(label_name).stem
    if stem.endswith("_L"):
        stem = stem[:-2]
    frame_num = parse_frame_number(label_name)

    if stem.startswith("0016E5_"):
        if frame_num >= 8000:
            return "0016E5_15Hz"
        return "0016E5"
    if "_f" in stem:
        return stem.split("_f", 1)[0]
    return stem.rsplit("_", 1)[0]


def parse_key_value(items: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid KEY=VALUE item: {item}")
        key, value = item.split("=", 1)
        out[key.strip()] = value.strip()
    return out


def build_tasks(labels_dir: Path, offsets: dict[str, int]) -> list[FrameTask]:
    tasks: list[FrameTask] = []
    for label_path in sorted(labels_dir.glob("*_L.png")):
        label_name = label_path.name
        sequence = detect_sequence(label_name)
        global_frame = parse_frame_number(label_name)
        if sequence not in offsets:
            raise KeyError(
                f"Missing frame offset for sequence '{sequence}'. "
                f"Please pass --frame_offset {sequence}=<number>."
            )
        local_frame = global_frame - offsets[sequence]
        if local_frame < 0:
            raise ValueError(
                f"Computed negative local frame for {label_name}: "
                f"global={global_frame}, offset={offsets[sequence]}"
            )
        tasks.append(
            FrameTask(
                sequence=sequence,
                label_name=label_name,
                raw_name=expected_raw_name(label_name),
                global_frame=global_frame,
                local_frame=local_frame,
            )
        )
    return tasks


def write_manifest(tasks: list[FrameTask], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence", "label_name", "raw_name", "global_frame", "local_frame"])
        for task in tasks:
            writer.writerow(
                [task.sequence, task.label_name, task.raw_name, task.global_frame, task.local_frame]
            )


def resolve_video_paths(videos_dir: Path | None, overrides: dict[str, str]) -> dict[str, Path]:
    video_paths: dict[str, Path] = {}
    for seq, default_name in DEFAULT_VIDEO_NAMES.items():
        if seq in overrides:
            video_paths[seq] = Path(overrides[seq]).resolve()
        elif videos_dir is not None:
            video_paths[seq] = (videos_dir / default_name).resolve()
    return video_paths


def extract_task(ffmpeg_exe: str, video_path: Path, task: FrameTask, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / task.raw_name
    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{task.local_frame})",
        "-vframes",
        "1",
        str(out_path),
    ]
    subprocess.run(cmd, check=True)


def main() -> None:
    args = parse_args()
    labels_dir = Path(args.labels_dir).resolve()
    videos_dir = Path(args.videos_dir).resolve() if args.videos_dir else None
    output_dir = Path(args.output_dir).resolve()

    if not labels_dir.exists():
        raise FileNotFoundError(f"labels_dir not found: {labels_dir}")

    offset_map = {k: int(v) for k, v in DEFAULT_FRAME_OFFSETS.items()}
    for k, v in parse_key_value(args.frame_offset).items():
        offset_map[k] = int(v)

    tasks = build_tasks(labels_dir=labels_dir, offsets=offset_map)

    if args.manifest_csv:
        write_manifest(tasks, Path(args.manifest_csv).resolve())
        print(f"[manifest] saved -> {Path(args.manifest_csv).resolve()}")

    print(f"[tasks] total={len(tasks)}")
    seq_counts: dict[str, int] = {}
    for task in tasks:
        seq_counts[task.sequence] = seq_counts.get(task.sequence, 0) + 1
    for seq, count in sorted(seq_counts.items()):
        print(f"[tasks] {seq}: {count}")

    for task in tasks[:12]:
        print(
            f"[sample] {task.label_name} -> {task.raw_name} "
            f"(sequence={task.sequence}, global={task.global_frame}, local={task.local_frame})"
        )

    if not args.extract:
        print("[extract] dry-run only. Add --extract to run ffmpeg.")
        return

    if videos_dir is None and not args.video_map:
        raise ValueError("Please provide --videos_dir or --video_map when using --extract.")

    video_paths = resolve_video_paths(videos_dir=videos_dir, overrides=parse_key_value(args.video_map))

    missing = sorted({task.sequence for task in tasks if task.sequence not in video_paths or not video_paths[task.sequence].exists()})
    if missing:
        raise FileNotFoundError(
            "Missing local video files for sequences: "
            + ", ".join(missing)
            + ". Please place them under --videos_dir or override with --video_map."
        )

    for task in tasks:
        extract_task(
            ffmpeg_exe=args.ffmpeg,
            video_path=video_paths[task.sequence],
            task=task,
            output_dir=output_dir,
        )
    print(f"[done] extracted raw frames -> {output_dir}")


if __name__ == "__main__":
    main()
