from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import zipfile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect CamVid label zip and generate expected raw-image mapping samples.")
    parser.add_argument("--zip_path", type=str, required=True)
    parser.add_argument("--sample_count", type=int, default=20)
    parser.add_argument("--output_file", type=str, default="")
    return parser.parse_args()


def expected_image_name(label_name: str) -> str:
    if "_L." in label_name:
        return label_name.replace("_L.", ".")
    return label_name


def sequence_key(label_name: str) -> str:
    stem = Path(label_name).stem
    if stem.endswith("_L"):
        stem = stem[:-2]
    if "_f" in stem:
        return stem.split("_f", 1)[0]
    if "_" in stem:
        return stem.rsplit("_", 1)[0]
    return stem


def main() -> None:
    args = parse_args()
    zip_path = Path(args.zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = sorted([n for n in zf.namelist() if not n.endswith("/")])

    seq_counts = Counter(sequence_key(name) for name in names)
    lines: list[str] = []
    lines.append(f"zip={zip_path}")
    lines.append(f"total_labels={len(names)}")
    lines.append("sequence_counts:")
    for seq_name, count in seq_counts.most_common():
        lines.append(f"  {seq_name}: {count}")
    lines.append("sample_pairs:")
    for label_name in names[: max(1, args.sample_count)]:
        lines.append(f"  {label_name} -> {expected_image_name(label_name)}")

    text = "\n".join(lines)
    print(text)

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(text, encoding="utf-8")
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
