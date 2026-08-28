from __future__ import annotations

import argparse
import itertools
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path


BLEU_PATTERN = re.compile(r"Test BLEU4 .*: ([0-9]*\.?[0-9]+) on (\d+) samples")


@dataclass
class DecodeConfig:
    decode_strategy: str
    beam_size: int
    length_penalty: float
    no_repeat_ngram_size: int


@dataclass
class DecodeResult:
    config: DecodeConfig
    bleu4: float
    samples: int
    ok: bool
    command: list[str]
    raw_output: str


def _parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> list[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def _build_configs(args) -> list[DecodeConfig]:
    cfgs: list[DecodeConfig] = [DecodeConfig("greedy", 1, 0.0, 0)]

    beam_sizes = _parse_int_list(args.beam_sizes)
    length_penalties = _parse_float_list(args.length_penalties)
    no_repeat_sizes = _parse_int_list(args.no_repeat_ngram_sizes)

    for b, lp, ng in itertools.product(beam_sizes, length_penalties, no_repeat_sizes):
        cfgs.append(
            DecodeConfig(
                decode_strategy="beam",
                beam_size=int(b),
                length_penalty=float(lp),
                no_repeat_ngram_size=int(ng),
            )
        )
    return cfgs


def _build_base_cmd(args) -> list[str]:
    cmd = [
        sys.executable,
        "src/exp4/main.py",
        "--mode",
        "test_eval",
        "--direction",
        args.direction,
        "--resume_ckpt",
        args.resume_ckpt,
        "--max_test_samples",
        str(args.max_test_samples),
        "--test_batch_size",
        str(args.test_batch_size),
        "--test_decode_len",
        str(args.test_decode_len),
        "--clear_cuda_cache_every",
        str(args.clear_cuda_cache_every),
        "--src_tokenizer",
        args.src_tokenizer,
        "--tgt_tokenizer",
        args.tgt_tokenizer,
    ]
    if args.src_spm_model_path:
        cmd += ["--src_spm_model_path", args.src_spm_model_path]
    if args.tgt_spm_model_path:
        cmd += ["--tgt_spm_model_path", args.tgt_spm_model_path]
    if args.ckpt_name:
        cmd += ["--ckpt_name", args.ckpt_name]
    if args.device:
        cmd += ["--device", args.device]
    return cmd


def _run_one(base_cmd: list[str], cfg: DecodeConfig, cwd: Path, timeout_s: int) -> DecodeResult:
    cmd = list(base_cmd)
    cmd += ["--decode_strategy", cfg.decode_strategy]
    if cfg.decode_strategy == "beam":
        cmd += [
            "--beam_size",
            str(cfg.beam_size),
            "--length_penalty",
            str(cfg.length_penalty),
            "--no_repeat_ngram_size",
            str(cfg.no_repeat_ngram_size),
        ]

    proc = subprocess.run(
        cmd,
        cwd=str(cwd),
        text=True,
        capture_output=True,
        timeout=timeout_s,
    )
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    m = BLEU_PATTERN.search(output)

    if proc.returncode != 0 or m is None:
        return DecodeResult(
            config=cfg,
            bleu4=-1.0,
            samples=0,
            ok=False,
            command=cmd,
            raw_output=output,
        )

    return DecodeResult(
        config=cfg,
        bleu4=float(m.group(1)),
        samples=int(m.group(2)),
        ok=True,
        command=cmd,
        raw_output=output,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Small-scale decode grid search for Exp4")
    parser.add_argument("--resume_ckpt", type=str, required=True)
    parser.add_argument("--direction", choices=["zh2en", "en2zh"], default="zh2en")
    parser.add_argument("--ckpt_name", type=str, default="")
    parser.add_argument("--device", type=str, default="")

    parser.add_argument("--max_test_samples", type=int, default=100)
    parser.add_argument("--test_batch_size", type=int, default=4)
    parser.add_argument("--test_decode_len", type=int, default=48)
    parser.add_argument("--clear_cuda_cache_every", type=int, default=1)

    parser.add_argument("--beam_sizes", type=str, default="4,5,6")
    parser.add_argument("--length_penalties", type=str, default="0.4,0.6,0.8")
    parser.add_argument("--no_repeat_ngram_sizes", type=str, default="2,3,4")

    parser.add_argument("--src_tokenizer", choices=["legacy", "spm"], default="legacy")
    parser.add_argument("--tgt_tokenizer", choices=["legacy", "spm"], default="legacy")
    parser.add_argument("--src_spm_model_path", type=str, default="")
    parser.add_argument("--tgt_spm_model_path", type=str, default="")

    parser.add_argument("--timeout_s", type=int, default=600)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--output_jsonl", type=str, default="src/exp4/decode_grid_results.jsonl")
    parser.add_argument("--full_test_samples", type=int, default=1000)
    parser.add_argument("--full_test_decode_len", type=int, default=128)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cwd = Path(__file__).resolve().parents[2]
    base_cmd = _build_base_cmd(args)
    configs = _build_configs(args)

    print(f"[grid] total configs: {len(configs)}")
    results: list[DecodeResult] = []
    for i, cfg in enumerate(configs, start=1):
        print(
            f"[grid] ({i}/{len(configs)}) strategy={cfg.decode_strategy} "
            f"beam={cfg.beam_size} lp={cfg.length_penalty} ngram={cfg.no_repeat_ngram_size}"
        )
        res = _run_one(base_cmd, cfg, cwd=cwd, timeout_s=args.timeout_s)
        results.append(res)
        if res.ok:
            print(f"[grid] BLEU4={res.bleu4:.4f} on {res.samples}")
        else:
            print("[grid] failed (see output_jsonl for raw logs)")

    out_path = Path(args.output_jsonl)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")

    ok_results = [r for r in results if r.ok]
    if not ok_results:
        print("[grid] no successful runs.")
        print(f"[grid] saved logs: {out_path}")
        return

    ok_results.sort(key=lambda x: x.bleu4, reverse=True)
    top_k = max(1, min(args.top_k, len(ok_results)))
    print(f"\n[grid] Top {top_k} configs:")
    for i in range(top_k):
        r = ok_results[i]
        print(
            f"{i+1}. BLEU4={r.bleu4:.4f} "
            f"strategy={r.config.decode_strategy} beam={r.config.beam_size} "
            f"lp={r.config.length_penalty} ngram={r.config.no_repeat_ngram_size}"
        )

    best = ok_results[0]
    print("\n[grid] Recommended full-eval command:")
    cmd = [
        "python",
        "src/exp4/main.py",
        "--mode",
        "test_eval",
        "--direction",
        args.direction,
        "--resume_ckpt",
        args.resume_ckpt,
        "--max_test_samples",
        str(args.full_test_samples),
        "--test_batch_size",
        str(args.test_batch_size),
        "--test_decode_len",
        str(args.full_test_decode_len),
        "--clear_cuda_cache_every",
        str(args.clear_cuda_cache_every),
        "--decode_strategy",
        best.config.decode_strategy,
    ]
    if best.config.decode_strategy == "beam":
        cmd += [
            "--beam_size",
            str(best.config.beam_size),
            "--length_penalty",
            str(best.config.length_penalty),
            "--no_repeat_ngram_size",
            str(best.config.no_repeat_ngram_size),
        ]
    cmd += ["--src_tokenizer", args.src_tokenizer, "--tgt_tokenizer", args.tgt_tokenizer]
    if args.src_spm_model_path:
        cmd += ["--src_spm_model_path", args.src_spm_model_path]
    if args.tgt_spm_model_path:
        cmd += ["--tgt_spm_model_path", args.tgt_spm_model_path]
    if args.ckpt_name:
        cmd += ["--ckpt_name", args.ckpt_name]
    if args.device:
        cmd += ["--device", args.device]

    print(" ".join(cmd))
    print(f"[grid] detailed logs saved: {out_path}")


if __name__ == "__main__":
    main()
