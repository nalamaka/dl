from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from datetime import datetime

import torch
from torch.utils.data import DataLoader

from config import Config
from data_utils import (
    BaseTokenizer,
    TranslationDataset,
    build_tokenizer,
    build_collate_fn,
    build_vocab,
    get_tokenizer_tag,
    split_train_val,
    load_parallel_data,
    load_test_source,
    load_vocab,
    parse_reference_pairs,
    save_vocab,
    tokenize,
    Vocab,
)
from engine import (
    build_scheduler,
    evaluate_bleu4,
    evaluate_loss,
    load_checkpoint,
    save_checkpoint,
    select_device,
    set_seed,
    train_one_epoch,
    translate_sentences,
)
from evaluator import corpus_bleu4
from model import TransformerSeq2Seq


def _bleu_to_display(bleu: float) -> float:
    return float(bleu * 100.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Exp4: Transformer-based NMT (zh<->en)")
    parser.add_argument("--mode", choices=["train", "eval", "translate", "test_eval", "test_preview"], required=True)
    parser.add_argument("--direction", choices=["zh2en", "en2zh"], default="zh2en")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume_ckpt", type=str, default="")
    parser.add_argument("--resume_policy", choices=["auto", "last", "best"], default="auto")
    parser.add_argument("--ckpt_name", type=str, default="")

    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--eval_batch_size", type=int, default=Config.eval_batch_size)
    parser.add_argument("--test_batch_size", type=int, default=Config.test_batch_size)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    parser.add_argument("--label_smoothing", type=float, default=Config.label_smoothing)
    parser.add_argument("--grad_clip", type=float, default=Config.grad_clip)
    parser.add_argument("--grad_accum_steps", type=int, default=Config.grad_accum_steps)
    parser.add_argument("--use_amp", action="store_true", default=Config.use_amp)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--save_every_steps", type=int, default=Config.save_every_steps)
    parser.add_argument("--val_ratio", type=float, default=Config.val_ratio)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=Config.num_workers)
    parser.add_argument("--max_src_len", type=int, default=Config.max_src_len)
    parser.add_argument("--max_tgt_len", type=int, default=Config.max_tgt_len)
    parser.add_argument("--max_decode_len", type=int, default=Config.max_decode_len)
    parser.add_argument("--max_eval_samples", type=int, default=Config.max_eval_samples)
    parser.add_argument("--max_test_samples", type=int, default=Config.max_test_samples)
    parser.add_argument("--clear_cuda_cache_every", type=int, default=Config.clear_cuda_cache_every)
    parser.add_argument("--eval_decode_len", type=int, default=Config.eval_decode_len)
    parser.add_argument("--test_decode_len", type=int, default=Config.test_decode_len)
    parser.add_argument("--decode_strategy", choices=["greedy", "beam"], default=Config.decode_strategy)
    parser.add_argument("--beam_size", type=int, default=Config.beam_size)
    parser.add_argument("--length_penalty", type=float, default=Config.length_penalty)
    parser.add_argument("--no_repeat_ngram_size", type=int, default=Config.no_repeat_ngram_size)
    parser.add_argument("--preview_samples", type=int, default=Config.preview_samples)
    parser.add_argument("--preview_max_decode_len", type=int, default=Config.preview_max_decode_len)
    parser.add_argument("--show_samples", type=int, default=10)
    parser.add_argument("--scheduler", choices=["none", "warmup_cosine"], default=Config.scheduler)
    parser.add_argument("--warmup_steps", type=int, default=Config.warmup_steps)
    parser.add_argument("--min_lr_ratio", type=float, default=Config.min_lr_ratio)

    parser.add_argument("--d_model", type=int, default=Config.d_model)
    parser.add_argument("--nhead", type=int, default=Config.nhead)
    parser.add_argument("--num_encoder_layers", type=int, default=Config.num_encoder_layers)
    parser.add_argument("--num_decoder_layers", type=int, default=Config.num_decoder_layers)
    parser.add_argument("--dim_feedforward", type=int, default=Config.dim_feedforward)
    parser.add_argument("--dropout", type=float, default=Config.dropout)
    parser.add_argument("--tie_embeddings", dest="tie_embeddings", action="store_true")
    parser.add_argument("--no_tie_embeddings", dest="tie_embeddings", action="store_false")
    parser.set_defaults(tie_embeddings=Config.tie_embeddings)

    parser.add_argument("--min_freq", type=int, default=Config.min_freq)
    parser.add_argument("--max_src_vocab_size", type=int, default=Config.max_src_vocab_size)
    parser.add_argument("--max_tgt_vocab_size", type=int, default=Config.max_tgt_vocab_size)
    parser.add_argument("--src_tokenizer", choices=["legacy", "spm"], default="legacy")
    parser.add_argument("--tgt_tokenizer", choices=["legacy", "spm"], default="legacy")
    parser.add_argument("--src_spm_model_path", type=str, default="")
    parser.add_argument("--tgt_spm_model_path", type=str, default="")
    parser.add_argument("--spm_vocab_size", type=int, default=16000)
    parser.add_argument("--spm_model_type", choices=["bpe", "unigram"], default="bpe")
    parser.add_argument("--spm_character_coverage", type=float, default=1.0)
    parser.add_argument("--auto_train_spm", action="store_true")

    parser.add_argument("--device", type=str, default=Config.device)
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--log_interval", type=int, default=Config.log_interval)
    parser.add_argument("--log_dir", type=str, default=str(Config.log_dir))
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--log_every_steps", type=int, default=Config.log_every_steps)

    parser.add_argument("--input_text", type=str, default="")
    parser.add_argument("--input_file", type=str, default="")
    parser.add_argument("--output_file", type=str, default="")
    return parser.parse_args()


def _safe_console_text(text: str) -> str:
    enc = sys.stdout.encoding or "utf-8"
    try:
        text.encode(enc)
        return text
    except UnicodeEncodeError:
        return text.encode(enc, errors="replace").decode(enc, errors="replace")


def _direction_paths(direction: str) -> tuple[Path, Path]:
    if direction == "zh2en":
        return Config.train_zh_path, Config.train_en_path
    return Config.train_en_path, Config.train_zh_path


def _default_ckpt_name(direction: str) -> str:
    return f"transformer_{direction}.pth"


def _tokenization_suffix(src_tokenizer: str, tgt_tokenizer: str, src_spm_model: Path | None, tgt_spm_model: Path | None) -> str:
    src_tag = get_tokenizer_tag(src_tokenizer, src_spm_model)
    tgt_tag = get_tokenizer_tag(tgt_tokenizer, tgt_spm_model)
    if src_tag == "legacy" and tgt_tag == "legacy":
        return ""
    return f"__{src_tag}-to-{tgt_tag}"


def _build_ckpt_paths(ckpt_name: str) -> tuple[Path, Path, Path]:
    base_path = Config.save_dir / ckpt_name
    stem = Path(ckpt_name).stem
    suffix = Path(ckpt_name).suffix or ".pth"
    best_path = Config.save_dir / f"{stem}.best{suffix}"
    last_path = Config.save_dir / f"{stem}.last{suffix}"
    return base_path, best_path, last_path


def _resolve_resume_path(args, base_path: Path, best_path: Path, last_path: Path) -> Path | None:
    if args.resume_ckpt:
        p = Path(args.resume_ckpt)
        return p if p.exists() else None

    if args.resume_policy == "last":
        return last_path if last_path.exists() else None
    if args.resume_policy == "best":
        if best_path.exists():
            return best_path
        if base_path.exists():
            return base_path
        return None

    # auto: prefer last for true "continue training", fallback to best/base.
    if last_path.exists():
        return last_path
    if best_path.exists():
        return best_path
    if base_path.exists():
        return base_path
    return None


def _resolve_infer_ckpt_path(args, base_path: Path, best_path: Path, last_path: Path) -> Path:
    if args.resume_ckpt:
        p = Path(args.resume_ckpt)
        if not p.exists():
            raise FileNotFoundError(f"checkpoint not found: {p}")
        return p

    # inference/eval prefers best.
    for p in (best_path, base_path, last_path):
        if p.exists():
            return p
    raise FileNotFoundError(
        f"checkpoint not found. tried: {best_path}, {base_path}, {last_path}"
    )


def _default_vocab_paths(direction: str, artifact_suffix: str = "") -> tuple[Path, Path]:
    src_vocab_path = Config.vocab_dir / f"src_vocab_{direction}{artifact_suffix}.json"
    tgt_vocab_path = Config.vocab_dir / f"tgt_vocab_{direction}{artifact_suffix}.json"
    return src_vocab_path, tgt_vocab_path


def _resolve_spm_model_paths(args, direction: str) -> tuple[Path | None, Path | None]:
    src_spm = Path(args.src_spm_model_path) if args.src_spm_model_path else None
    tgt_spm = Path(args.tgt_spm_model_path) if args.tgt_spm_model_path else None

    if args.src_tokenizer == "spm" and src_spm is None:
        src_spm = Config.tokenizer_dir / f"{direction}.src.spm.model"
    if args.tgt_tokenizer == "spm" and tgt_spm is None:
        tgt_spm = Config.tokenizer_dir / f"{direction}.tgt.spm.model"
    return src_spm, tgt_spm


def _build_tokenizers(args, direction: str, src_train_path: Path, tgt_train_path: Path) -> tuple[BaseTokenizer, BaseTokenizer, Path | None, Path | None]:
    src_spm_path, tgt_spm_path = _resolve_spm_model_paths(args, direction)
    src_tok = build_tokenizer(
        tokenizer_type=args.src_tokenizer,
        spm_model_path=src_spm_path,
        spm_train_files=[src_train_path],
        spm_vocab_size=args.spm_vocab_size,
        spm_model_type=args.spm_model_type,
        spm_character_coverage=args.spm_character_coverage,
        auto_train_spm=args.auto_train_spm,
    )
    tgt_tok = build_tokenizer(
        tokenizer_type=args.tgt_tokenizer,
        spm_model_path=tgt_spm_path,
        spm_train_files=[tgt_train_path],
        spm_vocab_size=args.spm_vocab_size,
        spm_model_type=args.spm_model_type,
        spm_character_coverage=args.spm_character_coverage,
        auto_train_spm=args.auto_train_spm,
    )
    return src_tok, tgt_tok, src_spm_path, tgt_spm_path


def build_model(args, src_vocab: Vocab, tgt_vocab: Vocab) -> TransformerSeq2Seq:
    return TransformerSeq2Seq(
        src_vocab_size=len(src_vocab.itos),
        tgt_vocab_size=len(tgt_vocab.itos),
        src_pad_idx=src_vocab.pad_idx,
        tgt_pad_idx=tgt_vocab.pad_idx,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        num_decoder_layers=args.num_decoder_layers,
        dim_feedforward=args.dim_feedforward,
        dropout=args.dropout,
        max_len=max(args.max_src_len, args.max_tgt_len) + 8,
        tie_embeddings=args.tie_embeddings,
    )


def _load_or_build_vocabs(
    args,
    train_pairs: list[tuple[str, str]],
    direction: str,
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer,
    artifact_suffix: str,
) -> tuple[Vocab, Vocab]:
    src_vocab_path, tgt_vocab_path = _default_vocab_paths(direction, artifact_suffix=artifact_suffix)
    if src_vocab_path.exists() and tgt_vocab_path.exists():
        src_vocab = load_vocab(src_vocab_path)
        tgt_vocab = load_vocab(tgt_vocab_path)
        return src_vocab, tgt_vocab

    src_tokens = [tokenize(src, tokenizer=src_tokenizer) for src, _ in train_pairs]
    tgt_tokens = [tokenize(tgt, tokenizer=tgt_tokenizer) for _, tgt in train_pairs]

    src_vocab = build_vocab(src_tokens, max_size=args.max_src_vocab_size, min_freq=args.min_freq)
    tgt_vocab = build_vocab(tgt_tokens, max_size=args.max_tgt_vocab_size, min_freq=args.min_freq)

    save_vocab(src_vocab, src_vocab_path)
    save_vocab(tgt_vocab, tgt_vocab_path)
    print(f"saved src vocab -> {src_vocab_path}")
    print(f"saved tgt vocab -> {tgt_vocab_path}")
    return src_vocab, tgt_vocab


def _build_dataloader(
    pairs: list[tuple[str, str]],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer,
    batch_size: int,
    max_src_len: int,
    max_tgt_len: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TranslationDataset(
        pairs=pairs,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_src_len=max_src_len,
        max_tgt_len=max_tgt_len,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )
    collate_fn = build_collate_fn(src_pad_idx=src_vocab.pad_idx, tgt_pad_idx=tgt_vocab.pad_idx)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )


def _print_epoch_preview(
    model,
    device: str,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    src_tokenizer: BaseTokenizer,
    tgt_tokenizer: BaseTokenizer,
    val_pairs: list[tuple[str, str]],
    preview_samples: int,
    preview_max_decode_len: int,
    batch_size: int,
    max_src_len: int,
    clear_cuda_cache_every: int,
    decode_strategy: str,
    beam_size: int,
    length_penalty: float,
    no_repeat_ngram_size: int,
) -> None:
    n = max(0, min(preview_samples, len(val_pairs)))
    if n <= 0:
        return

    preview_pairs = val_pairs[:n]
    src_texts = [src for src, _ in preview_pairs]
    ref_texts = [tgt for _, tgt in preview_pairs]
    pred_texts = translate_sentences(
        model=model,
        sentences=src_texts,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        max_src_len=max_src_len,
        max_decode_len=preview_max_decode_len,
        batch_size=max(1, min(batch_size, n)),
        clear_cuda_cache_every=clear_cuda_cache_every,
        decode_strategy=decode_strategy,
        beam_size=beam_size,
        length_penalty=length_penalty,
        no_repeat_ngram_size=no_repeat_ngram_size,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )

    print("[preview] ------------------------------")
    for i, (src, ref, pred) in enumerate(zip(src_texts, ref_texts, pred_texts), start=1):
        print(f"[preview {i}] SRC : {_safe_console_text(src)}")
        print(f"[preview {i}] REF : {_safe_console_text(ref)}")
        print(f"[preview {i}] PRED: {_safe_console_text(pred)}")
    print("[preview] ------------------------------")


def _pick_sample_indices(total: int, k: int) -> list[int]:
    if total <= 0 or k <= 0:
        return []
    if k >= total:
        return list(range(total))
    if k == 1:
        return [0]

    step = (total - 1) / float(k - 1)
    indices = [int(round(i * step)) for i in range(k)]
    deduped: list[int] = []
    seen: set[int] = set()
    for idx in indices:
        idx = max(0, min(total - 1, idx))
        if idx not in seen:
            seen.add(idx)
            deduped.append(idx)
    return deduped


def _print_test_samples(
    src_texts: list[str],
    ref_texts: list[str],
    pred_texts: list[str],
    show_samples: int,
) -> None:
    indices = _pick_sample_indices(min(len(src_texts), len(ref_texts), len(pred_texts)), show_samples)
    if not indices:
        return

    print("[samples] ------------------------------")
    for rank, idx in enumerate(indices, start=1):
        print(f"[sample {rank}] INDEX: {idx}")
        print(f"[sample {rank}] SRC : {_safe_console_text(src_texts[idx])}")
        print(f"[sample {rank}] REF : {_safe_console_text(ref_texts[idx])}")
        print(f"[sample {rank}] PRED: {_safe_console_text(pred_texts[idx])}")
    print("[samples] ------------------------------")


def _prepare_test_texts(args, tgt_tokenizer: BaseTokenizer) -> tuple[list[str], list[str], list[list[str]]]:
    if args.direction == "zh2en":
        test_src_lines = load_test_source(Config.test_zh_path)
        ref_pairs = parse_reference_pairs(Config.test_reference_path)
        ref_texts = [en for _, en in ref_pairs[: len(test_src_lines)]]
        refs = [tokenize(en, tokenizer=tgt_tokenizer) for _, en in ref_pairs[: len(test_src_lines)]]
    else:
        ref_pairs = parse_reference_pairs(Config.test_reference_path)
        test_src_lines = [en for _, en in ref_pairs]
        ref_texts = [zh for zh, _ in ref_pairs]
        refs = [tokenize(zh, tokenizer=tgt_tokenizer) for zh, _ in ref_pairs]

    if args.max_test_samples > 0:
        test_src_lines = test_src_lines[: args.max_test_samples]
        ref_texts = ref_texts[: args.max_test_samples]
        refs = refs[: args.max_test_samples]

    return test_src_lines, ref_texts, refs


def run_train(args):
    set_seed(args.seed)
    if args.no_amp:
        args.use_amp = False
    device = select_device(args.device)
    if device != "cuda":
        args.use_amp = False

    src_path, tgt_path = _direction_paths(args.direction)
    src_tokenizer, tgt_tokenizer, src_spm_path, tgt_spm_path = _build_tokenizers(
        args=args,
        direction=args.direction,
        src_train_path=src_path,
        tgt_train_path=tgt_path,
    )
    artifact_suffix = _tokenization_suffix(
        args.src_tokenizer,
        args.tgt_tokenizer,
        src_spm_path,
        tgt_spm_path,
    )
    max_samples = args.max_train_samples if args.max_train_samples > 0 else None
    pairs = load_parallel_data(src_path, tgt_path, max_samples=max_samples)
    train_pairs, val_pairs = split_train_val(pairs, args.val_ratio, args.seed)

    src_vocab, tgt_vocab = _load_or_build_vocabs(
        args,
        train_pairs,
        args.direction,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        artifact_suffix=artifact_suffix,
    )
    print(
        f"[train-config] device={device} use_amp={args.use_amp} "
        f"grad_accum_steps={max(1, args.grad_accum_steps)} "
        f"effective_batch_size={args.batch_size * max(1, args.grad_accum_steps)}"
    )
    train_loader = _build_dataloader(
        train_pairs,
        src_vocab,
        tgt_vocab,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=args.batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        num_workers=args.num_workers,
        shuffle=True,
    )
    train_eval_batch_size = (
        args.eval_batch_size if args.eval_batch_size > 0 else min(args.batch_size, 8)
    )
    val_loader = _build_dataloader(
        val_pairs,
        src_vocab,
        tgt_vocab,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=train_eval_batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        num_workers=args.num_workers,
        shuffle=False,
    )

    model = build_model(args, src_vocab, tgt_vocab).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)
    total_steps = max(1, len(train_loader) * max(1, args.epochs))
    scheduler = build_scheduler(
        optimizer=optimizer,
        scheduler_type=args.scheduler,
        warmup_steps=args.warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    if args.ckpt_name:
        ckpt_name = args.ckpt_name
    else:
        base_name = _default_ckpt_name(args.direction)
        if artifact_suffix:
            base = Path(base_name)
            ckpt_name = f"{base.stem}{artifact_suffix}{base.suffix or '.pth'}"
        else:
            ckpt_name = base_name
    ckpt_path, best_ckpt_path, last_ckpt_path = _build_ckpt_paths(ckpt_name)
    stem = Path(ckpt_name).stem
    suffix = Path(ckpt_name).suffix or ".pth"
    step_ckpt_path = Config.save_dir / f"{stem}.step{suffix}"

    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    if args.run_name.strip():
        run_name = args.run_name.strip()
    else:
        run_name = f"{stem}__{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    log_path = log_dir / f"{run_name}.jsonl"

    def _append_log(record: dict) -> None:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    _append_log(
        {
            "type": "meta",
            "run_name": run_name,
            "direction": args.direction,
            "device": device,
            "use_amp": bool(args.use_amp),
            "grad_accum_steps": int(max(1, args.grad_accum_steps)),
            "batch_size": int(args.batch_size),
            "effective_batch_size": int(args.batch_size * max(1, args.grad_accum_steps)),
            "decode_strategy": args.decode_strategy,
            "beam_size": int(args.beam_size),
            "length_penalty": float(args.length_penalty),
            "no_repeat_ngram_size": int(args.no_repeat_ngram_size),
            "src_tokenizer": args.src_tokenizer,
            "tgt_tokenizer": args.tgt_tokenizer,
        }
    )
    print(f"[log] writing train logs -> {log_path}")

    start_epoch = 1
    best_bleu = 0.0
    global_step = 0
    if args.resume:
        resume_path = _resolve_resume_path(args, ckpt_path, best_ckpt_path, last_ckpt_path)
        if resume_path is not None:
            epoch_done, best_bleu, global_step = load_checkpoint(
                model, optimizer, resume_path, device, scheduler=scheduler, scaler=scaler
            )
            start_epoch = epoch_done + 1
            print(
                f"resume from {resume_path}, start_epoch={start_epoch}, "
                f"best_bleu4={best_bleu:.4f}, global_step={global_step}, "
                f"use_amp={args.use_amp}, grad_accum_steps={args.grad_accum_steps}"
            )
        else:
            print("[resume] no available checkpoint found, training from scratch.")

    for epoch in range(start_epoch, args.epochs + 1):
        def _step_ckpt_callback(curr_step: int, info: dict):
            if args.save_every_steps > 0 and curr_step % args.save_every_steps == 0:
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    best_bleu4=best_bleu,
                    ckpt_path=step_ckpt_path,
                    scheduler=scheduler,
                    scaler=scaler,
                    global_step=curr_step,
                )
                print(f"[ckpt] step checkpoint saved -> {step_ckpt_path} (global_step={curr_step})")

            if args.log_every_steps > 0 and curr_step % args.log_every_steps == 0:
                _append_log(
                    {
                        "type": "step",
                        "epoch": int(epoch),
                        "global_step": int(curr_step),
                        "step_in_epoch": int(info.get("step_in_epoch", -1)),
                        "loss": float(info.get("loss", 0.0)),
                        "lr": float(info.get("lr", 0.0)),
                        "optimizer_step": bool(info.get("optimizer_step", False)),
                    }
                )

        train_loss, global_step = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
            pad_idx=tgt_vocab.pad_idx,
            label_smoothing=args.label_smoothing,
            grad_clip=args.grad_clip,
            grad_accum_steps=args.grad_accum_steps,
            use_amp=args.use_amp,
            scaler=scaler,
            log_interval=args.log_interval,
            scheduler=scheduler,
            global_step=global_step,
            on_step_end=_step_ckpt_callback,
        )
        val_loss = evaluate_loss(model, val_loader, device=device, pad_idx=tgt_vocab.pad_idx)
        val_bleu = evaluate_bleu4(
            model=model,
            dataloader=val_loader,
            tgt_vocab=tgt_vocab,
            device=device,
            max_decode_len=args.eval_decode_len,
            clear_cuda_cache_every=args.clear_cuda_cache_every,
            decode_strategy=args.decode_strategy,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} val_bleu4={_bleu_to_display(val_bleu):.2f}"
        )
        _append_log(
            {
                "type": "epoch",
                "epoch": int(epoch),
                "global_step": int(global_step),
                "train_loss": float(train_loss),
                "val_loss": float(val_loss),
                "val_bleu4": _bleu_to_display(val_bleu),
                "val_bleu4_raw": float(val_bleu),
                "best_bleu4_before_update": _bleu_to_display(best_bleu),
                "best_bleu4_before_update_raw": float(best_bleu),
            }
        )
        _print_epoch_preview(
            model=model,
            device=device,
            src_vocab=src_vocab,
            tgt_vocab=tgt_vocab,
            src_tokenizer=src_tokenizer,
            tgt_tokenizer=tgt_tokenizer,
            val_pairs=val_pairs,
            preview_samples=args.preview_samples,
            preview_max_decode_len=args.preview_max_decode_len,
            batch_size=train_eval_batch_size,
            max_src_len=args.max_src_len,
            clear_cuda_cache_every=args.clear_cuda_cache_every,
            decode_strategy=args.decode_strategy,
            beam_size=args.beam_size,
            length_penalty=args.length_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )

        if val_bleu >= best_bleu:
            best_bleu = val_bleu
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_bleu,
                best_ckpt_path,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
            )
            # Keep backward-compatible base name.
            save_checkpoint(
                model,
                optimizer,
                epoch,
                best_bleu,
                ckpt_path,
                scheduler=scheduler,
                scaler=scaler,
                global_step=global_step,
            )
            print(f"saved best checkpoint -> {best_ckpt_path}")
            _append_log(
                {
                    "type": "best",
                    "epoch": int(epoch),
                    "global_step": int(global_step),
                    "best_bleu4": _bleu_to_display(best_bleu),
                    "best_bleu4_raw": float(best_bleu),
                    "best_ckpt_path": str(best_ckpt_path),
                }
            )

        # Always save "last" to support reliable resume after interruption.
        save_checkpoint(
            model,
            optimizer,
            epoch,
            best_bleu,
            last_ckpt_path,
            scheduler=scheduler,
            scaler=scaler,
            global_step=global_step,
        )
        print(f"saved last checkpoint -> {last_ckpt_path}")

    _append_log(
        {
            "type": "finish",
            "global_step": int(global_step),
            "best_bleu4": _bleu_to_display(best_bleu),
            "best_bleu4_raw": float(best_bleu),
            "best_ckpt_path": str(best_ckpt_path),
            "last_ckpt_path": str(last_ckpt_path),
        }
    )


def _load_infer_objects(args):
    device = select_device(args.device)
    src_path, tgt_path = _direction_paths(args.direction)
    src_tokenizer, tgt_tokenizer, src_spm_path, tgt_spm_path = _build_tokenizers(
        args=args,
        direction=args.direction,
        src_train_path=src_path,
        tgt_train_path=tgt_path,
    )
    artifact_suffix = _tokenization_suffix(
        args.src_tokenizer,
        args.tgt_tokenizer,
        src_spm_path,
        tgt_spm_path,
    )
    src_vocab_path, tgt_vocab_path = _default_vocab_paths(args.direction, artifact_suffix=artifact_suffix)
    if not (src_vocab_path.exists() and tgt_vocab_path.exists()):
        raise FileNotFoundError(
            f"vocab not found: {src_vocab_path} / {tgt_vocab_path}. "
            "Please run train first."
        )

    src_vocab = load_vocab(src_vocab_path)
    tgt_vocab = load_vocab(tgt_vocab_path)
    model = build_model(args, src_vocab, tgt_vocab).to(device)

    if args.ckpt_name:
        ckpt_name = args.ckpt_name
    else:
        base_name = _default_ckpt_name(args.direction)
        if artifact_suffix:
            base = Path(base_name)
            ckpt_name = f"{base.stem}{artifact_suffix}{base.suffix or '.pth'}"
        else:
            ckpt_name = base_name
    base_path, best_path, last_path = _build_ckpt_paths(ckpt_name)
    ckpt_path = _resolve_infer_ckpt_path(args, base_path, best_path, last_path)
    load_checkpoint(model, None, ckpt_path, device)
    print(f"loaded checkpoint -> {ckpt_path}")
    return model, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, device


def run_eval(args):
    model, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, device = _load_infer_objects(args)
    src_path, tgt_path = _direction_paths(args.direction)
    max_samples = args.max_train_samples if args.max_train_samples > 0 else None
    pairs = load_parallel_data(src_path, tgt_path, max_samples=max_samples)
    _, val_pairs = split_train_val(pairs, args.val_ratio, args.seed)
    if args.max_eval_samples > 0:
        val_pairs = val_pairs[: args.max_eval_samples]
    eval_batch_size = args.eval_batch_size if args.eval_batch_size > 0 else min(args.batch_size, 8)
    val_loader = _build_dataloader(
        val_pairs,
        src_vocab,
        tgt_vocab,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
        batch_size=eval_batch_size,
        max_src_len=args.max_src_len,
        max_tgt_len=args.max_tgt_len,
        num_workers=args.num_workers,
        shuffle=False,
    )
    val_bleu = evaluate_bleu4(
        model=model,
        dataloader=val_loader,
        tgt_vocab=tgt_vocab,
        device=device,
        max_decode_len=args.eval_decode_len,
        clear_cuda_cache_every=args.clear_cuda_cache_every,
        decode_strategy=args.decode_strategy,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
    )
    print(f"Validation BLEU4 ({args.direction}): {_bleu_to_display(val_bleu):.2f}")


def run_translate(args):
    model, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, device = _load_infer_objects(args)
    inputs: list[str] = []

    if args.input_text.strip():
        inputs = [args.input_text.strip()]
    elif args.input_file:
        p = Path(args.input_file)
        if not p.exists():
            raise FileNotFoundError(f"input_file not found: {p}")
        inputs = [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        raise ValueError("Please provide --input_text or --input_file for translate mode.")

    outputs = translate_sentences(
        model=model,
        sentences=inputs,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        max_src_len=args.max_src_len,
        max_decode_len=args.eval_decode_len,
        batch_size=(args.eval_batch_size if args.eval_batch_size > 0 else min(args.batch_size, 8)),
        clear_cuda_cache_every=args.clear_cuda_cache_every,
        decode_strategy=args.decode_strategy,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(outputs), encoding="utf-8")
        print(f"saved translation -> {out_path}")

    for i, (src, tgt) in enumerate(zip(inputs, outputs), start=1):
        print(f"[{i}] SRC: {_safe_console_text(src)}")
        print(f"[{i}] TGT: {_safe_console_text(tgt)}")


def run_test_eval(args):
    model, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, device = _load_infer_objects(args)

    test_src_lines, ref_texts, refs = _prepare_test_texts(args, tgt_tokenizer)

    test_batch_size = args.test_batch_size if args.test_batch_size > 0 else args.batch_size
    preds_text = translate_sentences(
        model=model,
        sentences=test_src_lines,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        max_src_len=args.max_src_len,
        max_decode_len=args.test_decode_len,
        batch_size=test_batch_size,
        clear_cuda_cache_every=args.clear_cuda_cache_every,
        decode_strategy=args.decode_strategy,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )
    preds = [tokenize(x, tokenizer=tgt_tokenizer) for x in preds_text]

    n = min(len(refs), len(preds))
    bleu4 = corpus_bleu4(refs[:n], preds[:n])
    print(f"Test BLEU4 ({args.direction}): {_bleu_to_display(bleu4):.2f} on {n} samples")

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("\n".join(preds_text), encoding="utf-8")
        print(f"saved predictions -> {out_path}")


def run_test_preview(args):
    model, src_vocab, tgt_vocab, src_tokenizer, tgt_tokenizer, device = _load_infer_objects(args)
    test_src_lines, ref_texts, _ = _prepare_test_texts(args, tgt_tokenizer)

    total = len(test_src_lines)
    indices = _pick_sample_indices(total, args.show_samples)
    if not indices:
        print("[samples] no samples selected.")
        return

    sample_src = [test_src_lines[idx] for idx in indices]
    sample_ref = [ref_texts[idx] for idx in indices]
    test_batch_size = min(args.test_batch_size if args.test_batch_size > 0 else args.batch_size, max(1, len(sample_src)))
    preds_text = translate_sentences(
        model=model,
        sentences=sample_src,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        max_src_len=args.max_src_len,
        max_decode_len=args.test_decode_len,
        batch_size=test_batch_size,
        clear_cuda_cache_every=args.clear_cuda_cache_every,
        decode_strategy=args.decode_strategy,
        beam_size=args.beam_size,
        length_penalty=args.length_penalty,
        no_repeat_ngram_size=args.no_repeat_ngram_size,
        src_tokenizer=src_tokenizer,
        tgt_tokenizer=tgt_tokenizer,
    )

    print(f"Test Preview ({args.direction}): showing {len(sample_src)} sampled cases from {total} official test samples")
    print("[samples] ------------------------------")
    for rank, (idx, src, ref, pred) in enumerate(zip(indices, sample_src, sample_ref, preds_text), start=1):
        print(f"[sample {rank}] INDEX: {idx}")
        print(f"[sample {rank}] SRC : {_safe_console_text(src)}")
        print(f"[sample {rank}] REF : {_safe_console_text(ref)}")
        print(f"[sample {rank}] PRED: {_safe_console_text(pred)}")
    print("[samples] ------------------------------")

    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        lines: list[str] = []
        for idx, src, ref, pred in zip(indices, sample_src, sample_ref, preds_text):
            lines.extend(
                [
                    f"[INDEX] {idx}",
                    f"[SRC] {src}",
                    f"[REF] {ref}",
                    f"[PRED] {pred}",
                    "",
                ]
            )
        out_path.write_text("\n".join(lines), encoding="utf-8")
        print(f"saved preview samples -> {out_path}")


def main():
    args = parse_args()
    if args.mode == "train":
        run_train(args)
    elif args.mode == "eval":
        run_eval(args)
    elif args.mode == "translate":
        run_translate(args)
    elif args.mode == "test_eval":
        run_test_eval(args)
    else:
        run_test_preview(args)


if __name__ == "__main__":
    main()
