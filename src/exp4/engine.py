from __future__ import annotations

import random
from pathlib import Path
import math
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F

from data_utils import BaseTokenizer, detokenize, tokenize, Vocab
from evaluator import BLEUAccumulator


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device(device_name: str) -> str:
    if device_name == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _is_cuda_oom_error(exc: RuntimeError) -> bool:
    msg = str(exc).lower()
    return ("out of memory" in msg) and ("cuda" in msg)


@torch.inference_mode()
def _decode_with_oom_retry(
    model,
    src: torch.Tensor,
    bos_idx: int,
    eos_idx: int,
    max_len: int,
    decode_strategy: str = "greedy",
    beam_size: int = 4,
    length_penalty: float = 0.6,
    no_repeat_ngram_size: int = 0,
) -> torch.Tensor:
    if decode_strategy not in {"greedy", "beam"}:
        raise ValueError(f"Unsupported decode strategy: {decode_strategy}")

    def _decode_once(inp: torch.Tensor) -> torch.Tensor:
        if decode_strategy == "beam":
            return model.beam_decode(
                src=inp,
                bos_idx=bos_idx,
                eos_idx=eos_idx,
                max_len=max_len,
                beam_size=beam_size,
                length_penalty=length_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
            )
        return model.greedy_decode(src=inp, bos_idx=bos_idx, eos_idx=eos_idx, max_len=max_len)

    try:
        return _decode_once(src)
    except RuntimeError as exc:
        if (not src.is_cuda) or src.size(0) <= 1 or (not _is_cuda_oom_error(exc)):
            raise
        torch.cuda.empty_cache()
        mid = src.size(0) // 2
        left = _decode_with_oom_retry(
            model,
            src[:mid],
            bos_idx,
            eos_idx,
            max_len,
            decode_strategy=decode_strategy,
            beam_size=beam_size,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        right = _decode_with_oom_retry(
            model,
            src[mid:],
            bos_idx,
            eos_idx,
            max_len,
            decode_strategy=decode_strategy,
            beam_size=beam_size,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )
        return torch.cat([left, right], dim=0)


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device: str,
    pad_idx: int,
    label_smoothing: float,
    grad_clip: float,
    grad_accum_steps: int = 1,
    use_amp: bool = False,
    scaler=None,
    log_interval: int = 100,
    scheduler=None,
    global_step: int = 0,
    on_step_end: Callable[[int, dict], None] | None = None,
) -> tuple[float, int]:
    model.train()
    running_loss = 0.0
    grad_accum_steps = max(1, int(grad_accum_steps))
    optimizer.zero_grad(set_to_none=True)

    for step, (src, tgt) in enumerate(dataloader, start=1):
        src = src.to(device)
        tgt = tgt.to(device)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
            logits = model(src, tgt_in)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                tgt_out.reshape(-1),
                ignore_index=pad_idx,
                label_smoothing=label_smoothing,
            )
        loss_for_backward = loss / grad_accum_steps

        if use_amp and scaler is not None:
            scaler.scale(loss_for_backward).backward()
        else:
            loss_for_backward.backward()

        should_step = (step % grad_accum_steps == 0) or (step == len(dataloader))
        if should_step:
            if use_amp and scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()

        global_step += 1
        if on_step_end is not None:
            on_step_end(
                global_step,
                {
                    "step_in_epoch": step,
                    "loss": float(loss.item()),
                    "lr": float(optimizer.param_groups[0]["lr"]),
                    "optimizer_step": bool(should_step),
                },
            )

        running_loss += loss.item()
        if step % log_interval == 0:
            avg = running_loss / step
            print(f"[train] step={step} avg_loss={avg:.4f}")

    return running_loss / max(1, len(dataloader)), global_step


@torch.no_grad()
def evaluate_loss(model, dataloader, device: str, pad_idx: int) -> float:
    model.eval()
    total_loss = 0.0

    for src, tgt in dataloader:
        src = src.to(device)
        tgt = tgt.to(device)
        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]

        logits = model(src, tgt_in)
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
            ignore_index=pad_idx,
        )
        total_loss += loss.item()

    return total_loss / max(1, len(dataloader))


@torch.inference_mode()
def evaluate_bleu4(
    model,
    dataloader,
    tgt_vocab: Vocab,
    device: str,
    max_decode_len: int,
    clear_cuda_cache_every: int = 0,
    decode_strategy: str = "greedy",
    beam_size: int = 4,
    length_penalty: float = 0.6,
    no_repeat_ngram_size: int = 0,
) -> float:
    model.eval()
    acc = BLEUAccumulator()

    for batch_idx, (src, tgt) in enumerate(dataloader, start=1):
        src = src.to(device)
        tgt = tgt.to(device)

        generated = _decode_with_oom_retry(
            model=model,
            src=src,
            bos_idx=tgt_vocab.bos_idx,
            eos_idx=tgt_vocab.eos_idx,
            max_len=max_decode_len,
            decode_strategy=decode_strategy,
            beam_size=beam_size,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        for i in range(src.size(0)):
            pred_tokens = tgt_vocab.decode(generated[i].tolist(), skip_special=True)
            ref_tokens = tgt_vocab.decode(tgt[i].tolist(), skip_special=True)
            acc.add(ref_tokens, pred_tokens)

        if device == "cuda" and clear_cuda_cache_every > 0 and (batch_idx % clear_cuda_cache_every == 0):
            torch.cuda.empty_cache()

    return acc.compute()


@torch.inference_mode()
def translate_sentences(
    model,
    sentences: list[str],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: str,
    max_src_len: int,
    max_decode_len: int,
    batch_size: int = 64,
    clear_cuda_cache_every: int = 0,
    decode_strategy: str = "greedy",
    beam_size: int = 4,
    length_penalty: float = 0.6,
    no_repeat_ngram_size: int = 0,
    src_tokenizer: BaseTokenizer | None = None,
    tgt_tokenizer: BaseTokenizer | None = None,
) -> list[str]:
    model.eval()
    outputs: list[str] = []

    for batch_idx, start in enumerate(range(0, len(sentences), batch_size), start=1):
        batch_sents = sentences[start : start + batch_size]
        src_tensors = []
        for sent in batch_sents:
            src_ids = src_vocab.encode(tokenize(sent, tokenizer=src_tokenizer), add_eos=True, max_len=max_src_len)
            src_tensors.append(torch.tensor(src_ids, dtype=torch.long))

        src_batch = torch.nn.utils.rnn.pad_sequence(
            src_tensors, batch_first=True, padding_value=src_vocab.pad_idx
        ).to(device)

        generated = _decode_with_oom_retry(
            model=model,
            src=src_batch,
            bos_idx=tgt_vocab.bos_idx,
            eos_idx=tgt_vocab.eos_idx,
            max_len=max_decode_len,
            decode_strategy=decode_strategy,
            beam_size=beam_size,
            length_penalty=length_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
        )

        for i in range(generated.size(0)):
            pred_tokens = tgt_vocab.decode(generated[i].tolist(), skip_special=True)
            outputs.append(detokenize(pred_tokens, tokenizer=tgt_tokenizer))

        if device == "cuda" and clear_cuda_cache_every > 0 and (batch_idx % clear_cuda_cache_every == 0):
            torch.cuda.empty_cache()

    return outputs


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    best_bleu4: float,
    ckpt_path: Path,
    scheduler=None,
    scaler=None,
    global_step: int = 0,
) -> None:
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "best_bleu4": best_bleu4,
            "global_step": int(global_step),
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        },
        ckpt_path,
    )


def load_checkpoint(model, optimizer, ckpt_path: Path, device: str, scheduler=None, scaler=None) -> tuple[int, float, int]:
    state = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    if optimizer is not None and state.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(state["optimizer_state_dict"])
    if scheduler is not None and state.get("scheduler_state_dict") is not None:
        scheduler.load_state_dict(state["scheduler_state_dict"])
    if scaler is not None and state.get("scaler_state_dict") is not None:
        scaler.load_state_dict(state["scaler_state_dict"])
    return (
        int(state.get("epoch", 0)),
        float(state.get("best_bleu4", 0.0)),
        int(state.get("global_step", 0)),
    )


def build_scheduler(
    optimizer,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.1,
):
    if scheduler_type == "none":
        return None

    if scheduler_type != "warmup_cosine":
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")

    warmup_steps = max(1, int(warmup_steps))
    total_steps = max(warmup_steps + 1, int(total_steps))
    min_lr_ratio = float(max(0.0, min(1.0, min_lr_ratio)))

    def _lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step + 1) / float(warmup_steps)

        progress = (current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=_lr_lambda)
