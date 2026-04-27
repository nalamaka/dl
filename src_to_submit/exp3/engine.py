import math
import csv
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
import torch.optim as optim


def _build_pad_mask(x: torch.Tensor, pad_idx: int | None):
    if pad_idx is None:
        return None
    return x.eq(pad_idx)


def _build_scheduler(
    optimizer,
    scheduler_type: str,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    scheduler_type = scheduler_type.lower()
    if scheduler_type == "none" or total_steps <= 0:
        return None

    min_lr_ratio = max(0.0, min(1.0, min_lr_ratio))
    warmup_steps = max(0, min(warmup_steps, total_steps - 1 if total_steps > 1 else 0))

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step + 1) / float(max(1, warmup_steps))

        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))

        if scheduler_type == "linear":
            return min_lr_ratio + (1.0 - progress) * (1.0 - min_lr_ratio)

        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + cosine * (1.0 - min_lr_ratio)

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def _extract_model_state_dict(state_obj):
    if isinstance(state_obj, dict) and "model_state_dict" in state_obj:
        return state_obj["model_state_dict"]
    return state_obj


def _save_training_checkpoint(
    save_path: Path,
    model,
    optimizer,
    scheduler,
    scaler,
    epoch: int,
    global_update_step: int,
    amp_enabled: bool,
):
    ckpt = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "global_update_step": global_update_step,
        "amp_enabled": bool(amp_enabled),
    }
    if scheduler is not None:
        ckpt["scheduler_state_dict"] = scheduler.state_dict()
    if scaler is not None and amp_enabled:
        ckpt["scaler_state_dict"] = scaler.state_dict()

    torch.save(ckpt, save_path)


def train_model(
    model,
    dataloader,
    epochs,
    lr,
    device,
    save_path: Path,
    pad_idx=None,
    grad_accum_steps: int = 1,
    weight_decay: float = 0.0,
    max_grad_norm: float = 1.0,
    warmup_ratio: float = 0.0,
    min_lr_ratio: float = 0.1,
    scheduler_type: str = "none",
    use_amp: bool = True,
    log_interval: int = 200,
    resume_from: Path | None = None,
    history_path: Path | None = None,
):
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx) if pad_idx is not None else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    amp_enabled = bool(use_amp and device.startswith("cuda"))
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    updates_per_epoch = math.ceil(len(dataloader) / max(1, grad_accum_steps))
    total_updates = max(1, updates_per_epoch * epochs)
    warmup_steps = int(total_updates * max(0.0, warmup_ratio))
    scheduler = _build_scheduler(
        optimizer=optimizer,
        scheduler_type=scheduler_type,
        total_steps=total_updates,
        warmup_steps=warmup_steps,
        min_lr_ratio=min_lr_ratio,
    )

    model.to(device)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    start_epoch = 1
    global_update_step = 0

    if resume_from is not None and resume_from.exists():
        ckpt = torch.load(resume_from, map_location=device)
        model_state = _extract_model_state_dict(ckpt)
        load_result = model.load_state_dict(model_state, strict=False)
        if load_result.missing_keys or load_result.unexpected_keys:
            print("[warning] Resume checkpoint model keys mismatch.")
            if load_result.missing_keys:
                print(f"[warning] Missing keys: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"[warning] Unexpected keys: {load_result.unexpected_keys}")

        if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            if scheduler is not None and "scheduler_state_dict" in ckpt:
                scheduler.load_state_dict(ckpt["scheduler_state_dict"])
            if amp_enabled and "scaler_state_dict" in ckpt:
                scaler.load_state_dict(ckpt["scaler_state_dict"])

            start_epoch = int(ckpt.get("epoch", 0)) + 1
            global_update_step = int(ckpt.get("global_update_step", 0))
            print(
                f"Resumed training from checkpoint: {resume_from} | "
                f"start_epoch={start_epoch}, global_update_step={global_update_step}"
            )
        else:
            print(
                "[warning] Resume checkpoint does not contain optimizer/scheduler states. "
                "Will continue from loaded model weights only."
            )

    csv_file = None
    csv_writer = None
    if history_path is not None:
        history_path.parent.mkdir(parents=True, exist_ok=True)
        append_mode = bool(start_epoch > 1 and history_path.exists())
        csv_file = history_path.open("a" if append_mode else "w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        if not append_mode:
            csv_writer.writerow(
                [
                    "epoch",
                    "train_loss",
                    "perplexity",
                    "learning_rate",
                    "gradient_norm",
                    "update_step",
                    "total_updates",
                ]
            )

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        last_grad_norm = 0.0

        for step, (x, y) in enumerate(dataloader, start=1):
            x = x.to(device)
            y = y.to(device)
            pad_mask = _build_pad_mask(x, pad_idx)

            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits = model(x, pad_mask=pad_mask)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                loss = loss / max(1, grad_accum_steps)

            scaler.scale(loss).backward()

            if step % grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                last_grad_norm = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                global_update_step += 1

            total_loss += float(loss.item()) * max(1, grad_accum_steps)

            if log_interval > 0 and step % log_interval == 0:
                lr_now = optimizer.param_groups[0]["lr"]
                print(
                    f"  step {step}/{len(dataloader)} | "
                    f"loss: {float(loss.item()) * max(1, grad_accum_steps):.4f} | "
                    f"lr: {lr_now:.6e} | grad_norm: {last_grad_norm:.4f}"
                )

        if len(dataloader) % max(1, grad_accum_steps) != 0:
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
            last_grad_norm = float(grad_norm.item() if hasattr(grad_norm, "item") else grad_norm)

            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

            if scheduler is not None:
                scheduler.step()

            global_update_step += 1

        avg_loss = total_loss / max(1, len(dataloader))
        ppl = torch.exp(torch.tensor(avg_loss)).item() if avg_loss < 20 else float("inf")
        lr_now = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch}/{epochs} | loss: {avg_loss:.4f} | ppl: {ppl:.2f} | "
            f"lr: {lr_now:.6e} | grad_norm: {last_grad_norm:.4f} | update_step: {global_update_step}/{total_updates}"
        )
        if csv_writer is not None:
            csv_writer.writerow(
                [
                    int(epoch),
                    f"{avg_loss:.6f}",
                    f"{ppl:.6f}",
                    f"{lr_now:.8e}",
                    f"{last_grad_norm:.6f}",
                    int(global_update_step),
                    int(total_updates),
                ]
            )
            csv_file.flush()

        _save_training_checkpoint(
            save_path=save_path,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            epoch=epoch,
            global_update_step=global_update_step,
            amp_enabled=amp_enabled,
        )

    if csv_file is not None:
        csv_file.close()

    print(f"Model saved to: {save_path}")

def _filter_special_tokens(logits: torch.Tensor, word2ix: Dict[str, int], allow_eop: bool) -> torch.Tensor:
    blocked = {"<START>", "</s>", "<PAD>", "<pad>"}
    if not allow_eop:
        blocked.add("<EOP>")
    for token in blocked:
        idx = word2ix.get(token)
        if idx is not None:
            logits[idx] = -float("inf")
    return logits


def _apply_repetition_penalty(logits: torch.Tensor, generated: list[int], penalty: float) -> torch.Tensor:
    if penalty <= 1.0:
        return logits
    used = set(generated)
    for idx in used:
        if logits[idx] < 0:
            logits[idx] *= penalty
        else:
            logits[idx] /= penalty
    return logits


def _top_k_top_p_filtering(logits: torch.Tensor, top_k: int = 0, top_p: float = 1.0) -> torch.Tensor:
    if top_k > 0:
        values, _ = torch.topk(logits, k=min(top_k, logits.size(-1)))
        min_topk = values[-1]
        logits = torch.where(logits < min_topk, torch.full_like(logits, -float("inf")), logits)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        sorted_mask = cumulative_probs > top_p
        sorted_mask[0] = False
        indices_to_remove = sorted_indices[sorted_mask]
        logits[indices_to_remove] = -float("inf")

    return logits


def _step_meta(content_len: int, line_length: int) -> tuple[int, int]:
    next_pos = content_len + 1
    block = line_length + 1
    line_no = (next_pos - 1) // block + 1
    pos_in_line = (next_pos - 1) % block + 1
    return line_no, pos_in_line


def _forced_token_for_position(
    content_len: int,
    line_length: int,
    num_lines: int,
    comma_idx: int | None,
    period_idx: int | None,
):
    if line_length <= 0 or num_lines <= 0:
        return None

    line_no, pos_in_line = _step_meta(content_len=content_len, line_length=line_length)
    if line_no > num_lines:
        return None

    if pos_in_line != line_length + 1:
        return None

    if line_no in (1, 3):
        return comma_idx
    return period_idx


def _constrain_to_indices(logits: torch.Tensor, allowed_indices: set[int]) -> torch.Tensor:
    if not allowed_indices:
        return logits

    mask = torch.full_like(logits, -float("inf"))
    valid = [idx for idx in allowed_indices if 0 <= idx < logits.numel()]
    if not valid:
        return logits
    mask[valid] = logits[valid]
    return mask


def _block_inner_punctuations(logits: torch.Tensor, word2ix: Dict[str, int]) -> torch.Tensor:
    puncts = {"，", "。", "！", "？", "；", ",", ".", "!", "?", ";"}
    for token in puncts:
        idx = word2ix.get(token)
        if idx is not None:
            logits[idx] = -float("inf")
    return logits


def _constrain_rhyme_seed(logits: torch.Tensor, word2ix: Dict[str, int], rhyme_lexicon: Dict[str, set[str]]) -> torch.Tensor:
    candidates = {word2ix[ch] for ch, group in rhyme_lexicon.items() if len(group) >= 1 and ch in word2ix}
    return _constrain_to_indices(logits, candidates)


def _constrain_rhyme_follow(
    logits: torch.Tensor,
    word2ix: Dict[str, int],
    target_rhyme_char: str,
    rhyme_lexicon: Dict[str, set[str]],
) -> torch.Tensor:
    group = set(rhyme_lexicon.get(target_rhyme_char, set()))
    group.add(target_rhyme_char)
    candidates = {word2ix[ch] for ch in group if ch in word2ix}
    return _constrain_to_indices(logits, candidates)


def _constrain_tone(
    logits: torch.Tensor,
    tone_index: Dict[str, set[int]],
    tone_type: str,
) -> torch.Tensor:
    candidates = set(tone_index.get(tone_type, set()))
    return _constrain_to_indices(logits, candidates)


def generate_poem(
    model,
    start_words: str,
    ix2word: Dict[int, str],
    word2ix: Dict[str, int],
    device: str,
    max_gen_len: int = 128,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 1.0,
    repetition_penalty: float = 1.1,
    line_length: int = 7,
    num_lines: int = 4,
    use_constraints: bool = True,
    use_rhyme_constraint: bool = False,
    rhyme_lexicon: Dict[str, set[str]] | None = None,
    rhyme_lines: tuple[int, ...] = (1, 2, 4),
    acrostic_text: str = "",
    use_tone_constraint: bool = False,
    tone_index: Dict[str, set[int]] | None = None,
):
    model.eval()
    model.to(device)

    start_token = word2ix.get("<START>")
    if start_token is None:
        raise KeyError("word2ix does not contain <START> token")

    comma_idx = word2ix.get("，")
    period_idx = word2ix.get("。")
    eop_idx = word2ix.get("<EOP>")

    prefix = [start_token]
    for ch in start_words:
        if ch not in word2ix:
            raise KeyError(f"Character '{ch}' not in vocabulary")
        prefix.append(word2ix[ch])

    generated = prefix[:]
    selected_rhyme_char = None
    rhyme_lexicon = rhyme_lexicon or {}
    tone_index = tone_index or {"ping": set(), "ze": set()}

    with torch.no_grad():
        for _ in range(max_gen_len):
            max_ctx = model.pos_embed.num_embeddings
            ctx = generated[-max_ctx:]
            input_ids = torch.tensor([ctx], dtype=torch.long, device=device)
            logits = model(input_ids)[0, -1, :]
            logits = logits / max(temperature, 1e-6)

            if use_constraints and line_length > 0 and num_lines > 0:
                total_target_len = num_lines * (line_length + 1)
                allow_eop = (len(generated) - 1) >= total_target_len
            else:
                allow_eop = len(generated) > 8
            logits = _filter_special_tokens(logits, word2ix=word2ix, allow_eop=allow_eop)

            line_no, pos_in_line = _step_meta(content_len=len(generated) - 1, line_length=line_length)
            is_line_end_char = pos_in_line == line_length
            is_line_start_char = pos_in_line == 1

            if use_tone_constraint and is_line_start_char and line_no <= num_lines:
                logits = _constrain_tone(logits, tone_index=tone_index, tone_type="ze")
            if use_tone_constraint and is_line_end_char and line_no <= num_lines:
                logits = _constrain_tone(logits, tone_index=tone_index, tone_type="ping")

            # In constrained mode, forbid punctuation inside a line; punctuation is only allowed at line end.
            if use_constraints and pos_in_line <= line_length:
                logits = _block_inner_punctuations(logits, word2ix=word2ix)

            if use_rhyme_constraint and is_line_end_char and line_no in rhyme_lines:
                if selected_rhyme_char is None:
                    logits = _constrain_rhyme_seed(logits, word2ix=word2ix, rhyme_lexicon=rhyme_lexicon)
                else:
                    logits = _constrain_rhyme_follow(
                        logits,
                        word2ix=word2ix,
                        target_rhyme_char=selected_rhyme_char,
                        rhyme_lexicon=rhyme_lexicon,
                    )

            forced_idx = None

            if is_line_start_char and acrostic_text and line_no <= len(acrostic_text):
                head_ch = acrostic_text[line_no - 1]
                head_idx = word2ix.get(head_ch)
                if head_idx is not None:
                    forced_idx = head_idx

            if forced_idx is None and use_constraints:
                forced_idx = _forced_token_for_position(
                    content_len=len(generated) - 1,
                    line_length=line_length,
                    num_lines=num_lines,
                    comma_idx=comma_idx,
                    period_idx=period_idx,
                )

            if forced_idx is not None:
                next_idx = int(forced_idx)
            else:
                logits = _apply_repetition_penalty(logits, generated, repetition_penalty)
                logits = _top_k_top_p_filtering(logits, top_k=top_k, top_p=top_p)
                probs = torch.softmax(logits, dim=-1)
                next_idx = int(torch.multinomial(probs, num_samples=1).item())

            if eop_idx is not None and next_idx == eop_idx:
                break

            generated.append(next_idx)

            if use_rhyme_constraint and is_line_end_char and line_no in rhyme_lines and selected_rhyme_char is None:
                selected_rhyme_char = ix2word.get(next_idx)

            if use_constraints and line_length > 0 and num_lines > 0:
                total_target_len = num_lines * (line_length + 1)
                if (len(generated) - 1) >= total_target_len:
                    break

    words = [ix2word[idx] for idx in generated[1:]]
    return "".join(words)
