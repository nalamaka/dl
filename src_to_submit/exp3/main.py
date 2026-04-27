import argparse
import json
from pathlib import Path

import torch

from config import Config
from data_utils import build_dataloader, prepareData, resolve_pad_index, sample_start_words
from engine import generate_poem, train_model
from evaluator import evaluate_poem, summarize_metrics
from model import PoetryModel
from prosody_utils import ToneAnalyzer, build_tone_index
from rhyme_utils import build_rhyme_lexicon


PUNCTS = {"，", "。", "！", "？", "；", ",", ".", "!", "?", ";"}


def parse_args():
    parser = argparse.ArgumentParser(description="Exp3: Automatic Chinese Poetry Generation")
    parser.add_argument("--mode", choices=["train", "generate", "evaluate"], required=True)
    parser.add_argument("--data_path", type=str, default=str(Config.data_path))
    parser.add_argument("--ckpt", type=str, default="./checkpoints/poetry_transformer.pth")
    parser.add_argument("--log_dir", type=str, default="./log")
    parser.add_argument("--history_file", type=str, default="")
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    parser.add_argument("--resume_from", type=str, default="", help="Path to checkpoint for resuming training")

    parser.add_argument("--epochs", type=int, default=Config.epochs)
    parser.add_argument("--batch_size", type=int, default=Config.batch_size)
    parser.add_argument("--grad_accum_steps", type=int, default=Config.grad_accum_steps)
    parser.add_argument("--lr", type=float, default=Config.lr)
    parser.add_argument("--weight_decay", type=float, default=Config.weight_decay)
    parser.add_argument("--warmup_ratio", type=float, default=Config.warmup_ratio)
    parser.add_argument("--min_lr_ratio", type=float, default=Config.min_lr_ratio)
    parser.add_argument("--scheduler_type", choices=["cosine", "linear", "none"], default=Config.scheduler_type)
    parser.add_argument("--max_grad_norm", type=float, default=Config.max_grad_norm)
    parser.add_argument("--use_amp", action="store_true", default=Config.use_amp)
    parser.add_argument("--no_amp", action="store_true")
    parser.add_argument("--log_interval", type=int, default=Config.log_interval)

    parser.add_argument("--start_words", type=str, default="")
    parser.add_argument("--acrostic_text", type=str, default="")
    parser.add_argument("--max_gen_len", type=int, default=Config.max_gen_len)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--top_k", type=int, default=40)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.15)

    # line_length is optional now: infer from prompt when punctuation exists, else fallback to 7.
    parser.add_argument("--line_length", type=int, default=None)
    parser.add_argument("--num_lines", type=int, default=4)

    parser.add_argument("--use_constraints", action="store_true")
    parser.add_argument("--no_constraints", action="store_true")
    parser.add_argument("--use_rhyme_constraint", action="store_true")
    parser.add_argument("--no_rhyme_constraint", action="store_true")

    parser.add_argument("--use_tone_constraint", action="store_true")
    parser.add_argument("--rhyme_lines", type=str, default="1,2,4")

    parser.add_argument("--eval_samples", type=int, default=Config.eval_samples)
    parser.add_argument("--prompts_file", type=str, default="")
    parser.add_argument("--eval_output", type=str, default="./log/eval_results.jsonl")
    parser.add_argument("--verbose_eval", action="store_true")

    # Better defaults for generation-time quality control.
    parser.set_defaults(use_constraints=True, use_rhyme_constraint=True)
    return parser.parse_args()


def _split_prompt_lines(prompt: str) -> list[str]:
    lines = []
    buf = []
    for ch in prompt:
        if ch in PUNCTS:
            s = "".join(buf).strip()
            if s:
                lines.append(s)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        lines.append(tail)
    return lines


def _infer_line_length(start_words: str, user_line_length: int | None, max_line_length: int) -> int:
    if user_line_length is not None:
        if user_line_length <= 0:
            raise ValueError("--line_length must be a positive integer")
        if user_line_length > max_line_length:
            raise ValueError(
                f"--line_length={user_line_length} exceeds max allowed line length {max_line_length}"
            )

    # Only trigger intelligent inference when the prompt contains explicit punctuation hints.
    has_hint = ("，" in start_words) or ("。" in start_words)
    if not has_hint:
        return user_line_length if user_line_length is not None else 7

    lines = _split_prompt_lines(start_words)
    if not lines:
        return user_line_length if user_line_length is not None else 7

    lengths = [len(x) for x in lines]
    uniq = sorted(set(lengths))
    if len(uniq) != 1:
        raise ValueError(
            f"Invalid prompt: inconsistent line lengths {lengths}. "
            "Please ensure all provided lines have the same length."
        )

    inferred = uniq[0]
    if inferred > max_line_length:
        raise ValueError(
            f"Invalid prompt: inferred line length {inferred} exceeds max allowed {max_line_length}."
        )

    if user_line_length is not None and user_line_length != inferred:
        raise ValueError(
            f"Prompt implies line_length={inferred}, but user specified --line_length={user_line_length}."
        )

    return inferred


def parse_rhyme_lines(raw: str, num_lines: int) -> tuple[int, ...]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    nums = tuple(int(p) for p in parts)
    if len(nums) < 2:
        raise ValueError("--rhyme_lines must provide at least two line numbers, e.g. 1,2,4")

    nums = tuple(sorted(set(nums)))
    if nums[0] < 1:
        raise ValueError("--rhyme_lines must be >= 1")
    if nums[-1] > num_lines:
        raise ValueError(
            f"--rhyme_lines contains line {nums[-1]}, but --num_lines={num_lines}."
        )
    return nums


def select_device():
    if Config.device == "cuda" and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def build_model(vocab_size: int):
    return PoetryModel(
        vocab_size=vocab_size,
        d_model=Config.d_model,
        nhead=Config.nhead,
        num_layers=Config.num_layers,
        dim_feedforward=Config.dim_feedforward,
        dropout=Config.dropout,
        max_seq_len=Config.max_seq_len,
        pattern_cycle=Config.pattern_cycle,
    )


def _load_prompts(args, poems, ix2word):
    if args.prompts_file:
        prompts = []
        with open(args.prompts_file, "r", encoding="utf-8") as f:
            for line in f:
                s = line.strip()
                if s:
                    prompts.append(s)
        if prompts:
            return prompts[: args.eval_samples]

    return sample_start_words(
        poems=poems,
        ix2word=ix2word,
        n_samples=args.eval_samples,
        line_length=args.line_length,
    )


def load_model_weights(model, ckpt_path: str, device: str):
    state = torch.load(ckpt_path, map_location=device)
    model_state = state.get("model_state_dict", state) if isinstance(state, dict) else state
    result = model.load_state_dict(model_state, strict=False)

    missing = list(result.missing_keys)
    unexpected = list(result.unexpected_keys)
    if missing or unexpected:
        print("[warning] Checkpoint architecture mismatch detected.")
        if missing:
            print(f"[warning] Missing keys: {missing}")
        if unexpected:
            print(f"[warning] Unexpected keys: {unexpected}")
        print("[warning] Model loaded with strict=False. Retraining is recommended for best quality.")


def main():
    args = parse_args()
    device = select_device()

    if args.no_amp:
        args.use_amp = False
    if args.no_constraints:
        args.use_constraints = False
    if args.no_rhyme_constraint:
        args.use_rhyme_constraint = False

    if args.num_lines <= 0:
        raise ValueError("--num_lines must be a positive integer")

    args.line_length = _infer_line_length(
        start_words=args.start_words,
        user_line_length=args.line_length,
        max_line_length=Config.max_line_length,
    )

    rhyme_lines = parse_rhyme_lines(args.rhyme_lines, num_lines=args.num_lines)

    if args.acrostic_text and len(args.acrostic_text) > args.num_lines:
        raise ValueError(
            f"acrostic_text length ({len(args.acrostic_text)}) exceeds --num_lines ({args.num_lines})."
        )

    if args.mode == "generate":
        # Smart default for acrostic generation: derive the first prefix char from acrostic text.
        if not args.start_words:
            if args.acrostic_text:
                args.start_words = args.acrostic_text[0]
            else:
                args.start_words = "\u6e56\u5149\u79cb\u6708\u4e24\u76f8\u548c"

        # Keep start_words and acrostic header consistent on first character.
        if args.acrostic_text and args.start_words and args.start_words[0] != args.acrostic_text[0]:
            print(
                f"[warning] start_words first char '{args.start_words[0]}' "
                f"!= acrostic first char '{args.acrostic_text[0]}'. Auto-correcting."
            )
            args.start_words = args.acrostic_text[0] + args.start_words[1:]

    poems, word2ix, ix2word = prepareData(Path(args.data_path))
    model = build_model(len(word2ix))

    if args.mode == "train":
        dataloader = build_dataloader(poems, args.batch_size)
        pad_idx = resolve_pad_index(word2ix)
        history_path = Path(args.history_file) if args.history_file else Path(args.log_dir) / "history.csv"
        resume_path = None
        if args.resume_from:
            resume_path = Path(args.resume_from)
        elif args.resume:
            resume_path = Path(args.ckpt)

        train_model(
            model=model,
            dataloader=dataloader,
            epochs=args.epochs,
            lr=args.lr,
            device=device,
            save_path=Path(args.ckpt),
            pad_idx=pad_idx,
            grad_accum_steps=args.grad_accum_steps,
            weight_decay=args.weight_decay,
            max_grad_norm=args.max_grad_norm,
            warmup_ratio=args.warmup_ratio,
            min_lr_ratio=args.min_lr_ratio,
            scheduler_type=args.scheduler_type,
            use_amp=args.use_amp,
            log_interval=args.log_interval,
            resume_from=resume_path,
            history_path=history_path,
        )
        return

    load_model_weights(model, args.ckpt, device)
    rhyme_lexicon = build_rhyme_lexicon(poems=poems, ix2word=ix2word)
    tone_analyzer = ToneAnalyzer()
    tone_index = build_tone_index(word2ix=word2ix, analyzer=tone_analyzer)
    if not tone_analyzer.available:
        print("[warning] pypinyin is not installed; tone metrics/constraints are unavailable.")

    if args.mode == "generate":
        poem = generate_poem(
            model=model,
            start_words=args.start_words,
            ix2word=ix2word,
            word2ix=word2ix,
            device=device,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            line_length=args.line_length,
            num_lines=args.num_lines,
            use_constraints=args.use_constraints,
            use_rhyme_constraint=args.use_rhyme_constraint,
            rhyme_lexicon=rhyme_lexicon,
            rhyme_lines=rhyme_lines,
            acrostic_text=args.acrostic_text,
            use_tone_constraint=args.use_tone_constraint,
            tone_index=tone_index,
        )
        print(poem)
        return

    prompts = _load_prompts(args, poems=poems, ix2word=ix2word)
    if not prompts:
        raise RuntimeError("No prompts available for evaluation.")

    outputs = []
    metrics = []
    for prompt in prompts:
        poem = generate_poem(
            model=model,
            start_words=prompt,
            ix2word=ix2word,
            word2ix=word2ix,
            device=device,
            max_gen_len=args.max_gen_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            line_length=args.line_length,
            num_lines=args.num_lines,
            use_constraints=args.use_constraints,
            use_rhyme_constraint=args.use_rhyme_constraint,
            rhyme_lexicon=rhyme_lexicon,
            rhyme_lines=rhyme_lines,
            acrostic_text=args.acrostic_text,
            use_tone_constraint=args.use_tone_constraint,
            tone_index=tone_index,
        )
        m = evaluate_poem(
            poem=poem,
            model=model,
            word2ix=word2ix,
            rhyme_lexicon=rhyme_lexicon,
            analyzer=tone_analyzer,
            device=device,
            line_length=args.line_length,
            num_lines=args.num_lines,
            rhyme_lines=rhyme_lines,
            acrostic_text=args.acrostic_text,
        )
        metric_dict = dict(m.__dict__)
        metric_dict["quality_score"] = summarize_metrics([m]).get("quality_score", 0.0)
        outputs.append({"prompt": prompt, "poem": poem, "metrics": metric_dict})
        metrics.append(m)

        if args.verbose_eval:
            print(f"[prompt] {prompt}")
            print(f"[poem]   {poem}")
            print(f"[score]  {m.__dict__}")
            print()

    summary = summarize_metrics(metrics)
    print("Evaluation Summary:")
    for k, v in summary.items():
        print(f"- {k}: {v:.4f}")

    if args.eval_output:
        out_path = Path(args.eval_output)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            for item in outputs:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"Saved detailed evaluation to: {out_path}")


if __name__ == "__main__":
    main()
