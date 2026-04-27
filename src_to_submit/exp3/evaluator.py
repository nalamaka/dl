from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F

from prosody_utils import ToneAnalyzer
from rhyme_utils import split_lines


@dataclass
class PoemMetrics:
    format_score: float
    line_count_ok: float
    line_length_ok: float
    punctuation_ok: float
    rhyme_score: float
    rhyme_124_score: float
    zeqi_pingshou_score: float
    tone_valid_ratio: float
    acrostic_score: float
    distinct1: float
    repetition2: float
    ppl: float


def _distinct1(text: str) -> float:
    chars = [c for c in text if c.strip()]
    if not chars:
        return 0.0
    return len(set(chars)) / len(chars)


def _repetition2(text: str) -> float:
    # Exclude punctuation to focus on semantic repetition.
    puncts = {"?", "?", "?", "?", "?", ",", ".", "!", "?", ";"}
    chars = [c for c in text if c.strip() and c not in puncts]
    if len(chars) < 2:
        return 0.0

    # Char-level repetition ratio: non-unique character proportion.
    char_repeat_ratio = 1.0 - (len(set(chars)) / len(chars))

    # Bigram-level repetition ratio: repeated occurrences over all bigram slots.
    bigrams = ["".join(chars[i:i + 2]) for i in range(len(chars) - 1)]
    counts = Counter(bigrams)
    repeated_occ = sum(cnt - 1 for cnt in counts.values() if cnt > 1)
    bigram_repeat_ratio = repeated_occ / max(1, len(bigrams))

    # Weighted combination; keeps metric sensitive on short poems.
    return 0.4 * char_repeat_ratio + 0.6 * bigram_repeat_ratio


def _format_metrics(poem: str, line_length: int, num_lines: int) -> tuple[float, float, float, float]:
    lines = split_lines(poem)
    line_count_ok = 1.0 if len(lines) == num_lines else 0.0

    if len(lines) == 0:
        line_length_ok = 0.0
    else:
        good = sum(1 for line in lines[:num_lines] if len(line) == line_length)
        line_length_ok = good / max(1, min(len(lines), num_lines))

    punct_count = sum(1 for ch in poem if ch in {"，", "。"})
    punctuation_ok = 1.0 if punct_count >= max(0, num_lines - 1) else punct_count / max(1, num_lines - 1)

    format_score = 0.4 * line_count_ok + 0.4 * line_length_ok + 0.2 * punctuation_ok
    return format_score, line_count_ok, line_length_ok, punctuation_ok


def _same_rhyme(a: str, b: str, rhyme_lexicon: Dict[str, set[str]]) -> bool:
    if not a or not b:
        return False
    if a == b:
        return True
    group = rhyme_lexicon.get(a)
    return bool(group and b in group)


def _rhyme_score_by_lines(poem: str, rhyme_lexicon: Dict[str, set[str]], rhyme_lines: tuple[int, ...]) -> float:
    lines = split_lines(poem)
    idxs = [i for i in rhyme_lines if 1 <= i <= len(lines)]
    if len(idxs) < 2:
        return 0.0

    ref = lines[idxs[0] - 1][-1] if lines[idxs[0] - 1] else ""
    if not ref:
        return 0.0

    total = 0
    hit = 0
    for i in idxs[1:]:
        ch = lines[i - 1][-1] if lines[i - 1] else ""
        total += 1
        if _same_rhyme(ref, ch, rhyme_lexicon):
            hit += 1

    return hit / max(1, total)


def _zeqi_pingshou_metrics(poem: str, analyzer: ToneAnalyzer, num_lines: int) -> tuple[float, float]:
    lines = split_lines(poem)[:num_lines]
    if not lines:
        return 0.0, 0.0

    total_checks = len(lines) * 2
    known_checks = 0
    hit = 0

    for line in lines:
        if not line:
            continue
        head = line[0]
        tail = line[-1]

        t_head = analyzer.tone_class(head)
        if t_head is not None:
            known_checks += 1
            if t_head == "ze":
                hit += 1

        t_tail = analyzer.tone_class(tail)
        if t_tail is not None:
            known_checks += 1
            if t_tail == "ping":
                hit += 1

    zeqi_pingshou_score = hit / max(1, known_checks)
    tone_valid_ratio = known_checks / max(1, total_checks)
    return zeqi_pingshou_score, tone_valid_ratio


def _acrostic_score(poem: str, acrostic_text: str, num_lines: int) -> float:
    if not acrostic_text:
        return 0.0

    lines = split_lines(poem)[:num_lines]
    if not lines:
        return 0.0

    total = min(len(lines), len(acrostic_text))
    if total <= 0:
        return 0.0

    hit = 0
    for i in range(total):
        if lines[i] and lines[i][0] == acrostic_text[i]:
            hit += 1
    return hit / total


def _poem_ppl(model, poem: str, word2ix: Dict[str, int], device: str) -> float:
    start_idx = word2ix.get("<START>")
    eop_idx = word2ix.get("<EOP>")
    if start_idx is None or eop_idx is None:
        return float("nan")

    ids = [start_idx]
    for ch in poem:
        idx = word2ix.get(ch)
        if idx is None:
            continue
        ids.append(idx)
    ids.append(eop_idx)

    if len(ids) < 3:
        return float("nan")

    x = torch.tensor([ids[:-1]], dtype=torch.long, device=device)
    y = torch.tensor([ids[1:]], dtype=torch.long, device=device)

    max_ctx = model.pos_embed.num_embeddings
    if x.size(1) > max_ctx:
        x = x[:, -max_ctx:]
        y = y[:, -max_ctx:]

    with torch.no_grad():
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
    return float(torch.exp(loss).item())


def evaluate_poem(
    poem: str,
    model,
    word2ix: Dict[str, int],
    rhyme_lexicon: Dict[str, set[str]],
    analyzer: ToneAnalyzer,
    device: str,
    line_length: int,
    num_lines: int,
    rhyme_lines: tuple[int, ...] = (1, 2, 4),
    acrostic_text: str = "",
) -> PoemMetrics:
    format_score, line_count_ok, line_length_ok, punctuation_ok = _format_metrics(
        poem,
        line_length=line_length,
        num_lines=num_lines,
    )

    rhyme_score = _rhyme_score_by_lines(poem, rhyme_lexicon, rhyme_lines)
    rhyme_124_score = _rhyme_score_by_lines(poem, rhyme_lexicon, (1, 2, 4))
    zeqi_pingshou_score, tone_valid_ratio = _zeqi_pingshou_metrics(poem, analyzer, num_lines=num_lines)

    metrics = PoemMetrics(
        format_score=format_score,
        line_count_ok=line_count_ok,
        line_length_ok=line_length_ok,
        punctuation_ok=punctuation_ok,
        rhyme_score=rhyme_score,
        rhyme_124_score=rhyme_124_score,
        zeqi_pingshou_score=zeqi_pingshou_score,
        tone_valid_ratio=tone_valid_ratio,
        acrostic_score=_acrostic_score(poem, acrostic_text=acrostic_text, num_lines=num_lines),
        distinct1=_distinct1(poem),
        repetition2=_repetition2(poem),
        ppl=_poem_ppl(model=model, poem=poem, word2ix=word2ix, device=device),
    )
    return metrics


def summarize_metrics(metrics_list: list[PoemMetrics]) -> Dict[str, float]:
    if not metrics_list:
        return {}

    out = {}
    keys = list(PoemMetrics.__dataclass_fields__.keys())
    for k in keys:
        vals = [getattr(m, k) for m in metrics_list if getattr(m, k) == getattr(m, k)]
        out[k] = sum(vals) / max(1, len(vals))

    out["quality_score"] = (
        0.2 * out.get("format_score", 0.0)
        + 0.2 * out.get("rhyme_score", 0.0)
        + 0.15 * out.get("zeqi_pingshou_score", 0.0)
        + 0.1 * out.get("tone_valid_ratio", 0.0)
        + 0.05 * out.get("acrostic_score", 0.0)
        + 0.15 * out.get("distinct1", 0.0)
        + 0.1 * (1.0 - out.get("repetition2", 0.0))
        + 0.05 * (1.0 / max(1.0, out.get("ppl", 1.0)))
    )
    return out
