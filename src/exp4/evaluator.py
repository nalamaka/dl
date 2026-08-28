from __future__ import annotations

import math
from collections import Counter


def _ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    if len(tokens) < n:
        return []
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def sentence_bleu4(reference_tokens: list[str], pred_tokens: list[str]) -> float:
    return corpus_bleu4([reference_tokens], [pred_tokens])


class BLEUAccumulator:
    def __init__(self):
        self.total_ref_len = 0
        self.total_pred_len = 0
        self.clipped_counts = [0, 0, 0, 0]
        self.total_counts = [0, 0, 0, 0]

    def add(self, ref: list[str], pred: list[str]) -> None:
        self.total_ref_len += len(ref)
        self.total_pred_len += len(pred)

        for n in range(1, 5):
            pred_ngrams = Counter(_ngrams(pred, n))
            ref_ngrams = Counter(_ngrams(ref, n))

            self.total_counts[n - 1] += sum(pred_ngrams.values())
            clipped = 0
            for ng, count in pred_ngrams.items():
                clipped += min(count, ref_ngrams.get(ng, 0))
            self.clipped_counts[n - 1] += clipped

    def compute(self) -> float:
        if self.total_pred_len == 0:
            return 0.0

        precisions = []
        for clipped, total in zip(self.clipped_counts, self.total_counts):
            precisions.append((clipped + 1.0) / (total + 1.0))

        if self.total_pred_len > self.total_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1.0 - (self.total_ref_len / max(1, self.total_pred_len)))

        bleu = bp * math.exp(sum(math.log(p) for p in precisions) / 4.0)
        return float(bleu)


def corpus_bleu4(references: list[list[str]], predictions: list[list[str]]) -> float:
    if not references or not predictions:
        return 0.0

    acc = BLEUAccumulator()

    for ref, pred in zip(references, predictions):
        acc.add(ref, pred)

    return acc.compute()
