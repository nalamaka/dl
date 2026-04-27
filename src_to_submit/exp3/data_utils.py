from pathlib import Path
import warnings
from typing import Dict, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


SPECIAL_TOKENS = {"<START>", "<EOP>", "</s>", "<PAD>", "<pad>"}
PUNCTUATIONS = {"，", "。", "！", "？", "；", ",", ".", "!", "?", ";"}


def _to_dict(obj):
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Unsupported dict-like object type: {type(obj)}")


def prepareData(npz_path: Path) -> Tuple[torch.Tensor, Dict[str, int], Dict[int, str]]:
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"dtype\(\): align should be passed as Python or NumPy boolean.*",
        )
        data = np.load(npz_path, allow_pickle=True)
        poems = torch.from_numpy(data["data"]).long()
        ix2word_raw = _to_dict(data["ix2word"])
        word2ix_raw = _to_dict(data["word2ix"])

    # Normalize key/value types for stable indexing.
    word2ix = {str(k): int(v) for k, v in word2ix_raw.items()}
    ix2word = {int(k): str(v) for k, v in ix2word_raw.items()}
    return poems, word2ix, ix2word


def build_dataloader(poems: torch.Tensor, batch_size: int) -> DataLoader:
    inputs = poems[:, :-1]
    targets = poems[:, 1:]
    dataset = TensorDataset(inputs, targets)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)


def resolve_pad_index(word2ix: Dict[str, int]):
    for token in ("</s>", "<PAD>", "<pad>"):
        if token in word2ix:
            return word2ix[token]
    return None


def decode_poem_tensor(tokens: torch.Tensor, ix2word: Dict[int, str]) -> str:
    chars = []
    for idx in tokens.tolist():
        token = ix2word.get(int(idx), "")
        if token in SPECIAL_TOKENS:
            continue
        chars.append(token)
    return "".join(chars)


def extract_lines(poem_text: str) -> list[str]:
    lines = []
    buf = []
    for ch in poem_text:
        if ch in PUNCTUATIONS:
            line = "".join(buf).strip()
            if line:
                lines.append(line)
            buf = []
        else:
            buf.append(ch)
    tail = "".join(buf).strip()
    if tail:
        lines.append(tail)
    return lines


def sample_start_words(
    poems: torch.Tensor,
    ix2word: Dict[int, str],
    n_samples: int,
    line_length: int,
) -> list[str]:
    prompts = []
    for row in poems:
        text = decode_poem_tensor(row, ix2word)
        lines = extract_lines(text)
        if not lines:
            continue
        first = lines[0]
        if len(first) >= line_length:
            prompts.append(first[:line_length])
        if len(prompts) >= n_samples:
            break
    return prompts
