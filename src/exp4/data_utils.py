from __future__ import annotations

import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

try:
    import sentencepiece as spm  # type: ignore
except Exception:
    spm = None


PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


def _read_lines_with_fallback(path: Path) -> list[str]:
    encodings = ("utf-8", "gb18030", "gbk")
    for enc in encodings:
        try:
            with path.open("r", encoding=enc) as f:
                return [line.rstrip("\n").replace("\ufeff", "") for line in f]
        except UnicodeDecodeError:
            continue
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        return [line.rstrip("\n").replace("\ufeff", "") for line in f]


class BaseTokenizer:
    name = "base"

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def detokenize(self, tokens: list[str]) -> str:
        raise NotImplementedError


class LegacySpaceTokenizer(BaseTokenizer):
    name = "legacy"

    def tokenize(self, text: str) -> list[str]:
        return [tok for tok in text.strip().split() if tok]

    def detokenize(self, tokens: list[str]) -> str:
        return " ".join(tokens)


class SentencePieceTokenizer(BaseTokenizer):
    def __init__(self, model_path: Path):
        if spm is None:
            raise ImportError(
                "sentencepiece is not installed. Install via: pip install sentencepiece"
            )
        if not model_path.exists():
            raise FileNotFoundError(f"SentencePiece model not found: {model_path}")
        self.model_path = model_path
        self.name = f"spm_{model_path.stem}"
        self.processor = spm.SentencePieceProcessor(model_file=str(model_path))

    def tokenize(self, text: str) -> list[str]:
        return list(self.processor.encode(text, out_type=str))

    def detokenize(self, tokens: list[str]) -> str:
        return str(self.processor.decode(tokens))


def ensure_sentencepiece_model(
    model_path: Path,
    input_paths: list[Path],
    vocab_size: int = 16000,
    model_type: str = "bpe",
    character_coverage: float = 1.0,
) -> Path:
    if model_path.exists():
        return model_path
    if spm is None:
        raise ImportError(
            "sentencepiece is not installed. Install via: pip install sentencepiece"
        )

    for p in input_paths:
        if not p.exists():
            raise FileNotFoundError(f"SentencePiece training input missing: {p}")

    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_prefix = str(model_path.with_suffix(""))
    input_arg = ",".join(str(p) for p in input_paths)
    spm.SentencePieceTrainer.train(
        input=input_arg,
        model_prefix=model_prefix,
        vocab_size=int(vocab_size),
        model_type=str(model_type),
        character_coverage=float(character_coverage),
        bos_id=-1,
        eos_id=-1,
        pad_id=-1,
        unk_id=0,
        train_extremely_large_corpus=False,
    )
    return model_path


def build_tokenizer(
    tokenizer_type: str = "legacy",
    spm_model_path: Path | None = None,
    spm_train_files: list[Path] | None = None,
    spm_vocab_size: int = 16000,
    spm_model_type: str = "bpe",
    spm_character_coverage: float = 1.0,
    auto_train_spm: bool = False,
) -> BaseTokenizer:
    if tokenizer_type == "legacy":
        return LegacySpaceTokenizer()
    if tokenizer_type != "spm":
        raise ValueError(f"Unsupported tokenizer_type: {tokenizer_type}")

    if spm_model_path is None:
        raise ValueError("spm_model_path is required when tokenizer_type=spm")
    if auto_train_spm:
        if not spm_train_files:
            raise ValueError("spm_train_files are required when auto_train_spm=True")
        ensure_sentencepiece_model(
            model_path=spm_model_path,
            input_paths=spm_train_files,
            vocab_size=spm_vocab_size,
            model_type=spm_model_type,
            character_coverage=spm_character_coverage,
        )
    return SentencePieceTokenizer(model_path=spm_model_path)


def get_tokenizer_tag(tokenizer_type: str, spm_model_path: Path | None = None) -> str:
    if tokenizer_type == "legacy":
        return "legacy"
    if tokenizer_type == "spm":
        if spm_model_path is None:
            return "spm"
        return f"spm_{spm_model_path.stem}"
    return tokenizer_type


def tokenize(text: str, tokenizer: BaseTokenizer | None = None) -> list[str]:
    if tokenizer is None:
        return LegacySpaceTokenizer().tokenize(text)
    return tokenizer.tokenize(text)


def detokenize(tokens: list[str], tokenizer: BaseTokenizer | None = None) -> str:
    if tokenizer is None:
        return LegacySpaceTokenizer().detokenize(tokens)
    return tokenizer.detokenize(tokens)


def load_parallel_data(src_path: Path, tgt_path: Path, max_samples: int | None = None) -> list[tuple[str, str]]:
    src_lines = _read_lines_with_fallback(src_path)
    tgt_lines = _read_lines_with_fallback(tgt_path)

    n = min(len(src_lines), len(tgt_lines))
    pairs: list[tuple[str, str]] = []
    for i in range(n):
        src = src_lines[i].strip()
        tgt = tgt_lines[i].strip()
        if not src or not tgt:
            continue
        pairs.append((src, tgt))

    if max_samples is not None:
        pairs = pairs[:max_samples]
    return pairs


def split_train_val(
    pairs: list[tuple[str, str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    if not pairs:
        return [], []

    rng = random.Random(seed)
    shuffled = list(pairs)
    rng.shuffle(shuffled)
    val_size = max(1, int(len(shuffled) * val_ratio))
    val_pairs = shuffled[:val_size]
    train_pairs = shuffled[val_size:]
    return train_pairs, val_pairs


@dataclass
class Vocab:
    itos: list[str]

    def __post_init__(self):
        self.stoi = {tok: idx for idx, tok in enumerate(self.itos)}

    @property
    def pad_idx(self) -> int:
        return self.stoi[PAD_TOKEN]

    @property
    def bos_idx(self) -> int:
        return self.stoi[BOS_TOKEN]

    @property
    def eos_idx(self) -> int:
        return self.stoi[EOS_TOKEN]

    @property
    def unk_idx(self) -> int:
        return self.stoi[UNK_TOKEN]

    def token_to_id(self, token: str) -> int:
        return self.stoi.get(token, self.unk_idx)

    def id_to_token(self, idx: int) -> str:
        if 0 <= idx < len(self.itos):
            return self.itos[idx]
        return UNK_TOKEN

    def encode(
        self,
        tokens: list[str],
        add_bos: bool = False,
        add_eos: bool = False,
        max_len: int | None = None,
    ) -> list[int]:
        ids = [self.token_to_id(tok) for tok in tokens]
        if add_bos:
            ids = [self.bos_idx] + ids
        if add_eos:
            ids = ids + [self.eos_idx]
        if max_len is not None:
            ids = ids[:max_len]
            if add_eos and ids and ids[-1] != self.eos_idx:
                ids[-1] = self.eos_idx
        return ids

    def decode(self, ids: Iterable[int], skip_special: bool = True) -> list[str]:
        out: list[str] = []
        specials = set(SPECIAL_TOKENS)
        for idx in ids:
            tok = self.id_to_token(int(idx))
            if tok == EOS_TOKEN:
                break
            if skip_special and tok in specials:
                continue
            out.append(tok)
        return out

    def to_dict(self) -> dict:
        return {"itos": self.itos}

    @staticmethod
    def from_dict(d: dict) -> "Vocab":
        return Vocab(itos=list(d["itos"]))


def build_vocab(
    tokenized_sentences: list[list[str]],
    max_size: int,
    min_freq: int,
) -> Vocab:
    counter = Counter()
    for toks in tokenized_sentences:
        counter.update(toks)

    items = [
        (tok, freq)
        for tok, freq in counter.items()
        if freq >= min_freq and tok not in SPECIAL_TOKENS
    ]
    items.sort(key=lambda x: (-x[1], x[0]))

    limit = max(0, max_size - len(SPECIAL_TOKENS))
    vocab_tokens = SPECIAL_TOKENS + [tok for tok, _ in items[:limit]]
    return Vocab(vocab_tokens)


def save_vocab(vocab: Vocab, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(vocab.to_dict(), f, ensure_ascii=False)


def load_vocab(path: Path) -> Vocab:
    with path.open("r", encoding="utf-8") as f:
        return Vocab.from_dict(json.load(f))


class TranslationDataset(Dataset):
    def __init__(
        self,
        pairs: list[tuple[str, str]],
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_src_len: int,
        max_tgt_len: int,
        src_tokenizer: BaseTokenizer | None = None,
        tgt_tokenizer: BaseTokenizer | None = None,
    ):
        self.src_ids: list[torch.Tensor] = []
        self.tgt_ids: list[torch.Tensor] = []

        for src, tgt in pairs:
            src_toks = tokenize(src, tokenizer=src_tokenizer)
            tgt_toks = tokenize(tgt, tokenizer=tgt_tokenizer)

            src_enc = src_vocab.encode(src_toks, add_bos=False, add_eos=True, max_len=max_src_len)
            tgt_enc = tgt_vocab.encode(tgt_toks, add_bos=True, add_eos=True, max_len=max_tgt_len)

            if len(src_enc) < 2 or len(tgt_enc) < 3:
                continue
            self.src_ids.append(torch.tensor(src_enc, dtype=torch.long))
            self.tgt_ids.append(torch.tensor(tgt_enc, dtype=torch.long))

    def __len__(self) -> int:
        return len(self.src_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.src_ids[index], self.tgt_ids[index]


def build_collate_fn(src_pad_idx: int, tgt_pad_idx: int):
    def _collate(batch: list[tuple[torch.Tensor, torch.Tensor]]):
        src_list = [x[0] for x in batch]
        tgt_list = [x[1] for x in batch]
        src = pad_sequence(src_list, batch_first=True, padding_value=src_pad_idx)
        tgt = pad_sequence(tgt_list, batch_first=True, padding_value=tgt_pad_idx)
        return src, tgt

    return _collate


def load_test_source(test_src_path: Path) -> list[str]:
    return [line.strip() for line in _read_lines_with_fallback(test_src_path) if line.strip()]


def parse_reference_pairs(reference_path: Path) -> list[tuple[str, str]]:
    # Reference file is ordered as: zh_line, en_line, zh_line, en_line, ...
    lines = [line.strip() for line in _read_lines_with_fallback(reference_path) if line.strip()]
    if len(lines) % 2 != 0:
        lines = lines[:-1]

    pairs = []
    for i in range(0, len(lines), 2):
        pairs.append((lines[i], lines[i + 1]))
    return pairs
