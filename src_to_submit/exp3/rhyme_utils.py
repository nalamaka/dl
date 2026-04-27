from collections import defaultdict
from typing import Dict

import torch


PUNCTUATIONS = {"，", "。", "！", "？", "；", ",", ".", "!", "?", ";"}
SPECIAL_TOKENS = {"<START>", "<EOP>", "</s>", "<PAD>", "<pad>"}


class _DisjointSet:
    def __init__(self):
        self.parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self.parent:
            self.parent[x] = x
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, a: str, b: str):
        ra = self.find(a)
        rb = self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def tokens_to_text(tokens: torch.Tensor, ix2word: Dict[int, str]) -> str:
    chars = []
    for idx in tokens.tolist():
        token = ix2word.get(int(idx), "")
        if token in SPECIAL_TOKENS:
            continue
        chars.append(token)
    return "".join(chars)


def split_lines(poem_text: str) -> list[str]:
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


def build_rhyme_lexicon(poems: torch.Tensor, ix2word: Dict[int, str]) -> Dict[str, set[str]]:
    dsu = _DisjointSet()

    for row in poems:
        text = tokens_to_text(row, ix2word)
        lines = split_lines(text)
        if len(lines) < 4:
            continue

        # For quatrain-style rhyme, lines 2 and 4 are primary rhyme positions.
        c2 = lines[1][-1] if lines[1] else ""
        c4 = lines[3][-1] if lines[3] else ""
        if c2 and c4:
            dsu.union(c2, c4)

    groups = defaultdict(set)
    for ch in list(dsu.parent.keys()):
        groups[dsu.find(ch)].add(ch)

    lexicon: Dict[str, set[str]] = {}
    for members in groups.values():
        for ch in members:
            lexicon[ch] = set(members)

    return lexicon
