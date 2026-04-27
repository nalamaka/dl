from __future__ import annotations

from typing import Dict


class ToneAnalyzer:
    def __init__(self):
        self.available = False
        self._lazy_error = None
        self._pinyin = None
        self._style = None

        try:
            from pypinyin import Style, pinyin  # type: ignore

            self._pinyin = pinyin
            self._style = Style
            self.available = True
        except Exception as e:  # pragma: no cover - fallback path
            self._lazy_error = e

    def tone_class(self, ch: str) -> str | None:
        if not self.available or not ch:
            return None

        try:
            py = self._pinyin(ch, style=self._style.TONE3, errors="ignore", strict=False)
            if not py or not py[0]:
                return None
            s = py[0][0]
            if not s:
                return None

            # TONE3 format typically ends with 1~4; neutral or unknown fallback to None.
            tone_num = None
            for c in reversed(s):
                if c.isdigit():
                    tone_num = int(c)
                    break

            if tone_num in (1, 2):
                return "ping"
            if tone_num in (3, 4):
                return "ze"
            return None
        except Exception:
            return None

    def is_ping(self, ch: str) -> bool:
        return self.tone_class(ch) == "ping"

    def is_ze(self, ch: str) -> bool:
        return self.tone_class(ch) == "ze"


def build_tone_index(word2ix: Dict[str, int], analyzer: ToneAnalyzer) -> Dict[str, set[int]]:
    ping = set()
    ze = set()

    if not analyzer.available:
        return {"ping": ping, "ze": ze}

    for ch, idx in word2ix.items():
        if len(ch) != 1:
            continue
        t = analyzer.tone_class(ch)
        if t == "ping":
            ping.add(idx)
        elif t == "ze":
            ze.add(idx)

    return {"ping": ping, "ze": ze}
