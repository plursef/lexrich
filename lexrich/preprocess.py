from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Iterable

from .config import PreprocessConfig


@dataclass(frozen=True)
class Token:
    text: str
    lemma: str
    pos: str | None
    start: int | None
    end: int | None


class TextPreprocessor:
    """Lightweight text preprocessing with rule-based fallbacks.

    This implementation intentionally avoids heavyweight NLP dependencies to
    keep the package easy to run in constrained environments. It performs
    lowercasing, naive lemmatization, rule-based POS tagging, stopword removal
    and length filtering according to :class:`PreprocessConfig`.
    """

    WORD_RE = re.compile(r"[A-Za-z']+")

    def __init__(self, cfg: PreprocessConfig):
        self.cfg = cfg
        self.stopwords = self._load_stopwords()

    def _load_stopwords(self) -> set[str]:
        stopwords_path = Path(__file__).with_suffix("").parent / "resources" / "stopwords.txt"
        if stopwords_path.exists():
            return {w.strip().lower() for w in stopwords_path.read_text().splitlines() if w.strip()}
        # Minimal fallback set
        return {"the", "and", "a", "an", "in", "on", "of", "for", "to", "with", "is", "are"}

    def _simple_pos(self, token: str) -> str:
        # Extremely naive rule-based POS tagging to satisfy config gates.
        lower = token.lower()
        if lower.endswith("ing") or lower.endswith("ed"):
            return "VERB"
        if lower.endswith("ly"):
            return "ADV"
        if lower.endswith("ous") or lower.endswith("ful") or lower.endswith("able"):
            return "ADJ"
        return "NOUN"

    def _lemmatize(self, token: str, pos: str) -> str:
        if not self.cfg.lemmatize:
            return token
        lower = token.lower()
        if pos == "VERB" and lower.endswith("ing"):
            return lower[:-3]
        if pos == "VERB" and lower.endswith("ed"):
            return lower[:-2]
        if pos == "NOUN" and lower.endswith("s") and len(lower) > 3:
            return lower[:-1]
        return lower

    def process(self, text: str) -> list[Token]:
        tokens: list[Token] = []
        for match in self.WORD_RE.finditer(text):
            raw = match.group(0)
            if self.cfg.lowercase:
                raw = raw.lower()
            pos = self._simple_pos(raw) if self.cfg.pos_tag else None
            lemma = self._lemmatize(raw, pos or "") if self.cfg.lemmatize else raw
            if len(lemma) < self.cfg.min_token_len:
                continue
            if self.cfg.remove_stopwords and lemma in self.stopwords:
                continue
            if self.cfg.keep_pos and pos and pos not in self.cfg.keep_pos:
                continue
            tokens.append(Token(text=raw, lemma=lemma, pos=pos, start=match.start(), end=match.end()))
        return tokens


__all__ = ["Token", "TextPreprocessor"]
