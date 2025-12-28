from __future__ import annotations

from dataclasses import dataclass
from collections import Counter, defaultdict
from typing import Iterable

from .config import GroupingConfig
from .preprocess import Token


@dataclass
class CorpusStats:
    counts: dict[str, int]
    pos_counts: dict[str, dict[str, int]]
    total_tokens: int
    total_types: int


class VocabularyExtractor:
    def __init__(self, cfg: GroupingConfig):
        self.cfg = cfg

    def extract(self, tokens: Iterable[Token]) -> CorpusStats:
        counts: Counter[str] = Counter()
        pos_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for tok in tokens:
            counts[tok.lemma] += 1
            if tok.pos:
                pos_counts[tok.lemma][tok.pos] += 1
        total_tokens = sum(counts.values())
        # Apply frequency constraints
        vocab = [lemma for lemma, freq in counts.most_common() if freq >= self.cfg.min_freq]
        if len(vocab) > self.cfg.max_vocab:
            vocab = vocab[: self.cfg.max_vocab]
        filtered_counts = {lemma: counts[lemma] for lemma in vocab}
        filtered_pos_counts = {lemma: dict(pos_counts[lemma]) for lemma in vocab}
        return CorpusStats(
            counts=filtered_counts,
            pos_counts=filtered_pos_counts,
            total_tokens=total_tokens,
            total_types=len(filtered_counts),
        )


__all__ = ["CorpusStats", "VocabularyExtractor"]
