from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol
import math

from .config import EmbeddingConfig


class EmbeddingModel(Protocol):
    def get_vector(self, lemma: str) -> list[float] | None:
        ...

    def has_word(self, lemma: str) -> bool:
        ...

    @property
    def dim(self) -> int:
        ...


@dataclass
class KeyedVectorsEmbedding:
    cfg: EmbeddingConfig

    def __post_init__(self) -> None:
        self.vectors: dict[str, list[float]] = {}
        if self.cfg.model_path:
            self._load_vectors(Path(self.cfg.model_path))

    def _load_vectors(self, path: Path) -> None:
        for line in path.read_text().splitlines():
            parts = line.strip().split()
            if not parts:
                continue
            word, *vals = parts
            if len(vals) != self.cfg.vector_dim:
                continue
            vec = [float(v) for v in vals]
            if self.cfg.normalize_vectors:
                norm = math.sqrt(sum(v * v for v in vec))
                if norm > 0:
                    vec = [v / norm for v in vec]
            self.vectors[word] = vec

    @property
    def dim(self) -> int:
        return self.cfg.vector_dim

    def has_word(self, lemma: str) -> bool:
        if lemma in self.vectors:
            return True
        if self.cfg.oov_strategy == "lowercase_fallback":
            return lemma.lower() in self.vectors
        return False

    def _handle_oov(self, lemma: str) -> list[float] | None:
        strategy = self.cfg.oov_strategy
        if strategy == "skip":
            return None
        if strategy == "lowercase_fallback":
            return self.vectors.get(lemma.lower())
        if strategy == "zero":
            return [0.0 for _ in range(self.cfg.vector_dim)]
        return None

    def get_vector(self, lemma: str) -> list[float] | None:
        vec = self.vectors.get(lemma)
        if vec is not None:
            return vec
        return self._handle_oov(lemma)


__all__ = ["EmbeddingModel", "KeyedVectorsEmbedding"]
