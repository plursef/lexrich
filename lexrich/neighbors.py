from __future__ import annotations

from typing import Protocol

from .config import NeighborConfig


class NeighborIndex(Protocol):
    def build(self, vectors: dict[str, list[float]]) -> None:
        ...

    def query(self, vec: list[float], top_k: int) -> list[tuple[str, float]]:
        ...


class ExactNeighborIndex:
    def __init__(self, cfg: NeighborConfig):
        self.cfg = cfg
        self.vocab: list[str] = []
        self.matrix: list[list[float]] = []

    def build(self, vectors: dict[str, list[float]]) -> None:
        self.vocab = list(vectors.keys())
        self.matrix = [vectors[w] for w in self.vocab]

    def _dot(self, a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    def query(self, vec: list[float], top_k: int) -> list[tuple[str, float]]:
        if not self.matrix:
            return []
        sims = [self._dot(row, vec) for row in self.matrix]
        pairs = list(zip(self.vocab, sims))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:top_k]


__all__ = ["NeighborIndex", "ExactNeighborIndex"]
