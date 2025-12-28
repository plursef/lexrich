from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Set

from .config import GroupingConfig, NeighborConfig
from .neighbors import NeighborIndex


@dataclass
class Cluster:
    id: int
    members: set[str]


@dataclass
class GroupResult:
    clusters: list[Cluster]
    neighborhoods: dict[str, set[str]]
    word_to_cluster: dict[str, int]


class UnionFind:
    def __init__(self, elements: list[str]):
        self.parent = {e: e for e in elements}
        self.rank = {e: 0 for e in elements}

    def find(self, x: str) -> str:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]

    def union(self, x: str, y: str) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx == ry:
            return
        if self.rank[rx] < self.rank[ry]:
            self.parent[rx] = ry
        elif self.rank[rx] > self.rank[ry]:
            self.parent[ry] = rx
        else:
            self.parent[ry] = rx
            self.rank[rx] += 1


class SimilarityGrouper:
    def __init__(self, cfg: GroupingConfig, neighbor_cfg: NeighborConfig):
        self.cfg = cfg
        self.neighbor_cfg = neighbor_cfg

    def group(
        self, vocab: list[str], vectors: dict[str, list[float]], index: NeighborIndex
    ) -> GroupResult:
        if self.cfg.mode == "threshold_graph":
            return self._group_threshold_graph(vocab, vectors, index)
        if self.cfg.mode == "topk_neighborhood":
            return self._group_topk_neighborhood(vocab, vectors, index)
        raise ValueError(f"Unknown grouping mode: {self.cfg.mode}")

    def _group_threshold_graph(
        self, vocab: list[str], vectors: dict[str, list[float]], index: NeighborIndex
    ) -> GroupResult:
        uf = UnionFind(vocab)
        for word in vocab:
            vec = vectors[word]
            neighbors = index.query(vec, self.neighbor_cfg.top_k)
            for nb_word, sim in neighbors:
                if nb_word == word:
                    continue
                if sim >= self.cfg.threshold:
                    uf.union(word, nb_word)
        cluster_map: dict[str, list[str]] = {}
        for word in vocab:
            root = uf.find(word)
            cluster_map.setdefault(root, []).append(word)
        clusters = [Cluster(id=i, members=set(words)) for i, words in enumerate(cluster_map.values())]
        word_to_cluster = {word: cid for cid, cluster in enumerate(clusters) for word in cluster.members}
        return GroupResult(clusters=clusters, neighborhoods={}, word_to_cluster=word_to_cluster)

    def _group_topk_neighborhood(
        self, vocab: list[str], vectors: dict[str, list[float]], index: NeighborIndex
    ) -> GroupResult:
        neighborhoods: dict[str, set[str]] = {}
        for word in vocab:
            vec = vectors[word]
            neigh = {
                nb_word
                for nb_word, sim in index.query(vec, self.neighbor_cfg.top_k)
                if sim >= self.cfg.threshold
            }
            neighborhoods[word] = neigh | {word}
        return GroupResult(clusters=[], neighborhoods=neighborhoods, word_to_cluster={})


__all__ = ["Cluster", "GroupResult", "SimilarityGrouper"]
