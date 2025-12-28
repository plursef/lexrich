from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    yaml = None

from .config import FieldConfig, _simple_yaml_load
from .embedding import EmbeddingModel
from .grouping import Cluster
from .neighbors import NeighborIndex


@dataclass
class SemanticField:
    name: str
    seeds: list[str]


class FieldLexiconBuilder:
    def __init__(self, cfg: FieldConfig, embed: EmbeddingModel, index: NeighborIndex):
        self.cfg = cfg
        self.embed = embed
        self.index = index

    def _load_seeds(self) -> dict[str, list[str]]:
        path = Path(self.cfg.seed_path)
        if not path.exists():
            return {}
        if yaml is not None:
            data = yaml.safe_load(path.read_text())
        else:
            data = _simple_yaml_load(path.read_text())
        seeds: dict[str, list[str]] = {}
        for name, entry in (data or {}).items():
            if isinstance(entry, dict):
                seeds[name] = entry.get("seeds", [])
            elif isinstance(entry, list):
                seeds[name] = entry
            else:
                continue
        return seeds

    def build(self) -> dict[str, set[str]]:
        seeds = self._load_seeds()
        field_lexicons: dict[str, set[str]] = {}
        for name, seed_list in seeds.items():
            lex = set(seed_list)
            if self.cfg.expand_fields:
                lex |= self._expand(seed_list)
            field_lexicons[name] = lex
        return field_lexicons

    def _expand(self, seed_list: Iterable[str]) -> set[str]:
        expanded: set[str] = set()
        for seed in seed_list:
            vec = self.embed.get_vector(seed)
            if vec is None:
                continue
            neighbors = self.index.query(vec, self.cfg.expand_top_k)
            for nb_word, sim in neighbors:
                if sim >= self.cfg.expand_threshold:
                    expanded.add(nb_word)
        return expanded


class FieldAssigner:
    def __init__(self, cfg: FieldConfig, field_lexicons: dict[str, set[str]], embed: EmbeddingModel):
        self.cfg = cfg
        self.field_lexicons = field_lexicons
        self.embed = embed

    def assign_words(self, vocab: list[str]) -> dict[str, str | None]:
        word_to_field: dict[str, str | None] = {}
        for word in vocab:
            candidates = [name for name, lex in self.field_lexicons.items() if word in lex]
            if not candidates:
                word_to_field[word] = None
            elif len(candidates) == 1:
                word_to_field[word] = candidates[0]
            else:
                # tie-breaker by closest seed vector
                word_to_field[word] = self._closest_field(word, candidates)
        return word_to_field

    def _closest_field(self, word: str, candidates: list[str]) -> str | None:
        vec = self.embed.get_vector(word)
        if vec is None:
            return candidates[0]
        best_field = candidates[0]
        best_sim = -1.0
        for field in candidates:
            sims = []
            for seed in self.field_lexicons[field]:
                seed_vec = self.embed.get_vector(seed)
                if seed_vec is None:
                    continue
                sims.append(sum(a * b for a, b in zip(seed_vec, vec)))
            if sims:
                avg = sum(sims) / len(sims)
                if avg > best_sim:
                    best_sim = avg
                    best_field = field
        return best_field

    def assign_clusters(self, clusters: list[Cluster]) -> dict[int, str | None]:
        cluster_to_field: dict[int, str | None] = {}
        word_assignments = self.assign_words([w for c in clusters for w in c.members])
        for cluster in clusters:
            counts: dict[str, int] = {}
            for word in cluster.members:
                field = word_assignments.get(word)
                if field:
                    counts[field] = counts.get(field, 0) + 1
            if not counts:
                cluster_to_field[cluster.id] = None
            else:
                cluster_to_field[cluster.id] = max(counts.items(), key=lambda x: x[1])[0]
        return cluster_to_field


__all__ = ["SemanticField", "FieldLexiconBuilder", "FieldAssigner"]
