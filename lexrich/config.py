from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import json

try:  # Optional dependency; fallback parser is provided below.
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - exercised in constrained environments
    yaml = None


def _simple_yaml_load(text: str) -> dict[str, Any]:
    """Parse a tiny subset of YAML used by the project.

    The seeds/field config only requires top-level mappings and inline lists,
    e.g. ``field: [a, b, c]``. This helper keeps the repository runnable even
    when PyYAML is unavailable.
    """

    result: dict[str, Any] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if value.startswith("[") and value.endswith("]"):
            items = [v.strip() for v in value[1:-1].split(",") if v.strip()]
            result[key] = {"seeds": items}
        else:
            result[key] = value
    return result


@dataclass
class PreprocessConfig:
    language: str = "en"
    lowercase: bool = True
    lemmatize: bool = True
    pos_tag: bool = True
    keep_pos: tuple[str, ...] = ("NOUN", "VERB", "ADJ", "ADV")
    remove_stopwords: bool = True
    min_token_len: int = 2


@dataclass
class EmbeddingConfig:
    model_type: str = "glove"  # "glove" | "word2vec" | "fasttext"
    model_path: str = ""
    vector_dim: int = 300
    oov_strategy: str = "skip"  # "skip" | "lowercase_fallback" | "subword" | "zero"
    normalize_vectors: bool = True


@dataclass
class NeighborConfig:
    index_type: str = "exact"  # "exact" | "ann"
    top_k: int = 30
    similarity: str = "cosine"


@dataclass
class GroupingConfig:
    mode: str = "threshold_graph"  # "threshold_graph" | "topk_neighborhood"
    threshold: float = 0.55
    max_vocab: int = 20000
    min_freq: int = 2
    graph_build: str = "topk_then_threshold"


@dataclass
class FieldConfig:
    seed_path: str = "lexrich/resources/seed_fields.yaml"
    expand_fields: bool = True
    expand_top_k: int = 50
    expand_threshold: float = 0.55
    assign_mode: str = "word"  # "word" | "cluster"


@dataclass
class MetricsConfig:
    enabled: list[str] = field(
        default_factory=lambda: [
            "FieldCoveragePer10k",
            "FieldTypeCount",
            "ClusterEntropy",
            "Dominance",
        ]
    )
    params: dict[str, dict[str, Any]] = field(default_factory=dict)


@dataclass
class AnalysisConfig:
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    neighbor: NeighborConfig = field(default_factory=NeighborConfig)
    grouping: GroupingConfig = field(default_factory=GroupingConfig)
    fields: FieldConfig = field(default_factory=FieldConfig)
    metrics: MetricsConfig = field(default_factory=MetricsConfig)
    debug: bool = False
    threshold_grid: list[float] | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> "AnalysisConfig":
        return cls(
            preprocess=PreprocessConfig(**data.get("preprocess", {})),
            embedding=EmbeddingConfig(**data.get("embedding", {})),
            neighbor=NeighborConfig(**data.get("neighbor", {})),
            grouping=GroupingConfig(**data.get("grouping", {})),
            fields=FieldConfig(**data.get("fields", {})),
            metrics=MetricsConfig(**data.get("metrics", {})),
            debug=data.get("debug", False),
            threshold_grid=data.get("threshold_grid"),
        )

    @classmethod
    def load(cls, path: str | Path) -> "AnalysisConfig":
        path = Path(path)
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is not None:
                data = yaml.safe_load(path.read_text())
            else:
                data = _simple_yaml_load(path.read_text())
        else:
            data = json.loads(path.read_text())
        return cls.from_mapping(data or {})

    def to_dict(self) -> dict[str, Any]:
        return {
            "preprocess": self.preprocess.__dict__,
            "embedding": self.embedding.__dict__,
            "neighbor": self.neighbor.__dict__,
            "grouping": self.grouping.__dict__,
            "fields": self.fields.__dict__,
            "metrics": {
                "enabled": list(self.metrics.enabled),
                "params": {k: dict(v) for k, v in self.metrics.params.items()},
            },
            "debug": self.debug,
            "threshold_grid": list(self.threshold_grid) if self.threshold_grid else None,
        }


__all__ = [
    "PreprocessConfig",
    "EmbeddingConfig",
    "NeighborConfig",
    "GroupingConfig",
    "FieldConfig",
    "MetricsConfig",
    "AnalysisConfig",
]
