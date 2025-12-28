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
    """Very small YAML subset parser used when PyYAML is unavailable.

    Supports:
    - Top-level mappings
    - Nested mappings via indentation (spaces)
    - Inline lists: ``[a, b, c]``

    This keeps the project runnable in constrained environments while being
    robust enough for ``seed_fields.yaml`` and minimal config files.
    """

    def _coerce_scalar(val: str) -> Any:
        # Strip matching quotes first
        if (val.startswith('"') and val.endswith('"')) or (val.startswith("'") and val.endswith("'")):
            val = val[1:-1]

        lower = val.lower()
        if lower in {"true", "false"}:
            return lower == "true"
        if lower in {"null", "none"}:
            return None

        # Try numeric conversions
        try:
            return int(val)
        except ValueError:
            pass
        try:
            return float(val)
        except ValueError:
            pass
        return val

    def _parse_inline(value: str) -> Any:
        if value.startswith("[") and value.endswith("]"):
            items = [v.strip() for v in value[1:-1].split(",") if v.strip()]
            return [_coerce_scalar(v) for v in items]
        return _coerce_scalar(value)

    lines = text.splitlines()

    def _next_nonempty(start: int) -> tuple[int, str] | tuple[None, None]:
        """Peek ahead to decide whether the upcoming block is a list."""
        for idx in range(start, len(lines)):
            candidate = lines[idx]
            if not candidate.strip() or candidate.lstrip().startswith("#"):
                continue
            indent = len(candidate) - len(candidate.lstrip(" "))
            return indent, candidate.strip()
        return None, None

    root: dict[str, Any] = {}
    stack: list[tuple[int, Any]] = [(0, root)]

    i = 0
    while i < len(lines):
        raw_line = lines[i]
        if not raw_line.strip() or raw_line.lstrip().startswith("#"):
            i += 1
            continue

        indent = len(raw_line) - len(raw_line.lstrip(" "))
        stripped = raw_line.strip()

        # List item (e.g., "- value")
        if stripped.startswith("-"):
            while stack and indent < stack[-1][0]:
                stack.pop()
            if not stack:
                stack = [(0, root)]
            parent = stack[-1][1]
            if not isinstance(parent, list):
                i += 1
                continue
            parent.append(_parse_inline(stripped[1:].strip()))
            i += 1
            continue

        if ":" not in stripped:
            i += 1
            continue

        key, value = stripped.split(":", 1)
        key = key.strip()
        value = value.strip()

        # Find parent container according to indent
        while stack and indent < stack[-1][0]:
            stack.pop()
        if not stack:
            stack = [(0, root)]
        parent = stack[-1][1]

        if value:
            parent[key] = _parse_inline(value)
        else:
            next_indent, next_line = _next_nonempty(i + 1)
            container: Any
            if next_indent is not None and next_indent > indent and next_line and next_line.startswith("-"):
                container = []
            else:
                container = {}
            parent[key] = container
            stack.append((indent + 1, container))

        i += 1

    return root


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
