from __future__ import annotations

from dataclasses import dataclass
from abc import ABC, abstractmethod
from typing import Any

from ..grouping import GroupResult


@dataclass
class FieldContext:
    field_name: str
    field_words: set[str]
    counts: dict[str, int]
    total_tokens: int
    group_result: GroupResult


@dataclass
class MetricResult:
    name: str
    value: float | dict
    details: dict | None = None


class RichnessMetric(ABC):
    name: str

    def __init__(self, params: dict[str, Any] | None = None):
        self.params = params or {}

    @abstractmethod
    def compute(self, ctx: FieldContext) -> MetricResult:
        ...


__all__ = ["FieldContext", "MetricResult", "RichnessMetric"]
