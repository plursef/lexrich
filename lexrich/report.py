from __future__ import annotations

import json
from typing import Iterable

from .analyzer import AnalysisResult


class ReportFormatter:
    def __init__(self, result: AnalysisResult):
        self.result = result

    def to_json(self, indent: int = 2) -> str:
        payload = {
            "total_tokens": self.result.total_tokens,
            "total_types": self.result.total_types,
            "per_field": {
                field: {m.name: m.value for m in metrics} for field, metrics in self.result.per_field.items()
            },
            "debug": self.result.debug,
        }
        return json.dumps(payload, indent=indent)

    def to_markdown_tables(self) -> str:
        lines = []
        for field, metrics in self.result.per_field.items():
            lines.append(f"### Field: {field}")
            lines.append("| Metric | Value |")
            lines.append("| --- | --- |")
            for m in metrics:
                lines.append(f"| {m.name} | {m.value:.4f}" if isinstance(m.value, float) else f"| {m.name} | {m.value}|")
            lines.append("")
        return "\n".join(lines)


__all__ = ["ReportFormatter"]
