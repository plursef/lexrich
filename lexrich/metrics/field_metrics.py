from __future__ import annotations

from .base import FieldContext, MetricResult, RichnessMetric


class FieldCoveragePer10kMetric(RichnessMetric):
    name = "FieldCoveragePer10k"

    def compute(self, ctx: FieldContext) -> MetricResult:
        token_total = sum(ctx.counts.get(w, 0) for w in ctx.field_words)
        value = (token_total / ctx.total_tokens * 10000) if ctx.total_tokens > 0 else 0.0
        return MetricResult(name=self.name, value=value, details={"tokens": token_total})


class FieldTypeCountMetric(RichnessMetric):
    name = "FieldTypeCount"

    def compute(self, ctx: FieldContext) -> MetricResult:
        types = [w for w in ctx.field_words if ctx.counts.get(w, 0) > 0]
        return MetricResult(name=self.name, value=len(types))


__all__ = ["FieldCoveragePer10kMetric", "FieldTypeCountMetric"]
