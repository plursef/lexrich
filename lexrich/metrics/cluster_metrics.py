from __future__ import annotations

import math
from collections import Counter

from .base import FieldContext, MetricResult, RichnessMetric


def _cluster_counts(ctx: FieldContext) -> Counter:
    counter: Counter[int] = Counter()
    if ctx.group_result.clusters:
        for word in ctx.field_words:
            if word in ctx.group_result.word_to_cluster:
                cid = ctx.group_result.word_to_cluster[word]
                counter[cid] += ctx.counts.get(word, 0)
    elif ctx.group_result.neighborhoods:
        for word in ctx.field_words:
            neigh = ctx.group_result.neighborhoods.get(word, {word})
            weight = ctx.counts.get(word, 0)
            counter.update({hash(frozenset(neigh)): weight})
    return counter


class ClusterEntropyMetric(RichnessMetric):
    name = "ClusterEntropy"

    def compute(self, ctx: FieldContext) -> MetricResult:
        counts = _cluster_counts(ctx)
        total = sum(counts.values())
        if total == 0:
            return MetricResult(name=self.name, value=0.0)
        entropy = 0.0
        for c in counts.values():
            p = c / total
            entropy -= p * math.log(p + 1e-12, 2)
        return MetricResult(name=self.name, value=entropy, details={"clusters": len(counts)})


class DominanceMetric(RichnessMetric):
    name = "Dominance"

    def compute(self, ctx: FieldContext) -> MetricResult:
        counts = _cluster_counts(ctx)
        total = sum(counts.values())
        if total == 0:
            return MetricResult(name=self.name, value=0.0)
        top_n = int(self.params.get("top_n", 1))
        largest = sum(v for _, v in counts.most_common(top_n))
        return MetricResult(name=self.name, value=largest / total if total else 0.0)


__all__ = ["ClusterEntropyMetric", "DominanceMetric"]
