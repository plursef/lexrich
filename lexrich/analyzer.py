from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

from .config import AnalysisConfig
from .preprocess import TextPreprocessor
from .extractor import VocabularyExtractor, CorpusStats
from .embedding import KeyedVectorsEmbedding, EmbeddingModel
from .neighbors import ExactNeighborIndex
from .grouping import SimilarityGrouper, GroupResult
from .fields import FieldLexiconBuilder, FieldAssigner
from .metrics import METRIC_REGISTRY, RichnessMetric, FieldContext, MetricResult


@dataclass
class AnalysisResult:
    total_tokens: int
    total_types: int
    per_field: dict[str, list[MetricResult]]
    overall: list[MetricResult] | None = None
    debug: dict | None = None


class RichnessAnalyzer:
    def __init__(self, cfg: AnalysisConfig):
        self.cfg = cfg
        self.preprocessor = TextPreprocessor(cfg.preprocess)
        self.extractor = VocabularyExtractor(cfg.grouping)
        self.embedding: EmbeddingModel = KeyedVectorsEmbedding(cfg.embedding)
        self.index = ExactNeighborIndex(cfg.neighbor)
        self.grouper = SimilarityGrouper(cfg.grouping, cfg.neighbor)

    def _select_vectors(self, vocab: list[str]) -> dict[str, any]:
        vectors = {}
        for word in vocab:
            vec = self.embedding.get_vector(word)
            if vec is not None:
                vectors[word] = vec
        return vectors

    def _run_metrics(
        self, field_words: dict[str, set[str]], counts: dict[str, int], group_result: GroupResult, total_tokens: int
    ) -> dict[str, list[MetricResult]]:
        results: dict[str, list[MetricResult]] = {}
        for field, words in field_words.items():
            ctx = FieldContext(
                field_name=field,
                field_words=words,
                counts=counts,
                total_tokens=total_tokens,
                group_result=group_result,
            )
            field_results: list[MetricResult] = []
            for metric_name in self.cfg.metrics.enabled:
                metric_cls = METRIC_REGISTRY.get(metric_name)
                if not metric_cls:
                    continue
                metric: RichnessMetric = metric_cls(self.cfg.metrics.params.get(metric_name, {}))
                field_results.append(metric.compute(ctx))
            results[field] = field_results
        return results

    def _prepare_field_words(
        self, assigner: FieldAssigner, vocab: list[str], group_result: GroupResult
    ) -> dict[str, set[str]]:
        field_words: dict[str, set[str]] = {name: set() for name in assigner.field_lexicons.keys()}
        if self.cfg.fields.assign_mode == "word":
            word_to_field = assigner.assign_words(vocab)
            for word, field in word_to_field.items():
                if field:
                    field_words[field].add(word)
        else:
            cluster_to_field = assigner.assign_clusters(group_result.clusters)
            for cluster in group_result.clusters:
                field = cluster_to_field.get(cluster.id)
                if not field:
                    continue
                field_words[field].update(cluster.members)
        return field_words

    def _analyze_once(self, text: str, threshold: float | None = None) -> tuple[AnalysisResult, GroupResult]:
        if threshold is not None:
            self.grouper.cfg.threshold = threshold
        tokens = self.preprocessor.process(text)
        stats: CorpusStats = self.extractor.extract(tokens)
        vocab = list(stats.counts.keys())
        vectors = self._select_vectors(vocab)
        self.index.build(vectors)
        group_result = self.grouper.group(vocab, vectors, self.index)
        field_builder = FieldLexiconBuilder(self.cfg.fields, self.embedding, self.index)
        lexicons = field_builder.build()
        assigner = FieldAssigner(self.cfg.fields, lexicons, self.embedding)
        field_words = self._prepare_field_words(assigner, vocab, group_result)
        per_field = self._run_metrics(field_words, stats.counts, group_result, stats.total_tokens)
        debug = None
        if self.cfg.debug:
            debug = {
                "lexicons": {k: sorted(v) for k, v in lexicons.items()},
                "clusters": [sorted(c.members) for c in group_result.clusters],
            }
        result = AnalysisResult(
            total_tokens=stats.total_tokens,
            total_types=stats.total_types,
            per_field=per_field,
            overall=None,
            debug=debug,
        )
        return result, group_result

    def analyze(self, text: str) -> AnalysisResult:
        if self.cfg.threshold_grid:
            sweep = {}
            base_debug = {}
            last_result = None
            for tau in self.cfg.threshold_grid:
                res, group = self._analyze_once(text, threshold=tau)
                sweep[tau] = {field: {m.name: m.value for m in metrics} for field, metrics in res.per_field.items()}
                base_debug[tau] = {
                    "clusters": [sorted(c.members) for c in group.clusters],
                }
                last_result = res
            if last_result and last_result.debug is not None:
                last_result.debug["threshold_sweep"] = sweep
                last_result.debug["threshold_groups"] = base_debug
            return last_result if last_result else AnalysisResult(0, 0, {})
        result, _ = self._analyze_once(text)
        return result


__all__ = ["RichnessAnalyzer", "AnalysisResult"]
