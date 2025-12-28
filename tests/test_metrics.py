from lexrich.metrics.field_metrics import FieldCoveragePer10kMetric, FieldTypeCountMetric
from lexrich.metrics.cluster_metrics import ClusterEntropyMetric, DominanceMetric
from lexrich.metrics.base import FieldContext
from lexrich.grouping import GroupResult, Cluster


def make_ctx():
    group = GroupResult(
        clusters=[Cluster(id=0, members={"happy", "joy"}), Cluster(id=1, members={"sad"})],
        neighborhoods={},
        word_to_cluster={"happy": 0, "joy": 0, "sad": 1},
    )
    return FieldContext(
        field_name="emotion",
        field_words={"happy", "joy", "sad"},
        counts={"happy": 3, "joy": 2, "sad": 1},
        total_tokens=10,
        group_result=group,
    )


def test_field_metrics():
    ctx = make_ctx()
    cov = FieldCoveragePer10kMetric().compute(ctx)
    types = FieldTypeCountMetric().compute(ctx)
    assert cov.value > 0
    assert types.value == 3


def test_cluster_metrics():
    ctx = make_ctx()
    ent = ClusterEntropyMetric().compute(ctx)
    dom = DominanceMetric({"top_n": 1}).compute(ctx)
    assert ent.value > 0
    assert dom.value > 0
