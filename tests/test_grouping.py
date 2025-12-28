from lexrich.config import GroupingConfig, NeighborConfig
from lexrich.grouping import SimilarityGrouper
from lexrich.neighbors import ExactNeighborIndex


def test_threshold_graph_groups_words():
    vocab = ["good", "bad", "cat"]
    vectors = {
        "good": [1.0, 0.0],
        "bad": [0.9, 0.1],
        "cat": [0.0, 1.0],
    }
    cfg = GroupingConfig(threshold=0.8, mode="threshold_graph")
    neighbor_cfg = NeighborConfig(top_k=3)
    index = ExactNeighborIndex(neighbor_cfg)
    index.build(vectors)
    grouper = SimilarityGrouper(cfg, neighbor_cfg)
    result = grouper.group(vocab, vectors, index)
    # good and bad should be in same cluster, cat isolated
    assert len(result.clusters) == 2
    cluster_sizes = sorted(len(c.members) for c in result.clusters)
    assert cluster_sizes == [1, 2]
