import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from cluster import cluster_embeddings


class TestClusterEmbeddings:
    def test_empty_input(self):
        labels, reduced = cluster_embeddings(np.empty((0, 1024), dtype=float))
        assert labels.size == 0
        assert reduced.size == 0

    def test_single_paper(self):
        labels, reduced = cluster_embeddings(np.ones((1, 1024), dtype=float))
        assert labels.size == 1
        assert labels[0] == 0

    def test_two_papers(self):
        labels, reduced = cluster_embeddings(np.ones((2, 1024), dtype=float))
        assert labels.size == 2
        assert np.all(labels == 0)

    def test_normal_clustering(self):
        rng = np.random.RandomState(42)
        clusters = [rng.randn(10, 1024) + offset for offset in [0, 5, 10]]
        data = np.vstack(clusters).astype(float)
        labels, reduced = cluster_embeddings(data)
        assert len(labels) == 30
        assert len(set(labels)) >= 2
