import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from key_paper import rank_key_papers


def _make_papers(n, seed=0):
    rng = np.random.RandomState(seed)
    papers = []
    for i in range(n):
        papers.append(
            {
                "title": f"Paper {i}",
                "abstract": f"Abstract for paper {i}",
                "year": 2020 + i,
                "citation_count": rng.randint(0, 1000),
                "link": f"https://arxiv.org/abs/{i}",
                "_embedding": rng.randn(1024).astype(float),
            }
        )
    return papers


class TestRankKeyPapers:
    def test_empty_input(self):
        assert rank_key_papers([]) == []

    def test_single_paper(self):
        papers = _make_papers(1)
        result = rank_key_papers(papers)
        assert len(result) == 1
        assert result[0]["title"] == "Paper 0"
        assert "score" in result[0]

    def test_returns_top_k(self):
        papers = _make_papers(10)
        result = rank_key_papers(papers, top_k=3)
        assert len(result) == 3

    def test_no_internal_keys_leak(self):
        papers = _make_papers(5)
        result = rank_key_papers(papers, top_k=5)
        assert all(not k.startswith("_") for p in result for k in p)

    def test_score_ordering(self):
        papers = _make_papers(8)
        result = rank_key_papers(papers, top_k=8)
        scores = [p["score"] for p in result]
        assert scores == sorted(scores, reverse=True)
