"""Integration tests for the full pipeline with mocked external calls.

Tests verify:
- Pipeline completes without an LLM client (graceful degradation)
- Pipeline completes with a mocked LLM client (new analysis steps integrated)
- Relevance filter correctly removes noise in a full pipeline run
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from conftest import _mock_create


# ---------------------------------------------------------------------------
# Fixtures: mock the external modules
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_externals(monkeypatch, mixed_relevance_papers):
    """Replace crawler, embedding, clustering with deterministic stubs.

    This prevents any real API calls during integration tests.
    """
    import main as pipeline

    monkeypatch.setattr("main.fetch_papers", lambda _: mixed_relevance_papers)
    monkeypatch.setattr("main.generate_embeddings", lambda p, c: np.random.randn(len(p), 32))
    monkeypatch.setattr("main.cluster_embeddings", lambda e: (
        np.zeros(len(e), dtype=int), e
    ))
    monkeypatch.setattr("main._attach_embeddings", lambda p, e: (
        [p.update({"_embedding": e[i]}) for i, p in enumerate(p)]
    ))


@pytest.fixture
def mock_llm_calls(monkeypatch):
    """Replace all LLM module functions with deterministic mocks."""
    # llm_namer
    monkeypatch.setattr("main.refine_query", lambda k, c: k)
    monkeypatch.setattr("main.name_branch_with_llm", lambda p, c: None)

    # llm_analyzer — filter keeps only BEV papers
    filter_response = """[
  {"index": 0, "judgment": "relevant", "reason": "BEV"},
  {"index": 1, "judgment": "relevant", "reason": "BEV"},
  {"index": 2, "judgment": "irrelevant", "reason": "Sociology"},
  {"index": 3, "judgment": "relevant", "reason": "BEV"},
  {"index": 4, "judgment": "irrelevant", "reason": "Neuropsychology"},
  {"index": 5, "judgment": "relevant", "reason": "BEV"},
  {"index": 6, "judgment": "irrelevant", "reason": "Biofilms"}
]"""
    branch_response = """{
  "branch_name": "Camera-based BEV",
  "narrative": "The branch evolved from early camera-only methods.",
  "key_papers": [
    {"title": "Lift Splat Shoot", "year": 2020, "link": "https://arxiv.org/abs/2006.12345",
     "significance": "Pioneered BEV representation", "importance_rank": 1}
  ],
  "paradigm_shifts": [],
  "technical_forks": []
}"""
    evolution_response = """{
  "overview": "The field evolved from camera to multi-modal approaches.",
  "cross_branch_relationships": [],
  "temporal_ordering": ["Camera-based BEV"]
}"""
    validation_response = """{
  "quality_score": 7, "issues": [], "missing_topics": [], "suggested_improvements": []
}"""

    analyzer_client = MagicMock()

    call_count = [0]
    canned = [filter_response, branch_response, evolution_response, validation_response]

    def sequential_create(*args, **kwargs):
        idx = call_count[0]
        call_count[0] += 1
        return MagicMock(
            choices=[MagicMock(message=MagicMock(content=canned[min(idx, len(canned)-1)]))]
        )

    analyzer_client.chat.completions.create = sequential_create

    monkeypatch.setattr("main.build_analyzer_client", lambda: analyzer_client)
    monkeypatch.setattr("main.build_llm_client", lambda: MagicMock())

    return analyzer_client


# ======================================================================
# Integration tests
# ======================================================================

class TestFullPipeline:
    def test_pipeline_without_llm(self, mock_externals, monkeypatch):
        """Pipeline should complete gracefully when no LLM client is available."""
        monkeypatch.setattr("main.build_analyzer_client", lambda: None)
        monkeypatch.setattr("main.build_llm_client", lambda: None)
        import main as pipeline
        monkeypatch.setattr("sys.argv", ["main", "BEV perception"])
        pipeline.main()

    def test_pipeline_with_mocked_llm(self, mock_externals, mock_llm_calls, monkeypatch):
        """Pipeline should complete with full LLM analysis steps."""
        import main as pipeline
        monkeypatch.setattr("sys.argv", ["main", "BEV perception"])
        pipeline.main()

    def test_pipeline_relevance_filter_removes_noise(
        self, mock_externals, mock_llm_calls, monkeypatch
    ):
        """After relevance filter, no obviously irrelevant papers should remain."""
        import main as pipeline

        filtered_papers = []
        original_filter = pipeline.filter_relevant_papers

        def tracking_filter(papers, query, client, **kwargs):
            result = original_filter(papers, query, client, **kwargs)
            filtered_papers.extend(result)
            return result

        monkeypatch.setattr("main.filter_relevant_papers", tracking_filter)
        monkeypatch.setattr("sys.argv", ["main", "BEV perception"])
        pipeline.main()

        titles = {p["title"] for p in filtered_papers}
        assert "BEVFormer: Bird's Eye View Object Detection with Transformers" in titles
        assert "Faces of Inequality" not in titles
        assert "The Organization of Behavior" not in titles

    def test_pipeline_no_branches(self, mock_externals, monkeypatch):
        """Pipeline should handle empty/rejected branches gracefully."""
        monkeypatch.setattr("main.build_analyzer_client", lambda: None)
        monkeypatch.setattr("main.build_llm_client", lambda: None)
        monkeypatch.setattr("main.cluster_embeddings", lambda e: (
            np.array([-1, -1, -1, -1, -1, -1, -1], dtype=int), e
        ))
        import main as pipeline
        monkeypatch.setattr("sys.argv", ["main", "BEV perception"])
        pipeline.main()
