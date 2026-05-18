"""Tests for the LLM analyzer module.

Focus on:
- JSON extraction helpers (markdown fence stripping, error cases)
- Graceful degradation when no LLM client is available
- Mocked LLM responses for the main functions
"""

from unittest.mock import MagicMock

import pytest

from llm_analyzer import (
    _extract_json_array,
    _extract_json_object,
    filter_relevant_papers,
    analyze_branch,
    analyze_evolution,
    validate_output,
)
from conftest import (
    MockChatCompletion,
    _mock_create,
)


# ======================================================================
# _extract_json_array
# ======================================================================

class TestExtractJsonArray:
    def test_bare_array(self):
        assert _extract_json_array('[{"a": 1}]') == [{"a": 1}]

    def test_with_markdown_fences(self):
        raw = """```json
[{"index": 0, "judgment": "relevant"}]
```"""
        assert _extract_json_array(raw) == [{"index": 0, "judgment": "relevant"}]

    def test_with_markdown_no_lang(self):
        raw = """```
[{"x": 1}]
```"""
        assert _extract_json_array(raw) == [{"x": 1}]

    def test_empty_string(self):
        assert _extract_json_array("") is None

    def test_no_array_in_text(self):
        assert _extract_json_array("Just some text without brackets") is None

    def test_malformed_json(self):
        assert _extract_json_array("[{broken") is None

    def test_with_leading_text(self):
        raw = "Here is the result:\n\n[{\"key\": \"value\"}]"
        assert _extract_json_array(raw) == [{"key": "value"}]


# ======================================================================
# _extract_json_object
# ======================================================================

class TestExtractJsonObject:
    def test_bare_object(self):
        assert _extract_json_object('{"a": 1}') == {"a": 1}

    def test_with_markdown_fences(self):
        raw = '```json\n{"name": "test"}\n```'
        assert _extract_json_object(raw) == {"name": "test"}

    def test_empty_string(self):
        assert _extract_json_object("") is None

    def test_no_object(self):
        assert _extract_json_object("[1, 2, 3]") is None

    def test_malformed(self):
        assert _extract_json_object('{"bad') is None

    def test_with_preamble(self):
        raw = 'Here:\n\n{"narrative": "text"}\n'
        assert _extract_json_object(raw) == {"narrative": "text"}


# ======================================================================
# filter_relevant_papers
# ======================================================================

class TestFilterRelevantPapers:
    def test_no_client_returns_all(self, mixed_relevance_papers):
        result = filter_relevant_papers(mixed_relevance_papers, "BEV perception", None)
        assert len(result) == len(mixed_relevance_papers)

    def test_empty_input(self):
        assert filter_relevant_papers([], "test", MagicMock()) == []

    def test_all_relevant_kept(self, mixed_relevance_papers, relevance_filter_response):
        client = MagicMock()
        client.chat.completions.create = _mock_create(relevance_filter_response)
        result = filter_relevant_papers(
            mixed_relevance_papers, "BEV perception in autonomous driving", client
        )
        # 4 relevant papers should be kept
        assert len(result) == 4
        titles = {p["title"] for p in result}
        assert "BEVFormer: Bird's Eye View Object Detection with Transformers" in titles
        assert "Faces of Inequality" not in titles

    def test_min_score_relevant(self, mixed_relevance_papers, relevance_filter_response):
        client = MagicMock()
        client.chat.completions.create = _mock_create(relevance_filter_response)
        result = filter_relevant_papers(
            mixed_relevance_papers, "BEV perception", client, min_score="relevant"
        )
        # All 4 relevant papers
        assert len(result) == 4

    def test_parse_failure_fallback(self, mixed_relevance_papers):
        client = MagicMock()
        client.chat.completions.create = _mock_create("not valid json")
        result = filter_relevant_papers(
            mixed_relevance_papers, "BEV perception", client
        )
        assert len(result) == len(mixed_relevance_papers)

    def test_wrong_length_fallback(self, mixed_relevance_papers):
        client = MagicMock()
        # Return only 3 judgments for 7 papers
        client.chat.completions.create = _mock_create(
            '[{"index": 0, "judgment": "relevant"}, {"index": 1, "judgment": "relevant"}, {"index": 2, "judgment": "irrelevant"}]'
        )
        result = filter_relevant_papers(
            mixed_relevance_papers, "BEV perception", client
        )
        # Fallback keeps all
        assert len(result) == len(mixed_relevance_papers)

    def test_api_error_fallback(self, mixed_relevance_papers):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API timeout")
        result = filter_relevant_papers(
            mixed_relevance_papers, "BEV perception", client
        )
        assert len(result) == len(mixed_relevance_papers)

    def test_attaches_relevance_field(self, mixed_relevance_papers, relevance_filter_response):
        client = MagicMock()
        client.chat.completions.create = _mock_create(relevance_filter_response)
        result = filter_relevant_papers(
            mixed_relevance_papers, "BEV perception", client
        )
        for paper in result:
            assert "_relevance" in paper
            assert paper["_relevance"] in ("relevant", "borderline", "irrelevant")


# ======================================================================
# analyze_branch
# ======================================================================

class TestAnalyzeBranch:
    def test_no_client_returns_none(self, sample_papers):
        assert analyze_branch(sample_papers, {"branch_name": "Test"}, None) is None

    def test_empty_papers(self):
        assert analyze_branch([], {"branch_name": "Test"}, MagicMock()) is None

    def test_valid_analysis(self, sample_papers, branch_analysis_response):
        client = MagicMock()
        client.chat.completions.create = _mock_create(branch_analysis_response)
        result = analyze_branch(
            sample_papers,
            {"branch_name": "Detection Methods", "branch_id": 0, "keywords": ["detection"]},
            client,
        )
        assert result is not None
        assert result["branch_name"] == "Camera-based BEV Perception"
        assert len(result["key_papers"]) == 1
        assert "narrative" in result
        assert "paradigm_shifts" in result
        assert "technical_forks" in result

    def test_parse_failure_returns_none(self, sample_papers):
        client = MagicMock()
        client.chat.completions.create = _mock_create("not valid json")
        result = analyze_branch(
            sample_papers, {"branch_name": "Test", "branch_id": 0, "keywords": []}, client
        )
        assert result is None

    def test_api_error_returns_none(self, sample_papers):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("API error")
        result = analyze_branch(
            sample_papers, {"branch_name": "Test", "branch_id": 0, "keywords": []}, client
        )
        assert result is None


# ======================================================================
# analyze_evolution
# ======================================================================

class TestAnalyzeEvolution:
    def test_no_client_returns_none(self):
        assert analyze_evolution([{"branch_name": "B1"}], "Field") is None

    def test_empty_branches(self):
        assert analyze_evolution([], "Field", MagicMock()) is None

    def test_valid_analysis(self, evolution_analysis_response):
        client = MagicMock()
        client.chat.completions.create = _mock_create(evolution_analysis_response)
        branches = [
            {"branch_name": "Camera-based", "key_papers": [{"title": "A", "year": 2020}], "narrative": "First approach"},
            {"branch_name": "Multi-modal", "key_papers": [{"title": "B", "year": 2023}], "narrative": "Fusion approach"},
        ]
        result = analyze_evolution(branches, "BEV Perception", client)
        assert result is not None
        assert "overview" in result
        assert len(result["cross_branch_relationships"]) == 1
        assert len(result["temporal_ordering"]) == 2

    def test_parse_failure_returns_none(self):
        client = MagicMock()
        client.chat.completions.create = _mock_create("bad json")
        result = analyze_evolution([{"branch_name": "B1", "key_papers": [], "narrative": ""}], "Field", client)
        assert result is None


# ======================================================================
# validate_output
# ======================================================================

class TestValidateOutput:
    def test_no_client_returns_none(self):
        assert validate_output({"field": "Test"}) is None

    def test_valid_validation(self, validation_response):
        client = MagicMock()
        client.chat.completions.create = _mock_create(validation_response)
        field_map = {
            "field": "BEV Perception",
            "branches": [{"branch_name": "Camera-based", "paper_count": 10, "key_papers": []}],
        }
        result = validate_output(field_map, client)
        assert result is not None
        assert result["quality_score"] == 8
        assert len(result["issues"]) == 1

    def test_parse_failure_returns_none(self):
        client = MagicMock()
        client.chat.completions.create = _mock_create("bad")
        result = validate_output({"field": "T", "branches": []}, client)
        assert result is None

    def test_api_error_returns_none(self):
        client = MagicMock()
        client.chat.completions.create.side_effect = RuntimeError("fail")
        result = validate_output({"field": "T", "branches": []}, client)
        assert result is None
