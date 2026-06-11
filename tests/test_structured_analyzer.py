"""Tests for the structured analyzer — the core V3 abstraction."""
import pytest
from unittest.mock import MagicMock

from paper import Paper, StructuredUnderstanding
from structured_analyzer import (
    analyze_paper_structure,
    _build_analysis_prompt,
    _parse_analysis_response,
)


class TestBuildAnalysisPrompt:
    def test_basic_prompt(self, seed_paper_v3):
        prompt = _build_analysis_prompt(seed_paper_v3)
        assert "BEVFormer" in prompt
        assert "USER FOCUS" in prompt
        assert "temporal attention" in prompt

    def test_prompt_without_user_description(self, sample_paper_v3):
        prompt = _build_analysis_prompt(sample_paper_v3)
        assert "USER FOCUS" not in prompt

    def test_prompt_truncates_long_text(self, sample_paper_v3):
        sample_paper_v3.full_text = "x" * 200000
        prompt = _build_analysis_prompt(sample_paper_v3)
        assert len(prompt) < 155000  # Should be truncated

    def test_prompt_no_text(self):
        paper = Paper(id="empty", title="Empty")
        prompt = _build_analysis_prompt(paper)
        assert prompt == ""


class TestParseResponse:
    def test_valid_json(self, structured_analysis_response):
        result = _parse_analysis_response(structured_analysis_response)
        assert result is not None
        assert "problem" in result
        assert len(result["components"]) == 3
        assert len(result["formulas"]) == 2

    def test_json_in_markdown_fence(self):
        raw = "```json\n{\"problem\": \"test\", \"components\": []}\n```"
        result = _parse_analysis_response(raw)
        assert result is not None
        assert result["problem"] == "test"

    def test_malformed_json(self):
        result = _parse_analysis_response("not json at all")
        assert result is None

    def test_empty_string(self):
        assert _parse_analysis_response("") is None
        assert _parse_analysis_response(None) is None

    def test_missing_list_fields_become_empty(self):
        raw = '{"problem": "test"}'
        result = _parse_analysis_response(raw)
        assert result["components"] == []
        assert result["formulas"] == []
        assert result["contributions"] == []
        assert result["limitations"] == []

    def test_non_list_fields_are_normalized(self):
        raw = '{"problem": "test", "components": "not a list"}'
        result = _parse_analysis_response(raw)
        assert result["components"] == []


class TestAnalyzePaperStructure:
    def test_returns_basic_structure_without_client(self, sample_paper_v3):
        result = analyze_paper_structure(sample_paper_v3, client=None)
        # Graceful degradation: returns basic structure from abstract
        assert result is not None
        assert result.problem == sample_paper_v3.abstract

    def test_successful_analysis(self, sample_paper_v3, mock_llm_client,
                                  structured_analysis_response):
        mock_llm_client.chat.completions.create.return_value = \
            type("Response", (), {
                "choices": [
                    type("Choice", (), {
                        "message": type("Msg", (), {"content": structured_analysis_response})
                    })()
                ]
            })()
        result = analyze_paper_structure(sample_paper_v3, mock_llm_client)
        assert result is not None
        assert isinstance(result, StructuredUnderstanding)
        assert result.architecture_overview != ""
        assert len(result.components) == 3
        assert len(result.formulas) == 2
        assert len(result.main_results) == 2

    def test_empty_response_returns_none(self, sample_paper_v3, mock_llm_client):
        mock_llm_client.chat.completions.create.return_value = \
            type("Response", (), {
                "choices": [
                    type("Choice", (), {
                        "message": type("Msg", (), {"content": ""})
                    })()
                ]
            })()
        result = analyze_paper_structure(sample_paper_v3, mock_llm_client)
        assert result is None

    def test_api_error_returns_basic_structure(self, sample_paper_v3, mock_llm_client):
        mock_llm_client.chat.completions.create.side_effect = Exception("API error")
        result = analyze_paper_structure(sample_paper_v3, mock_llm_client)
        # Graceful degradation: returns basic structure from abstract
        assert result is not None
        assert result.problem == sample_paper_v3.abstract

    def test_malformed_response_returns_none(self, sample_paper_v3, mock_llm_client):
        mock_llm_client.chat.completions.create.return_value = \
            type("Response", (), {
                "choices": [
                    type("Choice", (), {
                        "message": type("Msg", (), {"content": "not valid json {{"})
                    })()
                ]
            })()
        result = analyze_paper_structure(sample_paper_v3, mock_llm_client)
        assert result is None

    def test_paper_without_full_text(self, mock_llm_client, structured_analysis_response):
        paper = Paper(
            id="abstract_only",
            title="Abstract Only Paper",
            abstract="Just an abstract.",
            year=2023,
        )
        mock_llm_client.chat.completions.create.return_value = \
            type("Response", (), {
                "choices": [
                    type("Choice", (), {
                        "message": type("Msg", (), {"content": structured_analysis_response})
                    })()
                ]
            })()
        result = analyze_paper_structure(paper, mock_llm_client)
        assert result is not None  # Uses abstract as fallback text
