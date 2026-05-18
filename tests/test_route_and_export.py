"""Tests for route analyzer and markdown exporter."""
import pytest
from unittest.mock import MagicMock

from paper import Paper
from route_analyzer import (
    analyze_routes,
    compare_with_mainstream,
    _fallback_routes,
    _fallback_comparison,
)
from markdown_exporter_v3 import export_markdown, _t, _ct, _escape_md
from paper import CitationType, Reference


class TestRouteAnalyzer:
    def test_fallback_routes_with_papers(self, sample_paper_v3):
        papers = [sample_paper_v3]
        result = _fallback_routes(papers)
        assert "branches" in result
        assert "overview" in result
        assert len(result["branches"]) == 1

    def test_fallback_routes_empty(self):
        result = _fallback_routes([])
        assert result["branches"] == []

    def test_analyze_routes_without_client(self, sample_paper_v3):
        result = analyze_routes([sample_paper_v3], sample_paper_v3, client=None)
        assert result is not None
        assert "branches" in result
        assert "overview" in result

    def test_analyze_routes_too_few_papers(self, sample_paper_v3):
        result = analyze_routes([sample_paper_v3], sample_paper_v3, None)
        assert result is not None

    def test_successful_analysis(self, sample_paper_v3, mock_llm_client,
                                   route_analysis_response):
        sample_paper_v3.structured = MagicMock()
        sample_paper_v3.structured.architecture_overview = "Test architecture"
        sample_paper_v3.structured.key_insight = "Test insight"

        paper2 = Paper(id="p2", title="Paper 2", year=2020)
        paper2.structured = MagicMock()
        paper2.structured.architecture_overview = "Another architecture"

        mock_llm_client.chat.completions.create.return_value = \
            type("Response", (), {
                "choices": [
                    type("Choice", (), {
                        "message": type("Msg", (), {"content": route_analysis_response})
                    })()
                ]
            })()

        result = analyze_routes([sample_paper_v3, paper2], sample_paper_v3, mock_llm_client)
        assert result is not None
        assert len(result["branches"]) == 2


class TestCompareWithMainstream:
    def test_fallback_comparison(self, sample_paper_v3):
        routes = {"branches": []}
        result = _fallback_comparison(sample_paper_v3, routes)
        assert "comparison_matrix" in result
        assert "narrative" in result

    def test_compare_without_client(self, sample_paper_v3):
        result = compare_with_mainstream(sample_paper_v3, {"branches": []}, client=None)
        assert result is not None

    def test_successful_comparison(self, sample_paper_v3, mock_llm_client,
                                     comparison_response):
        routes = {
            "branches": [
                {"name": "Transformer-based", "description": "Uses cross-attention",
                 "is_mainstream": True, "paper_ids": ["p1"]},
            ]
        }
        sample_paper_v3.structured = MagicMock()
        sample_paper_v3.structured.architecture_overview = "Test"

        mock_llm_client.chat.completions.create.return_value = \
            type("Response", (), {
                "choices": [
                    type("Choice", (), {
                        "message": type("Msg", (), {"content": comparison_response})
                    })()
                ]
            })()

        result = compare_with_mainstream(sample_paper_v3, routes, mock_llm_client)
        assert result is not None
        assert len(result["comparison_matrix"]) == 2


class TestMarkdownExporter:
    def test_export_basic(self, sample_paper_v3, tmp_path):
        out = tmp_path / "test_report.md"
        export_markdown(sample_paper_v3, None, None, None, out, lang="en")
        content = out.read_text()
        assert sample_paper_v3.title in content
        assert "Paper Overview" in content

    def test_export_with_structured(self, sample_paper_v3, sample_structured_understanding,
                                      tmp_path):
        sample_paper_v3.structured = sample_structured_understanding
        out = tmp_path / "test_report.md"
        export_markdown(sample_paper_v3, None, None, None, out, lang="en")
        content = out.read_text()
        assert "Method Architecture" in content
        assert "Encoder" in content
        assert "Decoder" in content

    def test_export_with_routes(self, sample_paper_v3, tmp_path):
        routes = {
            "overview": "Test field overview.",
            "branches": [
                {
                    "name": "Transformer-based",
                    "description": "Uses attention",
                    "paper_ids": ["p1"],
                    "is_mainstream": True,
                    "common_technical_tags": ["tag1"],
                }
            ],
        }
        out = tmp_path / "test_report.md"
        export_markdown(sample_paper_v3, routes, None, None, out, lang="en")
        content = out.read_text()
        assert "Field Technical Landscape" in content
        assert "Transformer-based" in content

    def test_export_with_comparison(self, sample_paper_v3, comparison_response, tmp_path):
        import json
        comparison = json.loads(comparison_response)
        out = tmp_path / "test_report.md"
        export_markdown(sample_paper_v3, None, comparison, None, out, lang="en")
        content = out.read_text()
        assert "Comparative Analysis" in content
        assert "Depth Estimation" in content

    def test_export_with_references(self, sample_paper_v3, tmp_path):
        refs = [
            Reference(paper_id="ref1", paper_title="Paper A",
                       citation_type=CitationType.SUPPORTING, is_key_reference=True,
                       context="cited in method"),
            Reference(paper_id="ref2", paper_title="Paper B",
                       citation_type=CitationType.FOUNDATIONAL, is_key_reference=True),
        ]
        out = tmp_path / "test_report.md"
        export_markdown(sample_paper_v3, None, None, refs, out, lang="en")
        content = out.read_text()
        assert "Reference Classification" in content
        assert "Paper A" in content
        assert "Supporting" in content

    def test_export_chinese(self, sample_paper_v3, sample_structured_understanding, tmp_path):
        sample_paper_v3.structured = sample_structured_understanding
        out = tmp_path / "test_report_zh.md"
        export_markdown(sample_paper_v3, None, None, None, out, lang="zh")
        content = out.read_text()
        assert "结构化解读" in content
        assert "论文概览" in content
        assert "方法架构" in content

    def test_t_helper(self):
        assert _t("overview", "en") == "Paper Overview"
        assert _t("overview", "zh") == "论文概览"
        assert _t("nonexistent", "en") == "nonexistent"

    def test_ct_helper(self):
        assert _ct(CitationType.SUPPORTING, "en") == "Supporting"
        assert _ct(CitationType.SUPPORTING, "zh") == "赞同"

    def test_escape_md(self):
        assert _escape_md("hello|world") == "hello\\|world"
        assert _escape_md("line1\nline2") == "line1 line2"
