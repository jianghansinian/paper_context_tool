"""Tests for text extractor module."""
import pytest
from text_extractor import (
    extract_text_from_pdf,
    find_references_section,
    estimate_token_count,
    truncate_text,
)


class TestFindReferencesSection:
    def test_finds_references(self):
        text = "Introduction\nSome text.\n\nReferences\n\n[1] Paper A\n[2] Paper B"
        refs = find_references_section(text)
        assert refs is not None
        assert "[1] Paper A" in refs

    def test_finds_REFERENCES_uppercase(self):
        text = "Conclusion\nFinal thoughts.\n\nREFERENCES\n\n[1] Foo\n[2] Bar"
        refs = find_references_section(text)
        assert refs is not None
        assert "[1] Foo" in refs

    def test_finds_bibliography(self):
        text = "Discussion\n\nBibliography\n\n1. Smith (2020)\n2. Jones (2021)"
        refs = find_references_section(text)
        assert refs is not None
        assert "Smith" in refs

    def test_takes_last_occurrence(self):
        text = "References\nfirst section\nAppendix\nReferences\nfinal section"
        refs = find_references_section(text)
        assert refs is not None
        assert "final section" in refs

    def test_no_references(self):
        text = "Introduction\nBody\nConclusion"
        refs = find_references_section(text)
        assert refs is None

    def test_empty_text(self):
        assert find_references_section("") is None


class TestEstimateTokenCount:
    def test_english_text(self):
        # 400 chars ≈ 100 tokens
        text = "a" * 400
        assert estimate_token_count(text) == 100

    def test_empty(self):
        assert estimate_token_count("") == 0

    def test_unicode(self):
        text = "你好世界" * 25  # 100 chars
        assert estimate_token_count(text) == 25


class TestTruncateText:
    def test_no_truncation_needed(self):
        text = "short text"
        assert truncate_text(text, max_chars=100) == text

    def test_truncates_at_paragraph_break(self):
        text = "A" * 500 + "\n\n" + "B" * 500
        result = truncate_text(text, max_chars=600)
        assert len(result) < 600
        assert result.endswith("A" * 500)

    def test_truncates_long_text(self):
        text = "x" * 10000
        result = truncate_text(text, max_chars=1000)
        assert len(result) <= 1000
