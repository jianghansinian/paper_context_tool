"""Tests for paper resolver module."""
import pytest
from unittest.mock import MagicMock, patch

from paper_resolver import _extract_arxiv_id, _is_pdf_path, _fetch_arxiv_metadata


class TestExtractArxivId:
    def test_canonical_id(self):
        assert _extract_arxiv_id("2203.17270") == "2203.17270"

    def test_abs_url(self):
        assert _extract_arxiv_id("https://arxiv.org/abs/2203.17270") == "2203.17270"

    def test_pdf_url(self):
        assert _extract_arxiv_id("https://arxiv.org/pdf/2203.17270.pdf") == "2203.17270"

    def test_url_with_version(self):
        assert _extract_arxiv_id("https://arxiv.org/abs/2203.17270v2") == "2203.17270"

    def test_short_url(self):
        assert _extract_arxiv_id("arxiv.org/abs/2203.17270") == "2203.17270"

    def test_old_style_id(self):
        assert _extract_arxiv_id("0704.0001") == "0704.0001"

    def test_non_arxiv_url(self):
        # Still tries to find arxiv ID pattern
        result = _extract_arxiv_id("https://example.com/some-page")
        assert result is None

    def test_whitespace_handling(self):
        assert _extract_arxiv_id("  2203.17270  ") == "2203.17270"


class TestIsPdfPath:
    def test_existing_pdf(self, tmp_path):
        pdf = tmp_path / "test.pdf"
        pdf.write_text("dummy pdf content")
        assert _is_pdf_path(str(pdf)) is True

    def test_non_existent(self):
        assert _is_pdf_path("/nonexistent/path/paper.pdf") is False

    def test_not_pdf_extension(self, tmp_path):
        txt = tmp_path / "test.txt"
        txt.write_text("dummy")
        assert _is_pdf_path(str(txt)) is False


class TestFetchArxivMetadata:
    @patch("paper_resolver.requests.get")
    def test_successful_fetch(self, mock_get):
        # Minimal arXiv API response (simplified)
        mock_resp = MagicMock()
        mock_resp.text = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom">
  <entry>
    <title>Test Paper Title</title>
    <summary>Test abstract text.</summary>
    <published>2022-06-15</published>
    <author><name>Alice</name></author>
    <author><name>Bob</name></author>
    <link rel="alternate" href="https://arxiv.org/abs/2203.17270"/>
  </entry>
</feed>"""
        mock_resp.raise_for_status = MagicMock()
        mock_get.return_value = mock_resp

        meta = _fetch_arxiv_metadata("2203.17270")
        assert meta is not None
        assert meta["title"] == "Test Paper Title"
        assert meta["abstract"] == "Test abstract text."
        assert meta["year"] == 2022
        assert len(meta["authors"]) == 2

    @patch("paper_resolver.requests.get")
    def test_api_error(self, mock_get):
        mock_get.side_effect = Exception("Connection error")
        meta = _fetch_arxiv_metadata("2203.17270")
        assert meta is None
