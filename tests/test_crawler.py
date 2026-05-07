import sys
from pathlib import Path

SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(SRC))

from crawler import _dedupe_papers, _paper_key, _normalize_space


class TestNormalizeSpace:
    def test_collapse_whitespace(self):
        assert _normalize_space("hello   world") == "hello world"

    def test_trim_edges(self):
        assert _normalize_space("  text  ") == "text"

    def test_none_returns_empty(self):
        assert _normalize_space(None) == ""


class TestPaperKey:
    def test_lowercase_and_strip(self):
        assert _paper_key({"title": "  HELLO World  "}) == "hello world"

    def test_missing_title(self):
        assert _paper_key({}) == ""


class TestDedupePapers:
    def test_remove_duplicates(self):
        papers = [
            {"title": "Title A", "abstract": "..."},
            {"title": "Title B", "abstract": "..."},
            {"title": "Title A", "abstract": "different abstract"},
        ]
        result = _dedupe_papers(papers)
        assert len(result) == 2
        titles = [p["title"] for p in result]
        assert titles == ["Title A", "Title B"]

    def test_skip_empty_title(self):
        papers = [
            {"title": "", "abstract": "..."},
            {"title": "Valid Title", "abstract": "..."},
            {"title": None, "abstract": "..."},
        ]
        result = _dedupe_papers(papers)
        assert len(result) == 1
        assert result[0]["title"] == "Valid Title"

    def test_dedupe_case_insensitive(self):
        papers = [
            {"title": "MY PAPER TITLE", "abstract": "..."},
            {"title": "my paper title", "abstract": "..."},
        ]
        result = _dedupe_papers(papers)
        assert len(result) == 1
