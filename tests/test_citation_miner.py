"""Tests for the citation miner module."""
import pytest
from unittest.mock import MagicMock, patch

from paper import Paper, Reference, CitationType
from citation_miner import CitationMiner, _paper_from_ss, _ss_headers


class TestPaperFromSS:
    def test_extracts_fields(self):
        data = {
            "citedPaper": {
                "paperId": "ss:123",
                "title": "Test Paper",
                "authors": [{"name": "Alice"}, {"name": "Bob"}],
                "year": 2022,
                "abstract": "An abstract.",
                "citationCount": 100,
                "externalIds": {"ArXiv": "2201.00001"},
                "url": "https://arxiv.org/abs/2201.00001",
            }
        }
        paper = _paper_from_ss(data, prefix="citedPaper")
        assert paper.id == "ss:123"
        assert paper.title == "Test Paper"
        assert paper.arxiv_id == "2201.00001"
        assert paper.citation_count == 100
        assert len(paper.authors) == 2
        assert paper.source == "semantic_scholar"

    def test_missing_fields(self):
        data = {"citedPaper": {"paperId": "ss:min"}}
        paper = _paper_from_ss(data, prefix="citedPaper")
        assert paper.id == "ss:min"
        assert paper.title == ""
        assert paper.authors == []


class TestCitationMiner:
    def test_initial_state(self):
        miner = CitationMiner()
        assert len(miner.get_all_papers()) == 0
        assert miner.get_key_papers(top_k=5) == []

    def test_register_paper(self, sample_paper_v3):
        miner = CitationMiner()
        miner.register_paper(sample_paper_v3)
        assert len(miner.get_all_papers()) == 1

    @patch("citation_miner.requests.get")
    def test_fetch_references(self, mock_get, sample_paper_v3, mock_ss_references_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_ss_references_response
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        miner = CitationMiner()
        miner.register_paper(sample_paper_v3)
        # Use _fetch_references directly to test
        refs = miner._fetch_references(sample_paper_v3, top_k=5)
        assert len(refs) >= 1
        assert refs[0].paper_id == "ss:lss001"

    @patch("citation_miner.requests.get")
    def test_fetch_references_404(self, mock_get, sample_paper_v3):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        miner = CitationMiner()
        refs = miner._fetch_references(sample_paper_v3)
        assert refs == []

    @patch("citation_miner.requests.get")
    def test_fetch_citations(self, mock_get, sample_paper_v3, mock_ss_citations_response):
        mock_resp = MagicMock()
        mock_resp.json.return_value = mock_ss_citations_response
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        miner = CitationMiner()
        papers = miner._fetch_citations(sample_paper_v3)
        assert len(papers) >= 1
        assert papers[0].title == "BEVFusion: Multi-Task Multi-Sensor Fusion with Unified BEV"

    def test_classify_from_ss_intent(self):
        miner = CitationMiner()
        ref = Reference(paper_id="test:1", context="based on their approach")
        result = miner._classify_from_ss_intent(ref)
        assert result.citation_type == CitationType.SUPPORTING

    def test_classify_from_ss_intent_contrasting(self):
        miner = CitationMiner()
        ref = Reference(paper_id="test:2", context="however, their method has limitations")
        result = miner._classify_from_ss_intent(ref)
        assert result.citation_type == CitationType.CONTRASTING

    def test_get_key_papers_scoring(self, sample_paper_v3):
        miner = CitationMiner()
        # Register several papers with different citation counts
        high_cite = Paper(id="ss:high", title="High", citation_count=1000, year=2023)
        low_cite = Paper(id="ss:low", title="Low", citation_count=10, year=2020)
        miner.register_paper(sample_paper_v3)
        miner.register_paper(high_cite)
        miner.register_paper(low_cite)

        key = miner.get_key_papers(top_k=2)
        assert len(key) == 2
        assert key[0].id == "ss:high"

    def test_classify_references_no_llm(self, sample_paper_v3):
        miner = CitationMiner()
        miner._ref_map[sample_paper_v3.id] = [
            Reference(paper_id="ss:1", context="based on their work"),
            Reference(paper_id="ss:2", context="however, this fails when"),
        ]
        result = miner.classify_references(sample_paper_v3, llm_client=None)
        assert len(result) == 2
        assert result[0].citation_type == CitationType.SUPPORTING
        assert result[1].citation_type == CitationType.CONTRASTING
