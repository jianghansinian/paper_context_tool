"""Semantic Scholar API client — primary data source for V3.1.

Endpoints used:
  - GET /paper/search           — title-based paper search (seed resolution)
  - GET /paper/{id}/references  — backward citations
  - GET /paper/{id}/citations   — forward citations
  - GET /paper/{id}             — paper metadata

Rate limit: 1 req/s (with API key), 100/5min (without key).
No daily quota — ideal as primary data source.
"""

from __future__ import annotations

import json
import os
import time
import urllib.request
import urllib.error
import urllib.parse
from pathlib import Path
from typing import Optional

import config

_SS_API = "https://api.semanticscholar.org/graph/v1"
_SS_TIMEOUT = 30
_SS_USER_AGENT = "paper-context-tool/3.1 (mailto:research@example.com)"

# Fields to request from SS API (minimize response size)
_PAPER_FIELDS = "paperId,title,year,citationCount,authors,externalIds,abstract,publicationDate"
_SEARCH_FIELDS = "paperId,title,year,citationCount,authors,externalIds,abstract,publicationDate"

# Rate limiter state
_last_request_time = 0.0


def _has_api_key() -> bool:
    return bool(os.getenv("SS_API_KEY", "") or config.SS_API_KEY)


def _min_interval() -> float:
    """Minimum interval between requests. 1s with key, 3.5s without."""
    return 1.0 if _has_api_key() else 3.5


def _wait_for_rate_limit():
    """Enforce minimum interval between SS API requests."""
    global _last_request_time
    elapsed = time.time() - _last_request_time
    min_wait = _min_interval()
    if elapsed < min_wait:
        time.sleep(min_wait - elapsed)
    _last_request_time = time.time()


def _ss_request(
    path: str,
    params: dict | None = None,
    retries: int = 3,
) -> dict | None:
    """Make a request to Semantic Scholar REST API. Returns parsed JSON or None."""
    if params is None:
        params = {}
    qs = urllib.parse.urlencode(params)
    url = f"{_SS_API}/{path}"
    if qs:
        url += f"?{qs}"

    global _last_request_time
    _wait_for_rate_limit()

    # Without API key, retry once with a short wait — SS free tier is strict
    actual_retries = 1 if not _has_api_key() else retries

    for attempt in range(actual_retries):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", _SS_USER_AGENT)
            api_key = os.getenv("SS_API_KEY", config.SS_API_KEY)
            if api_key:
                req.add_header("x-api-key", api_key)
            with urllib.request.urlopen(req, timeout=_SS_TIMEOUT) as resp:
                result = json.loads(resp.read().decode())
                _last_request_time = time.time()
                return result
        except urllib.error.HTTPError as e:
            if e.code == 429:
                if not _has_api_key():
                    _last_request_time = time.time()
                    return None  # fail fast without key, let caller try OA
                wait = int(e.headers.get("Retry-After", str((attempt + 1) * 10)))
                print(f"  SS API: 429 rate limit, waiting {wait}s (attempt {attempt + 1}/{actual_retries})")
                time.sleep(wait)
                continue
            if e.code in (404, 400):
                _last_request_time = time.time()
                return None
            if attempt < actual_retries - 1:
                time.sleep((attempt + 1) * 5)
                continue
            _last_request_time = time.time()
            return None
        except Exception:
            if attempt < actual_retries - 1:
                time.sleep(3)
                continue
            _last_request_time = time.time()
            return None
    _last_request_time = time.time()
    return None


def _ss_paper_to_dict(paper: dict) -> dict | None:
    """Convert a Semantic Scholar paper object to the unified internal dict format."""
    title = (paper.get("title") or "").strip()
    if not title:
        return None

    external_ids = paper.get("externalIds") or {}
    arxiv_id = external_ids.get("ArXiv", "")

    authors = [a.get("name", "") for a in (paper.get("authors") or []) if a.get("name")]

    pub_date = paper.get("publicationDate") or ""
    month = 1
    if pub_date and "-" in pub_date:
        try:
            month = int(pub_date.split("-")[1])
        except (ValueError, IndexError):
            month = 1

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "year": paper.get("year") or 0,
        "month": month,
        "abstract": (paper.get("abstract") or "").strip(),
        "citation_count": paper.get("citationCount") or 0,
        "source": "semantic_scholar",
        "_ss_id": paper.get("paperId", ""),
        "_oa_id": "",
        "_doi": external_ids.get("DOI", ""),
    }


def _ss_search(query: str, limit: int = 10) -> list[dict]:
    """Search Semantic Scholar for papers matching a query string.

    Used primarily for seed resolution (title-based exact search).
    """
    params = {
        "query": query,
        "limit": str(min(limit, 100)),
        "fields": _SEARCH_FIELDS,
    }
    data = _ss_request("paper/search", params)
    if not data:
        return []

    results = []
    for paper in data.get("data", []):
        d = _ss_paper_to_dict(paper)
        if d:
            results.append(d)
    return results


def _ss_get_references(paper_id: str, limit: int = 30) -> list[dict]:
    """Fetch papers that the given paper CITES (backward references).

    Args:
        paper_id: Semantic Scholar paper ID (e.g. "ArXiv:2203.17270" or CorpusId:12345)
        limit: max references to return.
    """
    if not paper_id:
        return []
    paper_id_enc = urllib.parse.quote(paper_id, safe="")
    params = {
        "limit": str(min(limit, 500)),
        "fields": _PAPER_FIELDS,
    }
    data = _ss_request(f"paper/{paper_id_enc}/references", params)
    if not data:
        return []

    results = []
    for entry in data.get("data", []):
        cited = entry.get("citedPaper")
        if not cited:
            continue
        d = _ss_paper_to_dict(cited)
        if d:
            d["source"] = "citation_expansion"
            results.append(d)
    return results


def _ss_get_citations(paper_id: str, limit: int = 30) -> list[dict]:
    """Fetch papers that CITE the given paper (forward citations).

    Args:
        paper_id: Semantic Scholar paper ID.
        limit: max citations to return.
    """
    if not paper_id:
        return []
    paper_id_enc = urllib.parse.quote(paper_id, safe="")
    params = {
        "limit": str(min(limit, 500)),
        "fields": _PAPER_FIELDS,
    }
    data = _ss_request(f"paper/{paper_id_enc}/citations", params)
    if not data:
        return []

    results = []
    for entry in data.get("data", []):
        citing = entry.get("citingPaper")
        if not citing:
            continue
        d = _ss_paper_to_dict(citing)
        if d:
            d["source"] = "forward_citation"
            results.append(d)
    return results


def _ss_get_paper(paper_id: str) -> dict | None:
    """Fetch full paper metadata from Semantic Scholar."""
    if not paper_id:
        return None
    paper_id_enc = urllib.parse.quote(paper_id, safe="")
    params = {"fields": _PAPER_FIELDS}
    data = _ss_request(f"paper/{paper_id_enc}", params)
    if not data:
        return None
    return _ss_paper_to_dict(data)


def _ss_field_search(field_name: str, limit: int = 20) -> list[dict]:
    """Search for top papers in a field (used as fallback when LLM seed generation fails).

    Searches by field name and returns papers sorted by citation count.
    """
    params = {
        "query": field_name,
        "limit": str(min(limit, 100)),
        "fields": _SEARCH_FIELDS,
    }
    data = _ss_request("paper/search", params)
    if not data:
        return []

    results = []
    for paper in data.get("data", []):
        d = _ss_paper_to_dict(paper)
        if d:
            results.append(d)

    results.sort(key=lambda p: -p.get("citation_count", 0))
    return results[:limit]
