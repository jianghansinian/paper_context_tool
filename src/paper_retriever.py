"""Paper retrieval V2: semantic-first, information-flow-inverted.

Core change from V1: retrieve broadly FIRST, then let LLM judge importance.
LLM is demoted from "judge" (generating milestone list upfront) to
"translator + final arbiter" (query expansion + post-hoc selection).

Pipeline:
  1. Query Expansion (1 LLM call) — search queries, NO milestone list
  2. Multi-source Broad Recall — SS semantic (primary) + arXiv keyword (auxiliary)
  3. Citation Expansion — top-30 one-hop, no seed limit
  4. Survey Calibration — extract refs from a recent survey
  5. Embedding-based Seminal Matching — cosine similarity, three-band threshold
  6. LLM Unified Selection (1 LLM call) — classify + select + list missing
  7. Closed-loop Recovery — SS-based search for missing papers
  8. Final Output — selected papers + confirmed_missing

Usage:
    from paper_retriever import retrieve_field_papers

    papers, report = retrieve_field_papers("BEV Perception", client)
    # papers: list[dict], report: dict with confirmed_missing
"""

from __future__ import annotations

import json
import math
import os
import re
import threading
import time
import urllib.request
import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from collections import defaultdict
from typing import Optional

import numpy as np
import requests
from openai import OpenAI

from llm_analyzer import _resolve_model, _extract_json_object
from embedding import build_embedding_client, get_embedding, _local_embedding
import config


# ═══════════════════════════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════════════════════════

ARXIV_API = "http://export.arxiv.org/api/query"
ARXIV_TIMEOUT = 30

_SS_API_BASE = "https://api.semanticscholar.org/graph/v1"
_SS_FIELDS = "title,authors,year,citationCount,externalIds,abstract"
_SS_TIMEOUT = 15

# Number of top-cited papers to use as seeds for citation expansion.
# No hard limit — all top-N are used.
_CITATION_EXPANSION_SEEDS = 30
_SS_REFS_PER_SEED = 15
_SS_CITES_PER_SEED = 15

# Embedding similarity thresholds for three-band classification
_EMBED_HIGH_THRESHOLD = 0.75  # Calibrated: DETR3D sim=0.796 is correct match for its milestone
_EMBED_LOW_THRESHOLD = 0.60


# ═══════════════════════════════════════════════════════════════════════════════
# arXiv API Search (auxiliary)
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_arxiv_date(published_str: str) -> tuple[int, int]:
    try:
        return int(published_str[:4]), int(published_str[5:7])
    except (ValueError, IndexError):
        return 0, 0


def search_arxiv(query: str, max_results: int = 100, search_field: str = "all") -> list[dict]:
    """Search arXiv for papers matching a query (auxiliary source).

    Args:
        query: search query string.
        max_results: max papers to return.
        search_field: arXiv search field — \"all\", \"ti\" (title), \"abs\" (abstract).

    Returns list of dicts: {arxiv_id, title, year, month, abstract}.
    """
    papers = []
    search_query = f"{search_field}:{query}"
    params = urllib.parse.urlencode({
        "search_query": search_query,
        "start": 0,
        "max_results": max_results,
        "sortBy": "relevance",
    })
    url = f"{ARXIV_API}?{params}"

    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", "PaperContextTool/1.0")
            with urllib.request.urlopen(req, timeout=ARXIV_TIMEOUT) as resp:
                raw = resp.read().decode("utf-8")
            break
        except urllib.error.HTTPError as e:
            if e.code in (503, 429):
                wait = (attempt + 1) * 15
                print(f"  arXiv {e.code}, waiting {wait}s...")
                time.sleep(wait)
                continue
            print(f"  arXiv HTTP {e.code}: {e.reason}")
            return papers
        except Exception as e:
            if attempt < 2:
                time.sleep(5)
                continue
            print(f"  arXiv API request failed: {e}")
            return papers
    else:
        print("  arXiv API: all retries exhausted")
        return papers

    ns = {"atom": "http://www.w3.org/2005/Atom",
          "arxiv": "http://arxiv.org/schemas/atom"}
    try:
        root = ET.fromstring(raw)
    except ET.ParseError as e:
        print(f"  arXiv XML parse error: {e}")
        return papers

    for entry in root.findall("atom:entry", ns):
        title_el = entry.find("atom:title", ns)
        summary_el = entry.find("atom:summary", ns)
        published_el = entry.find("atom:published", ns)

        title = title_el.text.strip().replace("\n", " ") if title_el is not None and title_el.text else ""
        abstract = summary_el.text.strip().replace("\n", " ") if summary_el is not None and summary_el.text else ""
        published = published_el.text if published_el is not None else ""

        year, month = _parse_arxiv_date(published)

        id_el = entry.find("atom:id", ns)
        arxiv_id = ""
        if id_el is not None and id_el.text:
            arxiv_url = id_el.text.strip()
            m = re.search(r"abs/([\w.\-]+?)(?:v\d+)?$", arxiv_url)
            if m:
                arxiv_id = m.group(1)

        if not arxiv_id and title:
            for link in entry.findall("atom:link", ns):
                href = link.get("href", "")
                m = re.search(r"abs/([\w.\-]+?)(?:v\d+)?$", href)
                if m:
                    arxiv_id = m.group(1)
                    break

        if not title:
            continue

        papers.append({
            "arxiv_id": arxiv_id,
            "title": title,
            "year": year,
            "month": month,
            "abstract": abstract,
            "citation_count": 0,
            "source": "arxiv",
        })

    return papers


# ═══════════════════════════════════════════════════════════════════════════════
# Semantic Scholar API (primary search + citation data)
# ═══════════════════════════════════════════════════════════════════════════════

def _ss_headers() -> dict:
    h = {}
    if config.SS_API_KEY:
        h["x-api-key"] = config.SS_API_KEY
    return h


def _ss_paper_to_dict(ss_data: dict) -> Optional[dict]:
    """Convert SS paper data to our candidate dict format."""
    title = (ss_data.get("title") or "").strip()
    if not title:
        return None
    external = ss_data.get("externalIds") or {}
    arxiv_id = external.get("ArXiv", "")
    authors = [a.get("name", "") for a in (ss_data.get("authors") or [])]
    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "year": ss_data.get("year") or 0,
        "month": 1,
        "abstract": ss_data.get("abstract") or "",
        "citation_count": ss_data.get("citationCount") or 0,
        "source": "ss",
    }


def _ss_semantic_search(query: str, limit: int = 100) -> list[dict]:
    """Search Semantic Scholar by semantic similarity (primary retrieval source).

    SS covers title+abstract, so papers like LSS whose title lacks
    the search term but whose abstract contains it ARE found.
    """
    try:
        params = urllib.parse.urlencode({
            "query": query,
            "limit": limit,
            "fields": _SS_FIELDS,
        })
        url = f"{_SS_API_BASE}/paper/search?{params}"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "PaperContextTool/1.0")
        for k, v in _ss_headers().items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=_SS_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  SS search failed for '{query[:60]}': {e}")
        return []

    results = []
    for item in data.get("data", []):
        d = _ss_paper_to_dict(item)
        if d:
            results.append(d)
    return results


def _ss_title_search(query: str, limit: int = 5) -> list[dict]:
    """Search Semantic Scholar by title keywords.

    Used as a recovery path to catch non-arXiv papers during closed-loop补搜.
    """
    return _ss_semantic_search(query, limit=limit)


def _ss_batch_enrich(papers: list[dict], field_name: str) -> list[dict]:
    """Enrich papers with citation counts from SS. Best-effort, silent on failure."""
    if not papers:
        return papers

    try:
        params = urllib.parse.urlencode({
            "query": field_name,
            "limit": min(len(papers) * 2, 100),
            "fields": "title,citationCount,externalIds",
        })
        url = f"{_SS_API_BASE}/paper/search?{params}"
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "PaperContextTool/1.0")
        for k, v in _ss_headers().items():
            req.add_header(k, v)
        with urllib.request.urlopen(req, timeout=_SS_TIMEOUT) as resp:
            data = json.loads(resp.read().decode())
    except Exception:
        return papers

    ss_results = data.get("data", [])
    if not ss_results:
        return papers

    ss_lookup = {}
    for item in ss_results:
        t = (item.get("title") or "").lower().strip()
        cc = item.get("citationCount") or 0
        if t:
            ss_lookup[t] = cc

    enriched = 0
    for p in papers:
        if p.get("citation_count", 0) > 0:
            continue
        t = p["title"].lower().strip()
        if t in ss_lookup and ss_lookup[t] > 0:
            p["citation_count"] = ss_lookup[t]
            enriched += 1
        else:
            for ss_t, cc in ss_lookup.items():
                if t[:50] == ss_t[:50] or (len(t) > 30 and t[:30] in ss_t):
                    p["citation_count"] = cc
                    enriched += 1
                    break

    if enriched:
        print(f"  SS enriched {enriched} papers with citation counts")
    return papers


# ═══════════════════════════════════════════════════════════════════════════════
# OpenAlex API (primary search + citation data, free, no key required)
# ═══════════════════════════════════════════════════════════════════════════════

_OPENALEX_API = "https://api.openalex.org"
_OPENALEX_USER_AGENT = "mailto:research@example.com"
_OPENALEX_TIMEOUT = 30


_OA_RATE_LOCK = threading.Lock()
_OA_LAST_REQUEST = 0.0


def _oa_throttle(min_interval: float = 0.25) -> None:
    """Global OpenAlex rate limiter shared by all worker threads (~4 req/s max)."""
    global _OA_LAST_REQUEST
    with _OA_RATE_LOCK:
        now = time.monotonic()
        wait = _OA_LAST_REQUEST + min_interval - now
        if wait > 0:
            time.sleep(wait)
        _OA_LAST_REQUEST = time.monotonic()


def _openalex_request(path: str, params: dict | None = None) -> dict | None:
    """Make a request to OpenAlex REST API. Returns parsed JSON or None on failure."""
    if params is None:
        params = {}
    _oa_throttle()
    qs = urllib.parse.urlencode(params)
    # OpenAlex uses standard URI encoding (space = %20), not form encoding (+).
    # urlencode encodes spaces as + by default; replace them.
    qs = qs.replace("+", "%20")
    url = f"{_OPENALEX_API}/{path}"
    if qs:
        url += f"?{qs}"
    for attempt in range(3):
        try:
            req = urllib.request.Request(url)
            req.add_header("User-Agent", _OPENALEX_USER_AGENT)
            api_key = os.getenv("OPENALEX_API_KEY", config.OPENALEX_API_KEY)
            if api_key:
                req.add_header("Authorization", f"Bearer {api_key}")
            with urllib.request.urlopen(req, timeout=_OPENALEX_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                wait = (attempt + 1) * 10
                time.sleep(wait)
                continue
            if e.code == 403:
                body = e.read().decode(errors="ignore")
                if "Insufficient budget" in body or "budget" in body.lower():
                    print("  OpenAlex: daily budget exhausted (HTTP 403).")
                    return None
            return None
        except Exception:
            if attempt < 2:
                time.sleep(5)
                continue
            return None
    return None


def _openalex_abstract(abstract_inverted: dict | None) -> str:
    """Reconstruct abstract text from OpenAlex inverted-index format."""
    if not abstract_inverted:
        return ""
    # Build word → position list
    words = []
    for word, positions in abstract_inverted.items():
        for pos in positions:
            words.append((pos, word))
    words.sort(key=lambda x: x[0])
    return " ".join(w for _, w in words)


def _openalex_paper_to_dict(work: dict) -> dict | None:
    """Convert OpenAlex work to unified candidate dict format."""
    title = (work.get("title") or "").strip()
    if not title:
        return None

    # Extract arXiv ID from OpenAlex IDs
    arxiv_id = ""
    ids = work.get("ids", {})
    if ids:
        oa_id = ids.get("openalex", "")
        # Format: https://openalex.org/W123456
        if oa_id:
            arxiv_id = f"oa:{oa_id.split('/')[-1]}"
        # Also check for arXiv-linked paper using primary location
    primary_loc = work.get("primary_location") or {}
    landing_url = primary_loc.get("landing_page_url") or ""
    if "arxiv.org" in landing_url:
        m = re.search(r"abs/([\w.\-]+)", landing_url)
        if m:
            arxiv_id = m.group(1)

    abstract = _openalex_abstract(work.get("abstract_inverted_index"))

    authors = []
    for a in work.get("authorships", []):
        name = a.get("author", {}).get("display_name", "")
        if name:
            authors.append(name)

    return {
        "arxiv_id": arxiv_id,
        "title": title,
        "authors": authors,
        "year": work.get("publication_year") or 0,
        "month": 1,
        "abstract": abstract,
        "citation_count": work.get("cited_by_count") or 0,
        "source": "openalex",
        "type": work.get("type") or "",
        "_oa_id": ids.get("openalex", ""),
        "_doi": ids.get("doi", "").replace("https://doi.org/", ""),
        "_raw_source_name": (primary_loc.get("source") or {}).get("display_name") or "",
    }


# V3.3.15: scholarly work types admitted by _openalex_search (client-side;
# OA type filter doesn't support OR syntax). conference-paper is a distinct
# OA type for CS venues — the old type:article filter excluded it too.
# Graph-node conversion is NOT filtered — a cited dataset is a valid node.
_ALLOWED_SEARCH_TYPES = {"article", "conference-paper", "preprint", "review"}


def _openalex_search(query: str, limit: int = 50) -> list[dict]:
    """Search OpenAlex for papers matching a query.

    OpenAlex search covers titles + abstracts + concepts, providing semantic-like
    retrieval without requiring an API key. Rate limit: ~10 req/s for anonymous access.
    """
    results = []
    per_page = min(limit, 200)
    page = 1
    collected = 0

    while collected < limit:
        params = {
            "search": query,
            "per_page": str(per_page),
            "page": str(page),
        }
        data = _openalex_request("works", params)
        if not data:
            break

        for work in data.get("results", []):
            d = _openalex_paper_to_dict(work)
            # V3.3.15: OA type filter rejects OR syntax (HTTP 400), so filter
            # client-side. Include preprint — arXiv-only papers are recorded as
            # preprint in OA; type:article alone excluded them (StreamPETR etc.)
            if d and d.get("type") in _ALLOWED_SEARCH_TYPES:
                results.append(d)
                collected += 1
                if collected >= limit:
                    break

        # Check if more pages available
        meta = data.get("meta", {})
        if meta.get("per_page", 0) * page >= meta.get("count", 0):
            break
        page += 1
        time.sleep(0.3)  # Polite delay

    return results


def _openalex_enrich(candidates: list[dict], field_name: str) -> list[dict]:
    """Enrich candidates with citation counts from OpenAlex.

    Uses bulk title matching against a broader OpenAlex search for the field.
    Much more robust than SS without API key.
    """
    if not candidates:
        return candidates

    # Search OpenAlex for the field broadly to build a citation lookup
    all_results = _openalex_search(field_name, limit=200)
    if not all_results:
        return candidates

    # Build title → citation_count lookup
    lookup = {}
    for r in all_results:
        t = r["title"].lower().strip()
        cc = r.get("citation_count", 0)
        if t and cc:
            lookup[t] = max(lookup.get(t, 0), cc)

    # Enrich candidates by fuzzy title match
    enriched = 0
    for c in candidates:
        if c.get("citation_count", 0) > 0:
            continue
        t = c["title"].lower().strip()
        # Exact match
        if t in lookup:
            c["citation_count"] = lookup[t]
            enriched += 1
        else:
            # Fuzzy match: first 50 chars or first 30 chars in either direction
            for oa_t, cc in lookup.items():
                if t[:50] == oa_t[:50] or (len(t) > 30 and t[:30] in oa_t) or (len(oa_t) > 30 and oa_t[:30] in t):
                    c["citation_count"] = cc
                    enriched += 1
                    break

    if enriched:
        print(f"  OpenAlex enriched {enriched} papers with citation counts")
    return candidates


def _openalex_get_references(oa_id: str, limit: int = 30) -> list[dict]:
    """Fetch referenced works for a given OpenAlex paper ID.

    Uses OA's filter=openalex_id batch query to fetch all references in
    ~2-3 requests instead of one-per-reference (saves 97%+ API calls).
    """
    if not oa_id:
        return []
    # Extract the W-prefixed ID portion
    oa_short = oa_id.split("/")[-1] if "/" in oa_id else oa_id
    work_data = _openalex_request(f"works/{oa_short}")
    if not work_data:
        return []

    ref_ids = work_data.get("referenced_works", [])[:limit]
    if not ref_ids:
        return []

    # Batch: OA supports filter=openalex_id:id1|id2|id3|... (up to ~50 per batch)
    BATCH_SIZE = 50
    results = []
    for batch_start in range(0, len(ref_ids), BATCH_SIZE):
        batch = ref_ids[batch_start:batch_start + BATCH_SIZE]
        ids = [u.split("/")[-1] for u in batch]
        filter_str = "openalex_id:" + "|".join(ids)
        data = _openalex_request("works", {"filter": filter_str, "per_page": str(BATCH_SIZE)})
        if data:
            for work in data.get("results", []):
                d = _openalex_paper_to_dict(work)
                if d:
                    d["source"] = "citation_expansion"
                    results.append(d)
        time.sleep(0.15)

    return results


def _openalex_get_citations(oa_id: str, limit: int = 30) -> list[dict]:
    """Fetch papers that cite the given OpenAlex paper ID (forward citations).

    Uses OA's filter=cites:{oa_short} to find citing papers.
    V3.3.10: mixed sampling — (1-f)·limit top-cited + f·limit newest, so that
    recent papers (low citation counts) are not systematically cut off by the
    cited_by_count sort. Returns list of paper dicts in unified format.
    """
    if not oa_id:
        return []
    oa_short = oa_id.split("/")[-1] if "/" in oa_id else oa_id
    params = {
        "filter": f"cites:{oa_short}",
        "per_page": "200",
        "sort": "cited_by_count:desc",
    }
    data = _openalex_request("works", params)
    if not data:
        return []

    converted = []
    for work in data.get("results", []):
        d = _openalex_paper_to_dict(work)
        if d:
            d["source"] = "forward_citation"
            converted.append(d)

    if len(converted) <= limit:
        return converted

    recent_fraction = config.V3_FORWARD_RECENT_FRACTION
    n_recent = max(0, int(limit * recent_fraction))
    n_top = limit - n_recent
    top = converted[:n_top]

    seen = {t.get("title", "").lower().strip() for t in top}
    recent = []
    for d in sorted(converted, key=lambda x: -(x.get("year") or 0)):
        if len(recent) >= n_recent:
            break
        t = d.get("title", "").lower().strip()
        if t in seen:
            continue
        recent.append(d)
        seen.add(t)

    return top + recent


def _openalex_survey_search(field_name: str) -> list[dict]:
    """Find recent survey/review papers for a field from OpenAlex."""
    query = f"{field_name} survey review"
    results = _openalex_search(query, limit=20)

    # Filter to likely surveys: title contains survey/review/comprehensive/overview
    surveys = []
    for r in results:
        t = r.get("title", "").lower()
        if any(w in t for w in ["survey", "review", "comprehensive", "overview", "taxonomy"]):
            surveys.append(r)

    if surveys:
        print(f"  OpenAlex survey search: {len(surveys)} candidates from {len(results)} results")
    return surveys


# ═══════════════════════════════════════════════════════════════════════════════
# Deduplication
# ═══════════════════════════════════════════════════════════════════════════════

def _deduplicate(papers: list[dict]) -> list[dict]:
    """Deduplicate by arxiv_id first, then by fuzzy title match."""
    seen_ids = set()
    result = []
    for p in papers:
        aid = p.get("arxiv_id", "")
        if aid and aid in seen_ids:
            continue
        if aid:
            seen_ids.add(aid)
        result.append(p)

    seen_titles = []
    final = []
    for p in result:
        t = p["title"].lower().strip()
        is_dup = False
        for s in seen_titles:
            if t[:50] == s[:50] or (len(t) > 30 and t[:30] in s) or (len(s) > 30 and s[:30] in t):
                is_dup = True
                break
        if not is_dup:
            seen_titles.append(t)
            final.append(p)
    return final


def _match_title(title: str, candidates: list[dict]) -> Optional[dict]:
    """Fuzzy match a title back to a candidate (for LLM output re-mapping)."""
    t = title.lower().strip()
    for c in candidates:
        ct = c["title"].lower().strip()
        if t == ct:
            return c
        if t[:50] == ct[:50]:
            return c
        if len(t) > 30 and t[:30] in ct:
            return c
        if len(ct) > 30 and ct[:30] in t:
            return c
    for c in candidates:
        ct = c["title"].lower().strip()
        if t[:25] == ct[:25]:
            return c
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1.2.5: Relevance-based Pre-rank
# ═══════════════════════════════════════════════════════════════════════════════

_PRE_RANK_TOP_N = 150


def _pre_rank(
    candidates: list[dict],
    field_specific_terms: list[str],
    exclusion_terms: list[str],
    top_n: int = _PRE_RANK_TOP_N,
) -> list[dict]:
    """Step 1.2.5: Score and rank candidates by field relevance.

    V2.3: Uses core_field_markers (narrow, distinguishing terms) for scoring.
    Callers pass core_field_markers as field_specific_terms — the parameter name
    is kept for backward compatibility.

    Does NOT delete papers — all candidates get a relevance_score.
    Returns all candidates sorted by score (descending).
    Callers should use only the top-N for LLM input.
    """
    if not candidates:
        return candidates

    # Normalize terms for matching
    spec_terms_lower = [t.lower().strip() for t in field_specific_terms if t.strip()]
    excl_terms_lower = [t.lower().strip() for t in exclusion_terms if t.strip()]

    for c in candidates:
        title_lower = c.get("title", "").lower()
        abstract_lower = c.get("abstract", "").lower()
        text_lower = title_lower + " " + abstract_lower

        # Title keyword hit ratio
        title_hits = sum(1 for t in spec_terms_lower if t in title_lower)
        title_hit_ratio = title_hits / max(len(spec_terms_lower), 1)

        # Abstract keyword hit ratio
        abstract_hits = sum(1 for t in spec_terms_lower if t in abstract_lower)
        abstract_hit_ratio = abstract_hits / max(len(spec_terms_lower), 1)

        # Exclusion penalty
        exclusion_penalty = 0.0
        for et in excl_terms_lower:
            if et in text_lower:
                exclusion_penalty = -1.0
                break

        c["relevance_score"] = (
            title_hit_ratio * 0.6
            + abstract_hit_ratio * 0.4
            + exclusion_penalty
        )

    candidates.sort(key=lambda c: (-c.get("relevance_score", 0), -c.get("citation_count", 0)))

    n_above = sum(1 for c in candidates if c.get("relevance_score", 0) > 0)
    n_zero = sum(1 for c in candidates if c.get("relevance_score", 0) == 0)
    n_below = sum(1 for c in candidates if c.get("relevance_score", 0) < 0)
    print(f"  Pre-rank: {n_above} relevant, {n_zero} neutral, {n_below} excluded (top-N={top_n})")
    if candidates:
        top = candidates[0]
        print(f"    Top: {top['title'][:70]} (score={top.get('relevance_score',0):.3f}, cit={top.get('citation_count',0)})")

    return candidates


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1.1: Query Expansion (1 LLM call)
# ═══════════════════════════════════════════════════════════════════════════════

_QUERY_EXPANSION_SYSTEM = """\
You are a research librarian who maps the terminology landscape of academic fields. \
Given a field name, you generate search queries designed to maximize recall.

Your role is TRANSLATOR: you convert a field name into the diverse vocabulary that \
different research communities use to describe similar concepts. You do NOT judge \
which papers are important — you only ensure we search with the right words.

Return ONLY a JSON object. No other text."""

_QUERY_EXPANSION_PROMPT = """\
Generate search queries for: **{field_name}**

Goal: produce queries that will find papers spanning the FULL evolution of this field — \
from earliest foundational work through latest SOTA.

1. BROAD QUERIES: 3-5 wide queries for semantic search. Use natural-language descriptions \
   of the core problems this field addresses. Include terminology from different eras \
   (older papers use different words than recent ones). These will be sent to a semantic \
   search engine, so use conceptual descriptions, not just keywords.

2. SPECIFIC QUERIES: 5-8 targeted queries for keyword search. Include:
   - Specific model/algorithm/technique names commonly associated with this field
   - Sub-problem formulations
   - Task descriptions using precise technical terms
   These will be sent to a keyword-based search engine (arXiv), so be precise.

3. SYNONYMS AND VARIANTS: Map the main concepts to their alternative names, abbreviations, \
   and rephrasings used in different sub-communities or time periods.

4. DISAMBIGUATION: If the field name contains abbreviations that could be confused with \
   other fields (e.g., BEV = Bird's Eye View, NOT Battery Electric Vehicle), provide THREE lists:
   - core_field_markers: 3-8 words/phrases that are UNIQUE to this field — they rarely appear \
     in papers from other fields. These alone should distinguish this field from lookalikes. \
     Example for BEV: "bird's eye view", "view transformation", "surround view", "lift splat". \
     Do NOT include broad terms like "perception" or "object detection" that appear in all CV papers.
   - field_specific_terms: broader words/phrases relevant to this field (can include common terms \
     like "perception", "autonomous driving" — these are used for ranking/scoring, not hard filtering)
   - exclusion_terms: words/phrases that indicate the paper is about a DIFFERENT field \
     and should be excluded

Return JSON:
```json
{{
  "broad_queries": ["natural language description 1", "..."],
  "specific_queries": ["keyword query 1", "..."],
  "synonyms_and_variants": {{
    "concept_name": ["variant1", "variant2"]
  }},
  "disambiguation": {{
    "core_field_markers": ["unique distinguishing term", "..."],
    "field_specific_terms": ["term that must appear in relevant papers", "..."],
    "exclusion_terms": ["term that indicates wrong field", "..."]
  }}
}}
```"""


def _generate_query_expansion(
    field_name: str,
    client: OpenAI,
    model: Optional[str] = None,
) -> dict:
    """Step 1.1: LLM generates search queries, NOT milestone lists."""
    model = _resolve_model(model)
    if not model or not client:
        return {
            "broad_queries": [field_name],
            "specific_queries": [field_name],
            "synonyms_and_variants": {},
        }

    prompt = _QUERY_EXPANSION_PROMPT.format(field_name=field_name)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _QUERY_EXPANSION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1024,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = resp.choices[0].message.content or ""
        data = _extract_json_object(raw)
        if not data:
            print("  Query expansion: failed to parse JSON, falling back to single query")
            return {
                "broad_queries": [field_name],
                "specific_queries": [field_name],
                "synonyms_and_variants": {},
            }

        n_broad = len(data.get("broad_queries", []))
        n_specific = len(data.get("specific_queries", []))
        n_syn = sum(len(v) for v in data.get("synonyms_and_variants", {}).values())
        print(f"  Query expansion: {n_broad} broad queries, {n_specific} specific queries,"
              f" {n_syn} synonym variants")
        return data
    except Exception as e:
        print(f"  Query expansion failed: {e}")
        return {
            "broad_queries": [field_name],
            "specific_queries": [field_name],
            "synonyms_and_variants": {},
        }


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1.2: Multi-source Broad Recall
# ═══════════════════════════════════════════════════════════════════════════════

def _broad_recall(query_expansion: dict, field_name: str) -> list[dict]:
    """Step 1.2: OpenAlex search (primary) + arXiv keyword search (auxiliary).

    Results merged and deduplicated. NO filtering — all papers enter the pool.
    """
    all_papers = []
    broad_queries = query_expansion.get("broad_queries", [field_name])
    specific_queries = query_expansion.get("specific_queries", [])

    # ── Primary: OpenAlex semantic search ──
    print(f"\n  [OpenAlex Search] {len(broad_queries)} queries")
    for q in broad_queries[:3]:  # Limit to top 3 broad queries for OpenAlex
        if not q:
            continue
        results = _openalex_search(q, limit=200)
        if results:
            all_papers.extend(results)
            print(f"    '{q[:60]}': {len(results)} results")
        else:
            print(f"    '{q[:60]}': 0 results")
        time.sleep(1.0)

    # ── Also search broad queries on arXiv for coverage ──
    all_keyword_queries = list(specific_queries)
    # Add field name + some broad queries for arXiv
    all_keyword_queries.append(field_name)
    if broad_queries:
        # Take the shortest broad query — best for keyword search
        all_keyword_queries.append(min(broad_queries, key=len))
    if not all_keyword_queries:
        all_keyword_queries = [field_name]

    print(f"\n  [arXiv Keyword Search] {len(all_keyword_queries)} queries")
    for q in all_keyword_queries:
        if not q:
            continue
        results = search_arxiv(q, max_results=100)
        if results:
            all_papers.extend(results)
            print(f"    '{q[:60]}': {len(results)} results")
        else:
            print(f"    '{q[:60]}': 0 results")
        time.sleep(3.0)

    papers = _deduplicate(all_papers)
    print(f"\n  Broad recall: {len(all_papers)} raw → {len(papers)} after dedup")
    return papers


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1.3: Citation Expansion
# ═══════════════════════════════════════════════════════════════════════════════

def _citation_expand(
    candidates: list[dict],
    field_name: str = "",
    core_field_markers: list[str] | None = None,
    exclusion_terms: list[str] | None = None,
) -> list[dict]:
    """Step 1.3: One-hop citation expansion from field-relevant high-cited papers.

    V2.2: Seeds must pass field relevance check using core_field_markers
    (narrow, distinguishing terms only), NOT broad field_specific_terms.
    """
    core_terms = [t.lower() for t in (core_field_markers or [])]
    excl_terms = [t.lower() for t in (exclusion_terms or [])]

    def _is_field_relevant(paper: dict) -> bool:
        title = paper.get("title", "").lower()
        abstract = paper.get("abstract", "").lower()
        text = title + " " + abstract
        # V2.2: Use core_field_markers (narrow, distinguishing terms) for filtering
        if core_terms and not any(t in text for t in core_terms):
            return False
        # Must NOT contain any exclusion term
        if excl_terms and any(t in text for t in excl_terms):
            return False
        return True

    # Take field-relevant top-N by citation count as seeds
    with_cit = [c for c in candidates if c.get("citation_count", 0) > 0 and _is_field_relevant(c)]
    n_seeds = min(_CITATION_EXPANSION_SEEDS, 15)
    seeds = sorted(with_cit, key=lambda c: -c.get("citation_count", 0))[:n_seeds]

    if len(seeds) < 2:
        print(f"  Citation expansion: {len(seeds)} field-relevant seeds (need ≥2), skipping")
        return []

    print(f"\n  [Citation Expansion] {len(seeds)} field-relevant seeds")

    existing_titles = {c["title"].lower().strip() for c in candidates}
    discovered: dict[str, dict] = {}  # keyed by title for dedup

    for i, seed in enumerate(seeds):
        oa_id = seed.get("_oa_id", "")
        title_snippet = seed["title"][:60]

        if not oa_id:
            # Try to find OpenAlex ID by searching for the title
            title_search = _openalex_search(seed["title"], limit=3)
            if title_search and title_search[0].get("_oa_id"):
                oa_id = title_search[0]["_oa_id"]
                seed["_oa_id"] = oa_id  # cache for later

        if not oa_id:
            continue

        # Fetch references from OpenAlex
        refs = _openalex_get_references(oa_id, limit=_SS_REFS_PER_SEED)
        for d in refs:
            t = d["title"].lower().strip()
            if t not in existing_titles and t not in discovered:
                d["source"] = "citation_expansion"
                discovered[t] = d

        print(f"    seed {i+1}/{len(seeds)}: +{len(refs)} refs from '{title_snippet}'")

    new_papers = list(discovered.values())
    if new_papers:
        print(f"  Citation expansion: +{len(new_papers)} papers from {len(seeds)} seeds")
    else:
        print(f"  Citation expansion: no new papers discovered")
    return new_papers


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1.4: Survey Calibration (0-1 LLM call, optional)
# ═══════════════════════════════════════════════════════════════════════════════

def _survey_calibrate(
    candidates: list[dict],
    field_name: str,
    client: OpenAI,
    model: Optional[str] = None,
    field_specific_terms: list[str] | None = None,
    exclusion_terms: list[str] | None = None,
) -> list[dict]:
    """Step 1.4: Find a field-relevant survey paper and extract its reference list.

    V2.1: Surveys must pass field relevance check to avoid BEV = Battery Electric Vehicle
    type mismatches.
    """
    spec_terms = [t.lower() for t in (field_specific_terms or [])]
    excl_terms = [t.lower() for t in (exclusion_terms or [])]

    surveys = _openalex_survey_search(field_name)
    if not surveys:
        print("  Survey calibration: no suitable survey found, skipping")
        return []

    # Filter surveys by field relevance
    def _is_field_relevant_survey(s: dict) -> bool:
        title = s.get("title", "").lower()
        abstract = s.get("abstract", "").lower()
        text = title + " " + abstract
        if spec_terms and not any(t in text for t in spec_terms):
            return False
        if excl_terms and any(t in text for t in excl_terms):
            return False
        return True

    relevant_surveys = [s for s in surveys if _is_field_relevant_survey(s)]
    if not relevant_surveys:
        print(f"  Survey calibration: {len(surveys)} found but none field-relevant, skipping")
        return []

    # Pick the best survey: recent, high citation, has OpenAlex ID
    survey = None
    for s in sorted(relevant_surveys, key=lambda s: (-s.get("citation_count", 0), -s.get("year", 0))):
        oa_id = s.get("_oa_id", "")
        if oa_id:
            survey = s
            break

    if not survey:
        print("  Survey calibration: no survey with usable OpenAlex ID, skipping")
        return []

    print(f"  Survey calibration: using \"{survey['title'][:80]}\" ({survey.get('year','?')}, {survey.get('citation_count',0)} cit)")

    # Fetch the survey's references via OpenAlex
    existing_titles = {c["title"].lower().strip() for c in candidates}
    refs = _openalex_get_references(survey["_oa_id"], limit=200)
    new_papers = []
    for d in refs:
        if d["title"].lower().strip() not in existing_titles:
            d["source"] = "survey_reference"
            new_papers.append(d)

    if new_papers:
        print(f"  Survey calibration: +{len(new_papers)} papers from survey references")
    else:
        print(f"  Survey calibration: all references already in pool")
    return new_papers


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1.5: Embedding-based Seminal Matching
# ═══════════════════════════════════════════════════════════════════════════════

_MILESTONE_SYSTEM = """\
You are a research historian who knows the landmark papers in academic fields. \
Given a field name, you describe the key milestone papers that define its evolution.

For each milestone, provide a natural-language DESCRIPTION of what the paper contributed \
(1-2 sentences) and any distinctive title keywords you recall.

IMPORTANT: Do NOT list incremental variants or derivative works. For each milestone, \
focus on the ORIGINAL paper that first introduced the idea. If you know multiple \
variants of a method (e.g. FQ-PETR, Graph-DETR3D), only list the canonical original.

Return ONLY a JSON object. No other text."""

_MILESTONE_PROMPT = """\
List the milestone (foundational) papers for: **{field_name}**

For each, give:
- "description": 1-2 sentence summary of the paper's core contribution. Use natural \
  language, including the key concepts, not just acronyms.
- "known_title_keywords": distinctive words from the actual title (3-6 words), if known
- "first_author": lead author's last name (if known)
- "year": approximate publication year (if known)

Cover the full evolution: earliest foundational work, mid-field breakthroughs, \
and paradigm-shifting recent work. Include specific model/algorithm names.

Return JSON:
```json
{{
  "milestones": [
    {{
      "description": "contribution summary in natural language",
      "known_title_keywords": "distinctive title words",
      "first_author": "LastName",
      "year": 2020
    }}
  ]
}}
```"""


def _generate_milestone_descriptions(
    field_name: str,
    client: OpenAI,
    model: Optional[str] = None,
) -> list[dict]:
    """Generate milestone descriptions for embedding matching.

    Called AFTER the candidate pool is assembled — these descriptions are used
    for matching within the pool, NOT for driving retrieval.
    """
    model = _resolve_model(model)
    if not model or not client:
        return []

    prompt = _MILESTONE_PROMPT.format(field_name=field_name)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _MILESTONE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = resp.choices[0].message.content or ""
        data = _extract_json_object(raw)
        if not data:
            print("  Milestone generation: failed to parse JSON")
            return []

        milestones = data.get("milestones", [])
        print(f"  Milestone descriptions: {len(milestones)} generated")
        for m in milestones:
            print(f"    - {m.get('description', '?')[:80]} ({m.get('first_author', '?')}, {m.get('year', '?')})")
        return milestones
    except Exception as e:
        print(f"  Milestone generation failed: {e}")
        return []


_MG_EMBED_THRESHOLD = 0.5  # Loose filter: drop clearly irrelevant papers


def _milestone_guided_search(
    milestones: list[dict],
    pool_titles: set[str],
    emb_client=None,
) -> list[dict]:
    """V2.3: Search with full description + embedding filter results.

    Uses the full milestone description (1-2 sentences) as search query,
    not just known_title_keywords. Search results are embedded and compared
    to the milestone description — only semantically relevant results pass.

    This solves two problems at once:
    1. Vocabulary gap: "Lift-Splat-Shoot" vs "Lift, Splat, Shoot" — embedding
       similarity is robust to hyphens/commas/spelling variants
    2. Noise: ~40% of keyword search results were irrelevant (biology, math)
       — embedding similarity to a BEV milestone description is naturally
       low for non-BEV papers
    """
    if not milestones:
        return []

    discovered: dict[str, dict] = {}

    # Pre-compute milestone embeddings for filtering
    milestone_texts = [_text_for_embedding(m) for m in milestones]
    milestone_embs = None
    if emb_client:
        try:
            milestone_embs = _compute_embeddings(milestone_texts, emb_client)
        except Exception:
            pass

    for ms_idx, m in enumerate(milestones):
        desc = m.get("description", "").strip()
        keywords = m.get("known_title_keywords", "").strip()
        # Use description as primary search query, keywords as fallback
        search_queries = []
        if desc:
            search_queries.append(desc[:200])  # First 200 chars of description
        if keywords and keywords not in desc:
            search_queries.append(keywords)

        all_results: list[dict] = []
        for q in search_queries:
            # OpenAlex search
            oa_results = _openalex_search(q, limit=10)
            all_results.extend(oa_results)
            time.sleep(1.0)
            # arXiv search
            arxiv_results = search_arxiv(q, max_results=5, search_field="all")
            all_results.extend(arxiv_results)
            time.sleep(1.0)

        if not all_results:
            continue

        # V2.3: Embedding filter — only keep semantically relevant results
        filtered = all_results
        if milestone_embs is not None and ms_idx < len(milestone_embs):
            ms_emb = milestone_embs[ms_idx]
            result_texts = [_text_for_embedding(r) for r in all_results]
            result_embs = _compute_embeddings(result_texts, emb_client)
            filtered = []
            for r, r_emb in zip(all_results, result_embs):
                sim = _cosine_similarity(ms_emb, r_emb)
                if sim >= _MG_EMBED_THRESHOLD:
                    r["_mg_sim"] = float(sim)
                    filtered.append(r)
                # else: silently dropped — embedding says it's not about this milestone

        # Dedup and add to discovered
        for r in filtered:
            t = r["title"].lower().strip()
            if t not in pool_titles and t not in discovered:
                r["source"] = "milestone_guided"
                discovered[t] = r

    new_papers = list(discovered.values())
    if new_papers:
        n_filtered = sum(1 for p in new_papers if "_mg_sim" in p)
        print(f"  Milestone-guided search: +{len(new_papers)} papers from {len(milestones)} milestones"
              f" ({n_filtered} passed embedding filter)")
        for p in new_papers[:10]:
            sim_str = f" [sim={p.get('_mg_sim', 0):.3f}]" if "_mg_sim" in p else ""
            print(f"    Found: {p['title'][:90]}{sim_str}")
        if len(new_papers) > 10:
            print(f"    ... and {len(new_papers) - 10} more")
    return new_papers


def _text_for_embedding(item: dict) -> str:
    """Build the text used for embedding: title + abstract."""
    title = item.get("title", "") or item.get("description", "")
    abstract = item.get("abstract", "") or ""
    return f"{title}\n{abstract}".strip()


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _compute_embeddings(texts: list[str], emb_client: Optional[OpenAI]) -> list[np.ndarray]:
    """Batch-compute embeddings for a list of texts.

    Falls back to local HashingVectorizer if no API client.
    """
    embeddings = []
    for text in texts:
        if emb_client:
            emb = get_embedding(text, emb_client)
        else:
            emb = _local_embedding(text)
        embeddings.append(emb)
    return embeddings


def _embedding_match(
    milestones: list[dict],
    candidates: list[dict],
    emb_client: Optional[OpenAI] = None,
) -> tuple[dict[int, list[str]], list[tuple[int, dict, float]], list[int]]:
    """Step 1.5: Match milestone descriptions to candidate papers via embedding similarity.

    Returns:
        (matched_map, ambiguous_list, unmatched_list)
        matched_map: {milestone_idx: [candidate_title, ...]} for high-confidence matches
        ambiguous_list: [(milestone_idx, candidate, similarity)] for borderline cases
        unmatched_list: [milestone_idx, ...] for no match above LOW threshold
    """
    if not milestones or not candidates:
        return {}, [], list(range(len(milestones)))

    print(f"\n  {'─'*60}")
    print(f"  Embedding-based Seminal Matching")
    print(f"  {'─'*60}")

    # Build embedding texts
    milestone_texts = [_text_for_embedding(m) for m in milestones]
    candidate_texts = [_text_for_embedding(c) for c in candidates]

    # Compute embeddings
    milestone_embs = _compute_embeddings(milestone_texts, emb_client)
    candidate_embs = _compute_embeddings(candidate_texts, emb_client)

    matched: dict[int, list[str]] = defaultdict(list)
    ambiguous: list = []  # (ms_idx, candidate, similarity)
    unmatched: list[int] = []

    for ms_idx, ms_emb in enumerate(milestone_embs):
        ms_info = milestones[ms_idx]
        best_sim = -1.0
        best_cand_idx = -1

        # Compute similarity to all candidates, find best match
        for cand_idx, cand_emb in enumerate(candidate_embs):
            sim = _cosine_similarity(ms_emb, cand_emb)
            if sim > best_sim:
                best_sim = sim
                best_cand_idx = cand_idx

        desc_snippet = ms_info.get("description", str(ms_idx))[:80]

        if best_sim >= _EMBED_HIGH_THRESHOLD:
            # High confidence: direct match
            matched[ms_idx] = [candidates[best_cand_idx]["title"]]
            print(f"  ✓ [{best_sim:.3f}] \"{desc_snippet}\"")
            print(f"     → {candidates[best_cand_idx]['title'][:80]}")
        elif best_sim >= _EMBED_LOW_THRESHOLD:
            # Borderline: send to LLM for final judgment
            ambiguous.append((ms_idx, candidates[best_cand_idx], best_sim))
            print(f"  ? [{best_sim:.3f}] \"{desc_snippet}\" — ambiguous, to LLM")
            print(f"     → {candidates[best_cand_idx]['title'][:80]}")
        else:
            # No match: below threshold
            unmatched.append(ms_idx)
            if best_cand_idx >= 0:
                print(f"  ✗ [{best_sim:.3f}] \"{desc_snippet}\" — unmatched")
                print(f"     best: {candidates[best_cand_idx]['title'][:80]}")
            else:
                print(f"  ✗ [no candidates] \"{desc_snippet}\"")

    n_high = len(matched)
    n_amb = len(ambiguous)
    n_miss = len(unmatched)
    print(f"  Result: {n_high} matched, {n_amb} ambiguous, {n_miss} unmatched")
    print()

    return dict(matched), ambiguous, unmatched


def _embedding_select(
    milestones: list[dict],
    candidates: list[dict],
    emb_client: Optional[OpenAI] = None,
    max_papers: int = 20,
) -> list[dict]:
    """V2.2: Select papers by max embedding similarity to any milestone.

    Deterministic fallback when LLM selection fails. Ranks candidates by
    their max cosine similarity to any milestone description, selects top-N.

    This is always field-relevant because milestone descriptions encode
    the field's core conceptual contributions (not generic ML concepts).
    """
    if not milestones or not candidates:
        return candidates[:max_papers]

    milestone_texts = [_text_for_embedding(m) for m in milestones]
    candidate_texts = [_text_for_embedding(c) for c in candidates]

    milestone_embs = _compute_embeddings(milestone_texts, emb_client)
    candidate_embs = _compute_embeddings(candidate_texts, emb_client)

    for i, c_emb in enumerate(candidate_embs):
        max_sim = max(_cosine_similarity(c_emb, m_emb) for m_emb in milestone_embs)
        candidates[i]["_milestone_sim"] = float(max_sim)

    ranked = sorted(candidates, key=lambda c: -c.get("_milestone_sim", 0))
    selected = ranked[:max_papers]
    for c in selected:
        c["rationale"] = f"embedding selection (max milestone sim={c.get('_milestone_sim', 0):.3f})"
        c["classification"] = "embedding_ranked"

    print(f"  Embedding selection: {len(selected)} papers by milestone similarity"
          f" (range: {selected[-1].get('_milestone_sim', 0):.3f}"
          f" – {selected[0].get('_milestone_sim', 0):.3f})")
    return selected


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1.6: LLM Unified Selection (1 LLM call)
# ═══════════════════════════════════════════════════════════════════════════════

_UNIFIED_SELECTION_SYSTEM = """\
You are a research historian selecting the most representative papers for a field and \
producing a complete map of the field's intellectual landscape.

You receive a candidate pool of papers. Your task has THREE parts:
1. CLASSIFY each paper as: MILESTONE (foundational, paradigm-setting), INCREMENTAL \
   (valid contribution building on milestones), DERIVATIVE (minor variant of a known method), \
   or IRRELEVANT (not about this field).
2. SELECT a representative subset (up to max_papers) spanning the field's full evolution.
3. LIST any IMPORTANT milestones you KNOW exist but are MISSING from this pool.

SELECTION PRINCIPLES:
- Temporal coverage: every major period must be represented
- Sub-topic diversity: different technical approaches each get their representative
- Milestones prioritized: if a paper is a clear landmark, include it even with low citations
- New influential work not overlooked: recent papers may have few citations but high impact

Return ONLY a JSON object. No other text."""

_UNIFIED_SELECTION_PROMPT = """\
Select up to {max_papers} most representative papers for: **{field_name}**

AMBIGUOUS MILESTONE MATCHES — these are possible matches for known milestone papers, \
but the confidence is borderline. For each, confirm (YES) or reject (NO) whether the \
candidate is the actual milestone paper described:
{ambiguous_text}

CANDIDATE POOL ({n_total} papers):
{candidates_text}

Pick papers that TOGETHER tell the complete story of this field — from origins to \
latest breakthroughs, covering all major technical directions.

After selecting, list any MILESTONE papers you KNOW should be here but are MISSING \
from the pool. Provide as much detail as possible (description, keywords, author, year).

Return JSON:
```json
{{
  "milestone_confirmations": {{
    "<amb index>": "YES or NO"
  }},
  "selected_papers": [
    {{
      "title": "exact title from list",
      "rationale": "one sentence why selected",
      "classification": "milestone|incremental|derivative"
    }}
  ],
  "rejected_papers": [
    {{
      "title": "exact title from list",
      "reason": "derivative_work|irrelevant|out_of_scope"
    }}
  ],
  "missing_papers": [
    {{
      "description": "what this paper contributed",
      "known_title_keywords": "distinctive phrase",
      "first_author": "LastName",
      "year": 2020
    }}
  ]
}}
```"""


def _llm_unified_select(
    candidates: list[dict],
    field_name: str,
    ambiguous: list,
    client: OpenAI,
    *,
    max_papers: int = 20,
    model: Optional[str] = None,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Step 1.6: Single LLM call for classification + selection + missing paper listing.

    Args:
        candidates: full candidate pool
        field_name: field name
        ambiguous: list of (ms_idx, candidate_dict, similarity) for borderline matches
        client: LLM client
        max_papers: max papers to select
        model: optional model override

    Returns:
        (selected_papers, rejected_papers, missing_papers)
    """
    if not client:
        print("  No LLM client for selection, using citation stratification")
        result = _stratify_by_citation(candidates, max_papers)
        for c in result:
            c["rationale"] = "citation stratification (no LLM)"
            c["classification"] = "unknown"
        return result, [], []

    model = _resolve_model(model)
    if not model:
        result = _stratify_by_citation(candidates, max_papers)
        for c in result:
            c["rationale"] = "citation stratification (no model)"
            c["classification"] = "unknown"
        return result, [], []

    # Sort by citation_count desc (best first in prompt)
    candidates_sorted = sorted(candidates, key=lambda c: (-c.get("citation_count", 0), -c.get("year", 0)))

    # Build ambiguous matches text
    ambiguous_lines = []
    for i, (ms_idx, cand, sim) in enumerate(ambiguous):
        ambiguous_lines.append(
            f"[AMB-{i}] Milestone description: (similarity={sim:.3f})\n"
            f"  Candidate: {cand['title']} ({cand.get('year','?')})\n"
            f"  Abstract: {cand.get('abstract','')[:200]}"
        )
    ambiguous_text = "\n".join(ambiguous_lines) if ambiguous_lines else "(none)"

    # Build candidate text
    lines = []
    for i, c in enumerate(candidates_sorted):
        cc_str = f" — {c['citation_count']} cit" if c.get("citation_count") else ""
        abstract_snip = (c.get("abstract") or "")[:200]
        lines.append(
            f"[{i + 1}] {c['title']} ({c.get('year','?')}){cc_str}\n"
            f"    {abstract_snip}"
        )
    candidates_text = "\n\n".join(lines)

    prompt = _UNIFIED_SELECTION_PROMPT.format(
        field_name=field_name,
        max_papers=max_papers,
        n_total=len(candidates_sorted),
        ambiguous_text=ambiguous_text,
        candidates_text=candidates_text,
    )
    system = _UNIFIED_SELECTION_SYSTEM

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = resp.choices[0].message.content or ""
        data = _extract_json_object(raw)
        if not data:
            print("  LLM selection: failed to parse JSON, using citation stratification")
            result = _stratify_by_citation(candidates, max_papers)
            for c in result:
                c.setdefault("rationale", "citation stratification (parse error)")
                c.setdefault("classification", "unknown")
            return result, [], []

        # ── Map selected titles back to candidates ──
        selected_titles = [item.get("title", "") for item in data.get("selected_papers", [])]
        selection_meta = {}
        for item in data.get("selected_papers", []):
            selection_meta[item.get("title", "")] = {
                "rationale": item.get("rationale", ""),
                "classification": item.get("classification", "unknown"),
            }

        selected = []
        for sel_title in selected_titles:
            matched = _match_title(sel_title, candidates_sorted)
            if matched:
                meta = selection_meta.get(sel_title, {})
                matched["rationale"] = meta.get("rationale", "")
                matched["classification"] = meta.get("classification", "unknown")
                selected.append(matched)

        # ── Rejected papers ──
        rejected = []
        reject_titles = [item.get("title", "") for item in data.get("rejected_papers", [])]
        reject_reasons = {}
        for item in data.get("rejected_papers", []):
            reject_reasons[item.get("title", "")] = item.get("reason", "")

        for rt in reject_titles:
            matched = _match_title(rt, candidates_sorted)
            if matched:
                matched["reject_reason"] = reject_reasons.get(rt, "")
                rejected.append(matched)

        # ── Missing papers ──
        missing = data.get("missing_papers", [])

        if len(selected) < 5:
            print(f"  LLM selection returned only {len(selected)} papers, using citation stratification")
            result = _stratify_by_citation(candidates, max_papers)
            for c in result:
                c.setdefault("rationale", "citation stratification (<5 fallback)")
                c.setdefault("classification", "unknown")
            return result, rejected, missing

        n_milestone = sum(1 for c in selected if c.get("classification") == "milestone")
        print(f"  LLM selected {len(selected)} papers ({n_milestone} milestone,"
              f" {len(rejected)} rejected, {len(missing)} missing flagged)")
        return selected, rejected, missing

    except Exception as e:
        print(f"  LLM selection failed: {e}, using citation stratification")
        result = _stratify_by_citation(candidates, max_papers)
        for c in result:
            c.setdefault("rationale", "citation stratification (error fallback)")
            c.setdefault("classification", "unknown")
        return result, [], []


def _stratify_by_citation(papers: list[dict], max_total: int) -> list[dict]:
    """Fallback: select top papers by citation, stratified by year."""
    by_year: dict[int, list[dict]] = defaultdict(list)
    for p in papers:
        by_year[p.get("year", 0)].append(p)

    if not by_year:
        return papers[:max_total]

    years = sorted(by_year.keys())
    n_years = len(years)
    min_per_year = min(3, max_total // n_years) if n_years > 0 else 5
    allocated = {y: min(min_per_year, len(by_year[y])) for y in years}
    used = sum(allocated.values())
    remaining = max_total - used

    if remaining > 0:
        total_papers = sum(len(by_year[y]) for y in years)
        for y in years:
            if remaining <= 0:
                break
            extra = max(1, int(remaining * len(by_year[y]) / total_papers))
            extra = min(extra, len(by_year[y]) - allocated[y])
            allocated[y] += extra
            remaining -= extra

    result = []
    for y in years:
        year_papers = sorted(
            by_year[y],
            key=lambda p: (p.get("citation_count", 0), bool(p.get("arxiv_id")), p.get("month", 0)),
            reverse=True,
        )
        result.extend(year_papers[:allocated[y]])

    result.sort(key=lambda p: (-p["year"], -p.get("month", 0)))
    return result[:max_total]


# ═══════════════════════════════════════════════════════════════════════════════
# Step 1.7: Closed-loop Recovery
# ═══════════════════════════════════════════════════════════════════════════════

def _closed_loop_recover(
    missing_papers: list[dict],
    unmatched_milestones: list[dict],
    selected: list[dict],
    field_name: str,
) -> tuple[list[dict], list[dict]]:
    """Step 1.7: Recover missing papers via SS + arXiv + author search.

    Unlike V1's recovery which reused the same arXiv keyword search,
    this uses SS semantic search as primary, plus author and title paths.

    Returns:
        (recovered_papers, confirmed_missing) — confirmed_missing are the papers
        that could NOT be found after all recovery attempts.
    """
    all_missing = list(missing_papers)

    # Add unmatched milestones not already in missing list
    existing_descs = {m.get("description", "")[:40] for m in missing_papers}
    for um in unmatched_milestones:
        desc = um.get("description", "")[:40]
        if desc not in existing_descs:
            all_missing.append({
                "description": um.get("description", ""),
                "known_title_keywords": um.get("known_title_keywords", ""),
                "first_author": um.get("first_author", ""),
                "year": um.get("year", 0),
            })

    if not all_missing:
        return [], []

    print(f"\n  [Closed-loop Recovery] {len(all_missing)} papers to find")
    existing_titles = {s["title"].lower().strip() for s in selected}
    recovered = []
    confirmed_missing = []

    for mp in all_missing:
        desc = mp.get("description", "")
        keywords = mp.get("known_title_keywords", "")
        author = mp.get("first_author", "")
        year = mp.get("year", 0)

        search_text = desc or keywords
        if not search_text:
            continue

        paths_tried = []
        best = None

        # ── Path 1: OpenAlex semantic search with description ──
        paths_tried.append("openalex_semantic")
        oa_results = _openalex_search(search_text[:200], limit=10)
        if oa_results:
            best = oa_results[0]

        time.sleep(1.0)

        # ── Path 2: arXiv author search (independent of keyword quality) ──
        if not best and author:
            paths_tried.append("arxiv_author")
            # Use search_field="au" so the query is correctly scoped to author names.
            # search_arxiv prepends {search_field}:{query}, so query must be just
            # the author name, not "au:Name".
            author_results = search_arxiv(author, max_results=10, search_field="au")
            if author_results:
                # Filter by year proximity when we have a target year
                if year:
                    author_results.sort(key=lambda p: abs(p.get("year", 0) - year))
                best = author_results[0]
                best["source"] = "recovery_author"

        time.sleep(3.0)

        # ── Path 3: OpenAlex title search with keywords ──
        if not best and keywords:
            paths_tried.append("openalex_title")
            title_results = _openalex_search(keywords, limit=5)
            if title_results:
                best = title_results[0]
                best["source"] = "recovery_title"

        if best:
            bt = best["title"].lower().strip()
            if bt not in existing_titles:
                best["rationale"] = f"recovered: {search_text[:80]}"
                best["recovery_paths_tried"] = paths_tried
                recovered.append(best)
                existing_titles.add(bt)
                print(f"  ✓ Recovered [{paths_tried[-1]}]: {best['title'][:80]}")
        else:
            confirmed_missing.append({
                "description": desc,
                "known_title_keywords": keywords,
                "first_author": author,
                "year": year,
                "recovery_attempts": paths_tried,
                "all_failed": True,
            })
            snippet = desc[:60] if desc else keywords[:60]
            print(f"  ✗ Not found: \"{snippet}\" (tried: {', '.join(paths_tried)})")

    if recovered:
        print(f"  Recovery: +{len(recovered)} papers found")
    if confirmed_missing:
        print(f"  CONFIRMED MISSING: {len(confirmed_missing)} papers could not be found"
              f" — see confirmed_missing in output")

    return recovered, confirmed_missing


# ═══════════════════════════════════════════════════════════════════════════════
# Top-level API
# ═══════════════════════════════════════════════════════════════════════════════

def retrieve_field_papers(
    field_name: str,
    client: Optional[OpenAI] = None,
    *,
    max_papers: int = 20,
    fast: bool = False,
    no_survey: bool = False,
    model: Optional[str] = None,
) -> tuple[list[dict], dict]:
    """Retrieve representative papers for a research field (V2: semantic-first).

    Args:
        field_name: e.g. "BEV Perception".
        client: LLM client for query expansion + milestone description + selection.
        max_papers: maximum papers to return.
        fast: if True, skip all LLM calls, single SS search → top by citations.
        no_survey: if True, skip survey calibration (saves 0-1 LLM call).
        model: optional model override.

    Returns:
        (papers, report) — papers: list[dict], report: dict with confirmed_missing
    """
    print(f"\n{'='*60}")
    print(f"  Paper Retrieval V2: {field_name}")
    print(f"{'='*60}")

    # ── Fast mode: single search, no LLM ──
    if fast or not client:
        if fast:
            print("  --fast mode: single OpenAlex search, no LLM")
        print(f"  Searching OpenAlex for: {field_name}")
        candidates = _openalex_search(field_name, limit=100)
        candidates = _deduplicate(candidates)
        candidates = _openalex_enrich(candidates, field_name)
        if not candidates:
            print("  No papers found. Try different keywords.")
            return [], {"confirmed_missing": [], "total_recalled": 0}
        candidates.sort(key=lambda c: (-c.get("citation_count", 0), -c.get("year", 0)))
        print(f"  Found {len(candidates)} candidates"
              f" (top: {candidates[0]['title'][:60]}..., {candidates[0].get('citation_count', 0)} cit)")
        result = candidates[:max_papers]
        for c in result:
            c["rationale"] = "top by relevance/citations (fast mode)"
        return result, {"confirmed_missing": [], "total_recalled": len(candidates)}

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1.1: Query Expansion
    # ═══════════════════════════════════════════════════════════════════════
    query_expansion = _generate_query_expansion(field_name, client, model=model)

    # Extract disambiguation terms for field filtering
    disambiguation = query_expansion.get("disambiguation", {})
    core_field_markers = disambiguation.get("core_field_markers", [])
    field_specific_terms = disambiguation.get("field_specific_terms", [field_name])
    exclusion_terms = disambiguation.get("exclusion_terms", [])
    # If no core_field_markers, fall back to field_specific_terms (backward compat)
    if not core_field_markers:
        core_field_markers = list(field_specific_terms)
    # Also add synonyms as field-specific terms (for scoring, not filtering)
    for variants in query_expansion.get("synonyms_and_variants", {}).values():
        field_specific_terms.extend(variants)
    print(f"  Disambiguation: {len(core_field_markers)} core markers,"
          f" {len(field_specific_terms)} scoring terms, {len(exclusion_terms)} exclusion terms")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1.2: Multi-source Broad Recall (SS primary + arXiv auxiliary)
    # ═══════════════════════════════════════════════════════════════════════
    candidates = _broad_recall(query_expansion, field_name)

    if not candidates:
        print("  No papers found. Trying fallback with field name directly...")
        candidates = _openalex_search(field_name, limit=200) + search_arxiv(field_name, max_results=100)
        candidates = _deduplicate(candidates)

    if not candidates:
        print("  No papers found.")
        print("  Try different keywords or use --papers to provide a manual list.")
        return [], {"confirmed_missing": [], "total_recalled": 0}

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1.3: Citation Expansion
    # ═══════════════════════════════════════════════════════════════════════
    # First, enrich with citation counts so we can pick top seeds
    candidates = _openalex_enrich(candidates, field_name)

    snowballed = _citation_expand(
        candidates, field_name,
        core_field_markers=core_field_markers,
        exclusion_terms=exclusion_terms,
    )
    if snowballed:
        candidates = candidates + snowballed
        candidates = _deduplicate(candidates)
        print(f"  After citation expansion: {len(candidates)} candidates")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1.4: Survey Calibration — DISABLED in V2.2
    # ═══════════════════════════════════════════════════════════════════════
    # V2.1 testing showed OpenAlex survey search returns completely irrelevant
    # papers (e.g., botany papers for "BEV" queries). This step injects noise.
    # Disabled by default until a reliable survey discovery mechanism exists.
    if not no_survey and False:  # force-disabled in V2.2
        survey_papers = _survey_calibrate(
            candidates, field_name, client, model=model,
            field_specific_terms=field_specific_terms,
            exclusion_terms=exclusion_terms,
        )
        if survey_papers:
            candidates = candidates + survey_papers
            candidates = _deduplicate(candidates)
            print(f"  After survey calibration: {len(candidates)} candidates")

    print(f"  Candidate pool: {len(candidates)} papers (no pre-filter)")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1.2.5: Relevance-based Pre-rank
    # ═══════════════════════════════════════════════════════════════════════
    candidates = _pre_rank(candidates, core_field_markers, exclusion_terms)

    # LLM selection only sees top-N by relevance score
    llm_candidates = candidates[:_PRE_RANK_TOP_N]
    print(f"  Sending Top-{len(llm_candidates)} to LLM selection (from {len(candidates)} total)")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1.5: Generate milestone descriptions + milestone-guided search + embedding matching
    # ═══════════════════════════════════════════════════════════════════════
    milestones = _generate_milestone_descriptions(field_name, client, model=model)

    # Build embedding client
    emb_client = build_embedding_client()

    # ── V2.3: Milestone-guided search with embedding filtering ──
    # Search with full description (not just keywords) + embed filter noise
    pool_titles = {c["title"].lower().strip() for c in candidates}
    mg_papers = _milestone_guided_search(milestones, pool_titles, emb_client=emb_client)
    if mg_papers:
        candidates = candidates + mg_papers
        candidates = _deduplicate(candidates)
        # Re-rank after adding milestone-guided papers
        candidates = _pre_rank(candidates, core_field_markers, exclusion_terms)
        llm_candidates = candidates[:_PRE_RANK_TOP_N]
        print(f"  After milestone-guided search: {len(candidates)} candidates,"
              f" top-{len(llm_candidates)} for LLM")

    # ── V2.3.1: Promote milestone-discovered papers to top-150 ──
    # Papers found by milestone-guided search already passed embedding validation
    # (sim > _MG_EMBED_THRESHOLD). They should not be excluded by keyword-based
    # pre-rank — that would contradict the embedding-based admission decision.
    if mg_papers:
        llm_titles = {c["title"].lower().strip() for c in llm_candidates}
        for p in mg_papers:
            t = p["title"].lower().strip()
            if t not in llm_titles:
                p["_mg_promoted"] = True
                llm_candidates.insert(0, p)
        if any(p.get("_mg_promoted") for p in llm_candidates):
            n_promoted = sum(1 for p in llm_candidates if p.get("_mg_promoted"))
            llm_candidates = llm_candidates[:_PRE_RANK_TOP_N]
            promoted_titles = [p["title"][:80] for p in llm_candidates if p.get("_mg_promoted")]
            print(f"  V2.3.1: Promoted {n_promoted} milestone-discovered papers to top-150:"
                  f" {promoted_titles}")

    matched_map, ambiguous, unmatched_indices = _embedding_match(
        milestones, llm_candidates, emb_client,
    )

    # Mark high-confidence matches in candidates
    matched_candidate_titles = set()
    for titles in matched_map.values():
        for t in titles:
            matched_candidate_titles.add(t.lower().strip())

    for c in candidates:
        if c["title"].lower().strip() in matched_candidate_titles:
            c["is_seminal"] = True

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1.6: LLM Unified Selection
    # ═══════════════════════════════════════════════════════════════════════
    selected, rejected, missing_from_llm = _llm_unified_select(
        llm_candidates, field_name, ambiguous, client,
        max_papers=max_papers, model=model,
    )

    # ── V2.2: Embedding-based fallback if LLM selection failed ──
    # Citation stratification (V2.1 fallback) selects general ML papers.
    # Embedding-based selection uses milestone descriptions as anchors —
    # it always picks field-relevant papers because milestones encode
    # the field's core conceptual vocabulary.
    if all(c.get("classification", "") == "unknown" for c in selected):
        print("  LLM selection appears to have used citation fallback, switching to embedding selection")
        selected = _embedding_select(milestones, llm_candidates, emb_client, max_papers)
        rejected = []
        missing_from_llm = []

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1.7: Closed-loop Recovery
    # ═══════════════════════════════════════════════════════════════════════
    unmatched_milestones = [milestones[i] for i in unmatched_indices]
    recovered, confirmed_missing = _closed_loop_recover(
        missing_from_llm, unmatched_milestones, selected, field_name,
    )

    # Add recovered papers to selection if there's room
    if recovered:
        for r in recovered:
            if len(selected) < max_papers:
                selected.append(r)
            else:
                # Swap with a non-milestone paper if possible
                for i, s in enumerate(selected):
                    if s.get("classification") != "milestone":
                        selected[i] = r
                        break

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1.8: Build final output
    # ═══════════════════════════════════════════════════════════════════════

    # Clean internal flags from output papers
    _INTERNAL_KEYS = {"_mg_promoted", "_mg_sim", "_milestone_sim", "_oa_id"}
    for p in selected:
        for k in _INTERNAL_KEYS:
            p.pop(k, None)

    report = {
        "confirmed_missing": confirmed_missing,
        "total_recalled": len(candidates),
        "total_selected": len(selected),
        "n_milestone_matched": len(matched_map),
        "n_milestone_ambiguous": len(ambiguous),
        "n_milestone_unmatched": len(unmatched_indices),
        "n_recovered": len(recovered),
    }

    # Print summary
    print(f"\n  {'='*60}")
    print(f"  Retrieval Complete")
    print(f"  {'='*60}")
    print(f"  Selected:      {len(selected)} papers")
    print(f"  Recovered:     {len(recovered)} papers")
    print(f"  Confirmed missing: {len(confirmed_missing)} papers")
    if confirmed_missing:
        print(f"  ── MISSING (manual check recommended) ──")
        for cm in confirmed_missing:
            info = cm.get("description", "") or cm.get("known_title_keywords", "")
            print(f"    • {info[:100]} ({cm.get('first_author','?')}, {cm.get('year','?')})")

    return selected, report
