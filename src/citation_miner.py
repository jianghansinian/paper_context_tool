"""Citation miner for V3 pipeline.

Uses Semantic Scholar API for backward/forward citation mining with recursive
expansion to discover the paper landscape around a seed paper.
"""
from __future__ import annotations

import json
import textwrap
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

import requests

import config
from paper import Paper, Reference, CitationType

_SS_API_BASE = "https://api.semanticscholar.org/graph/v1"
_REF_FIELDS = (
    "contexts,intents,isInfluential,"
    "citedPaper.paperId,citedPaper.title,citedPaper.authors,"
    "citedPaper.year,citedPaper.citationCount,citedPaper.externalIds,"
    "citedPaper.abstract,citedPaper.url"
)
_CITE_FIELDS = (
    "citingPaper.paperId,citingPaper.title,citingPaper.authors,"
    "citingPaper.year,citingPaper.citationCount,citingPaper.externalIds,"
    "citingPaper.abstract,citingPaper.url"
)
_PAPER_FIELDS = (
    "title,authors,year,abstract,citationCount,externalIds,url,referenceCount"
)
_MAX_WORKERS = 5


def _ss_headers() -> dict:
    h = {}
    if config.SS_API_KEY:
        h["x-api-key"] = config.SS_API_KEY
    return h


def _paper_from_ss(ss_data: dict, prefix: str = "citedPaper") -> Paper:
    """Create a Paper from Semantic Scholar API response data."""
    p = ss_data.get(prefix, ss_data)
    external = p.get("externalIds", {}) or {}
    arxiv_id = external.get("ArXiv", "")
    authors = [a.get("name", "") for a in (p.get("authors") or [])]
    paper_id = p.get("paperId", "")
    return Paper(
        id=paper_id,
        arxiv_id=arxiv_id,
        title=p.get("title") or "",
        authors=authors,
        year=p.get("year") or 0,
        abstract=p.get("abstract") or "",
        citation_count=p.get("citationCount") or 0,
        url=p.get("url") or f"https://arxiv.org/abs/{arxiv_id}" if arxiv_id else "",
        source="semantic_scholar",
    )


class CitationMiner:
    """Mines citations via Semantic Scholar API to discover the paper landscape."""

    def __init__(self):
        self._papers: dict[str, Paper] = {}
        self._ref_map: dict[str, list[Reference]] = {}  # paper_id -> references

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_paper(self, paper: Paper) -> None:
        """Register a paper in the miner's pool."""
        key = paper.id or paper.arxiv_id or paper.title.lower()
        if key:
            self._papers[key] = paper

    def mine_references(self, paper: Paper, max_depth: int = 2,
                        llm_client=None) -> dict[str, Paper]:
        """Backward citation mining: recursively find papers this paper cites.

        Returns dict of all discovered papers (keyed by Semantic Scholar ID).
        """
        self.register_paper(paper)
        to_expand = [paper]
        depth = 0

        while to_expand and depth < max_depth:
            if depth == 0:
                top_k = config.REFERENCE_TOP_K_LEVEL1
            else:
                top_k = config.REFERENCE_TOP_K_LEVEL2 // max(len(to_expand), 1)

            next_level = []
            for p in to_expand:
                refs = self._fetch_references(p, top_k=top_k if depth == 0 else 5)
                self._ref_map[p.id] = refs
                for ref in refs:
                    ref_paper = self._get_or_fetch_paper(ref.paper_id)
                    if ref_paper and ref_paper.id not in self._ref_map:
                        self.register_paper(ref_paper)
                        next_level.append(ref_paper)
                time.sleep(config.SS_REQUEST_DELAY)

            # Deduplicate by ID
            seen = set()
            next_level = [p for p in next_level if p.id not in seen and not seen.add(p.id)]
            to_expand = next_level
            depth += 1

        return self._papers

    def mine_citations(self, paper: Paper) -> dict[str, Paper]:
        """Forward citation mining: find papers that cite this paper."""
        self.register_paper(paper)
        citing_papers = self._fetch_citations(paper)
        for p in citing_papers:
            self.register_paper(p)
        return self._papers

    def classify_references(self, paper: Paper, llm_client=None) -> list[Reference]:
        """Classify references using LLM based on citation context.

        Falls back to Semantic Scholar intents if LLM is unavailable.
        """
        refs = self._ref_map.get(paper.id, [])
        if not refs:
            return []

        if llm_client is None:
            return [self._classify_from_ss_intent(r) for r in refs]

        return self._llm_classify_refs(paper, refs, llm_client)

    def get_key_papers(self, top_k: int = None) -> list[Paper]:
        """Return the top-K most important papers from the discovered pool.

        Sorted by a weighted score of citation count + citation type boost +
        reference frequency.
        """
        if top_k is None:
            top_k = config.KEY_PAPERS_TOTAL

        # Compute reference frequency: how many papers in the pool cite each paper
        ref_freq: dict[str, int] = {}
        for paper_id, refs in self._ref_map.items():
            for ref in refs:
                ref_freq[ref.paper_id] = ref_freq.get(ref.paper_id, 0) + 1

        scored = []
        max_cites = max((p.citation_count for p in self._papers.values()), default=1)
        max_freq = max(ref_freq.values(), default=1)
        current_year = 2026

        for paper in self._papers.values():
            # Skip the seed paper itself (we'll analyze it separately)
            if paper.id and paper.id in self._ref_map:
                # This is a seed or already-analyzed paper
                pass

            cite_score = paper.citation_count / max(max_cites, 1)
            recency_score = min(paper.year / current_year, 1.0) if paper.year else 0
            freq_score = ref_freq.get(paper.id, 0) / max(max_freq, 1)

            # Citation type bonus from references
            type_bonus = 0
            for refs_list in self._ref_map.values():
                for ref in refs_list:
                    if ref.paper_id == paper.id:
                        if ref.citation_type == CitationType.FOUNDATIONAL:
                            type_bonus = max(type_bonus, 1.0)
                        elif ref.citation_type == CitationType.SUPPORTING:
                            type_bonus = max(type_bonus, 0.5)
                        elif ref.citation_type == CitationType.CONTRASTING:
                            type_bonus = max(type_bonus, 0.3)

            score = (
                config.V3_W_CITATION * cite_score
                + config.V3_W_RECENCY * recency_score
                + config.V3_W_CITATION_TYPE * type_bonus
                + config.V3_W_REF_FREQ * freq_score
            )
            scored.append((score, paper))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [paper for _, paper in scored[:top_k]]

    def get_all_papers(self) -> dict[str, Paper]:
        return self._papers

    # ------------------------------------------------------------------
    # Internal: Semantic Scholar API calls
    # ------------------------------------------------------------------

    def _get_or_fetch_paper(self, paper_id: str) -> Optional[Paper]:
        """Get paper from local pool or fetch from Semantic Scholar."""
        if paper_id in self._papers:
            return self._papers[paper_id]

        url = f"{_SS_API_BASE}/paper/{paper_id}"
        try:
            resp = requests.get(url, params={"fields": _PAPER_FIELDS},
                                headers=_ss_headers(),
                                timeout=config.SS_REQUEST_TIMEOUT)
            resp.raise_for_status()
            data = resp.json()
            if data and "paperId" in data:
                paper = _paper_from_ss(data, prefix="")
                self._papers[paper_id] = paper
                return paper
        except Exception:
            pass
        return None

    def _fetch_references(self, paper: Paper, top_k: int = 15) -> list[Reference]:
        """Fetch references from Semantic Scholar for a given paper."""
        paper_id = paper.id
        if not paper_id or paper_id == paper_id:  # ensure it's a valid SS ID
            if paper.arxiv_id:
                paper_id = f"ArXiv:{paper.arxiv_id}"

        url = f"{_SS_API_BASE}/paper/{paper_id}/references"
        params = {"limit": 500, "fields": _REF_FIELDS}
        try:
            resp = requests.get(url, params=params, headers=_ss_headers(),
                                timeout=config.SS_REQUEST_TIMEOUT)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"Failed to fetch references for {paper_id}: {exc}")
            return []

        refs = []
        for item in data.get("data", [])[:500]:
            cited = item.get("citedPaper", {})
            contexts = item.get("contexts", [])
            ref = Reference(
                paper_id=cited.get("paperId", ""),
                paper_title=cited.get("title", ""),
                context=" ".join(contexts) if contexts else "",
                citation_type=CitationType.NOT_CLASSIFIED,
                is_key_reference=bool(item.get("isInfluential")),
            )
            refs.append(ref)

        # Sort by influence then citation count, take top_k
        refs.sort(key=lambda r: (
            r.is_key_reference,
            len(r.context) > 0,
        ), reverse=True)
        return refs[:top_k]

    def _fetch_citations(self, paper: Paper) -> list[Paper]:
        """Fetch citing papers from Semantic Scholar."""
        paper_id = paper.id
        if not paper_id and paper.arxiv_id:
            paper_id = f"ArXiv:{paper.arxiv_id}"

        url = f"{_SS_API_BASE}/paper/{paper_id}/citations"
        params = {"limit": 500, "fields": _CITE_FIELDS}
        try:
            resp = requests.get(url, params=params, headers=_ss_headers(),
                                timeout=config.SS_REQUEST_TIMEOUT)
            if resp.status_code == 404:
                return []
            resp.raise_for_status()
            data = resp.json()
        except Exception as exc:
            print(f"Failed to fetch citations for {paper_id}: {exc}")
            return []

        papers = []
        for item in data.get("data", []):
            paper = _paper_from_ss(item, prefix="citingPaper")
            papers.append(paper)
        return papers

    # ------------------------------------------------------------------
    # Internal: Citation classification
    # ------------------------------------------------------------------

    def _classify_from_ss_intent(self, ref: Reference) -> Reference:
        """Fallback: classify from Semantic Scholar intent data."""
        ctx_lower = ref.context.lower() if ref.context else ""

        if any(w in ctx_lower for w in ("based on", "following", "extends",
                                         "we adopt", "as in", "similar to")):
            ref.citation_type = CitationType.SUPPORTING
        elif any(w in ctx_lower for w in ("however", "in contrast", "limitation",
                                           "unlike", "while", "although")):
            ref.citation_type = CitationType.CONTRASTING
        elif ref.is_key_reference:
            ref.citation_type = CitationType.FOUNDATIONAL
        elif ctx_lower:
            ref.citation_type = CitationType.RELATED

        return ref

    def _llm_classify_refs(self, paper: Paper, refs: list[Reference],
                           llm_client) -> list[Reference]:
        """Use LLM to classify citation types based on context."""
        if not refs:
            return refs

        # Build a batch prompt
        items = []
        for i, ref in enumerate(refs[:30]):  # max 30 per batch
            ctx = ref.context[:200] if ref.context else "(no context available)"
            items.append(f"[{i}] Title: {ref.paper_title}\n    Context in paper: \"{ctx}\"")

        items_text = "\n".join(items)

        user_lens = ""
        if paper.user_description:
            user_lens = (
                f"USER FOCUS: {paper.user_description}\n"
                "Prioritize references related to this focus.\n\n"
            )

        prompt = textwrap.dedent(f"""\
            Classify the relationship between the citing paper and each reference.

            Citing paper: "{paper.title}" ({paper.year})

            {user_lens}
            For each reference, classify as one of:
            - "supporting": The citing paper builds upon or adopts this work's approach
            - "contrasting": The citing paper disagrees with or offers an alternative
            - "foundational": This is foundational/classic work in the field
            - "related": Merely mentioned as related work, no direct comparison

            References:
            {items_text}

            Return a JSON array:
            [{{"index": 0, "type": "supporting", "reason": "brief reason", "is_key": true}}, ...]
        """)

        try:
            resp = llm_client.chat.completions.create(
                model=config.LLM_ANALYZER_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=2000,
            )
            raw = resp.choices[0].message.content or ""
            # Parse JSON
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1]
                if cleaned.endswith("```"):
                    cleaned = cleaned[:-3]
            results = json.loads(cleaned)
        except Exception:
            # Fall back to SS intent classification
            return [self._classify_from_ss_intent(r) for r in refs]

        type_map = {
            "supporting": CitationType.SUPPORTING,
            "contrasting": CitationType.CONTRASTING,
            "foundational": CitationType.FOUNDATIONAL,
            "related": CitationType.RELATED,
        }
        result_map = {r.get("index"): r for r in results}

        for i, ref in enumerate(refs):
            r = result_map.get(i, {})
            ref.citation_type = type_map.get(r.get("type", ""), CitationType.NOT_CLASSIFIED)
            ref.is_key_reference = r.get("is_key", False)

        return refs
