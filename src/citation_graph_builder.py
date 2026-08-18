"""V3.3 Step 3-5: Citation graph construction, ranking, and diversified selection.

V3.3 changes:
  - OpenAlex primary (with API Key), COCI fallback, SS removed
  - Bibliographic coupling (Step 3b): Ochiai coefficient on L1 nodes
  - Config-driven limits and weights
  - Field relevance filter: prevents noise papers from unrelated fields entering top-20
  - V3.3.14: PageRank + coupling_degree removed from ranking;
    citation_density → age-adjusted citation_rate (γ=0.8, β=0.2)
"""

from __future__ import annotations

import concurrent.futures
import datetime
import hashlib
import json
import math
import os
import re
import threading
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import networkx as nx

import config
from paper_retriever import (
    _openalex_request,
    _openalex_paper_to_dict,
    _openalex_get_references,
    _openalex_get_citations,
    _openalex_search,
)
from open_citations_client import _coci_get_references, _coci_get_citations
from ss_client import _ss_get_references, _ss_get_citations, _ss_get_paper, _ss_search, _has_api_key
from openai import OpenAI
from llm_analyzer import _resolve_model, _extract_json_object, _extract_json_array


# ── Node ID helpers ──

def _node_id(paper: dict) -> str:
    """Get the canonical graph node ID for a paper dict.

    Prefers OA short ID > DOI > title hash.
    """
    oa = paper.get("_oa_id", "")
    if oa:
        return _oa_short(oa)
    doi = paper.get("_doi", "")
    if doi:
        return f"doi:{doi}"
    return hashlib.md5(paper.get("title", "").lower().encode()).hexdigest()[:16]


def _oa_short(oa_id: str) -> str:
    """Extract short ID from OpenAlex URL or return as-is if already short."""
    return oa_id.split("/")[-1] if "/" in oa_id else oa_id


# ── Cache helpers ──

def _cache_dir(sub: str) -> Path:
    d = Path(config.V3_CACHE_DIR) / sub
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_key(raw_id: str) -> str:
    return hashlib.md5(raw_id.encode()).hexdigest()[:16]


_CACHE_LOCK = threading.Lock()


def _cache_get(sub: str, raw_id: str) -> list[dict] | None:
    key = _cache_key(raw_id)
    with _CACHE_LOCK:
        path = _cache_dir(sub) / f"{key}.json"
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            age_days = (time.time() - data.get("_cached_at", 0)) / 86400
            if age_days > config.V3_CACHE_TTL_DAYS:
                path.unlink(missing_ok=True)
                return None
            return data.get("papers", [])
        except Exception:
            return None


def _cache_put(sub: str, raw_id: str, papers: list[dict]):
    key = _cache_key(raw_id)
    with _CACHE_LOCK:
        path = _cache_dir(sub) / f"{key}.json"
        try:
            path.write_text(json.dumps({
                "_cached_at": time.time(),
                "papers": papers,
            }, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass


def _normalize_title(title: str) -> str:
    """Normalize title for comparison: lowercase, remove punctuation, collapse whitespace.

    Whitespace collapse catches OA entries that differ only by embedded newlines
    (e.g. "Rigs by Implicitly" vs "Rigs by\\n Implicitly" — same paper, two entries).
    Some OA records embed the newline as a literal two-char "\\n" escape instead
    of a real newline; unescape those first so both variants normalize identically.
    """
    title = title.replace(r"\n", " ").replace(r"\r", " ").replace(r"\t", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", "", title.lower())).strip()


def _dedup_by_title(papers: list[dict]) -> list[dict]:
    """Deduplicate papers by normalized title. Keeps first occurrence (highest score)."""
    seen: set[str] = set()
    deduped: list[dict] = []
    for p in papers:
        t = _normalize_title(p.get("title", ""))
        if t and t not in seen:
            seen.add(t)
            deduped.append(p)
    return deduped


def _is_arxiv_hosted(p: dict) -> bool:
    """True if the OA record is the arXiv preprint version of a paper."""
    src = (p.get("_raw_source_name") or "").lower()
    doi = p.get("_doi") or ""
    return "arxiv" in src or doi.startswith("10.48550")


def _enrich_ss_citations(node_data: dict[str, dict]) -> None:
    """V3.3.11: Override OA citation_count with SS for arXiv-hosted nodes.

    OA's arXiv records undercount citations (measured median 2.5x vs SS, up
    to 19.5x). Query SS by exact normalized-title match, keep max(OA, SS).
    No fuzzy matching — a wrong citation count is worse than a missing one.
    Results cached under `ss_citations` keyed by normalized title.
    """
    by_title: dict[str, list[str]] = defaultdict(list)
    for nid, d in node_data.items():
        if not _is_arxiv_hosted(d):
            continue
        t = _normalize_title(d.get("title", ""))
        if t:
            by_title[t].append(nid)
    if not by_title:
        print("  [SS enrich] no arXiv-hosted nodes to enrich")
        return

    have_key = _has_api_key()
    if not have_key:
        print(f"  [SS enrich] skipped {len(by_title)} arXiv-hosted nodes: no SS_API_KEY")
        return

    print(f"  [SS enrich] {len(by_title)} unique arXiv-hosted titles, ~{len(by_title)}s ...")
    n_queried = 0
    n_updated = 0
    examples: list[str] = []
    for norm_t, nids in by_title.items():
        cached = _cache_get("ss_citations", norm_t)
        if cached is None:
            title = node_data[nids[0]].get("title", "")
            results = _ss_search(title, limit=5)
            match = next(
                (r for r in results if _normalize_title(r.get("title", "")) == norm_t),
                None,
            )
            cached = [{"ss_citation_count": match.get("citation_count", 0)}] if match else []
            if cached:
                _cache_put("ss_citations", norm_t, cached)
            n_queried += 1
        if not cached:
            continue
        ss_cit = cached[0].get("ss_citation_count", 0)
        if not ss_cit:
            continue
        for nid in nids:
            d = node_data[nid]
            old = d.get("citation_count", 0)
            if ss_cit > old:
                d["citation_count"] = ss_cit
                d["_ss_enriched"] = True
                n_updated += 1
                if len(examples) < 5:
                    examples.append(f"{d.get('title', '')[:50]} ({old}->{ss_cit})")
    print(f"  [SS enrich] queried {n_queried}, updated {n_updated} nodes")
    for e in examples:
        print(f"    {e}")


# ── Step 3: Citation graph construction ──

# Normalized title → node_id. V3.3.10: the same paper can enter the graph via
# two records (arXiv preprint vs published version, or OA vs SS fallback) with
# different IDs (e.g. LSS as a 41-cit LNCS seed node and a separate 1720-cit
# SS node). Title is the identity; _add_node merges duplicates into one node.
_TITLE_NODE_MAP: dict[str, str] = {}


def _add_node(graph: nx.DiGraph, node_data: dict[str, dict], paper: dict) -> str:
    """Insert a paper into the graph, merging by normalized title.

    First node wins its ID; later records with the same normalized title merge
    their data into it (citation_count=max, fill empty identity fields).
    Returns the canonical node ID.
    """
    t = _normalize_title(paper.get("title", ""))
    existing = _TITLE_NODE_MAP.get(t) if t else None
    if existing is not None and existing in node_data:
        nd = node_data[existing]
        nd["citation_count"] = max(nd.get("citation_count", 0), paper.get("citation_count", 0))
        for key in ("_oa_id", "_doi", "arxiv_id", "_raw_source_name", "_contribution"):
            if not nd.get(key) and paper.get(key):
                nd[key] = paper[key]
        return existing

    nid = _node_id(paper)
    if nid in node_data:
        nd = node_data[nid]
        nd["citation_count"] = max(nd.get("citation_count", 0), paper.get("citation_count", 0))
    else:
        node_data[nid] = dict(paper)
        graph.add_node(nid)
    if t:
        _TITLE_NODE_MAP[t] = nid
    return nid

def _resolve_dois(dois: list[str], limit: int) -> list[dict]:
    """Resolve a list of DOIs to paper dicts via OpenAlex DOI filter (batch)."""
    batched = dois[:limit]
    if not batched:
        return []
    # OA filter=doi:10.xxx|10.yyy|... supports batch queries
    BATCH_SIZE = 50
    results = []
    for batch_start in range(0, len(batched), BATCH_SIZE):
        batch = batched[batch_start:batch_start + BATCH_SIZE]
        filter_str = "doi:" + "|".join(batch)
        data = _openalex_request("works", {"filter": filter_str, "per_page": str(BATCH_SIZE)})
        if data:
            for work in data.get("results", []):
                d = _openalex_paper_to_dict(work)
                if d:
                    results.append(d)
        time.sleep(0.1)
    return results


def _refs_anomalous(result: list[dict] | None, paper: dict) -> bool:
    """Check if reference results are suspiciously few/empty.

    A real academic paper virtually always has 10+ references.
    0 results or very few results signals an API issue, not a genuine empty list.
    """
    if result is None:
        return True
    if len(result) == 0:
        return True
    # Fewer than 3 refs for any paper is anomalous (even short papers have 5-10+)
    if len(result) < 3:
        return True
    return False


def _cites_anomalous(result: list[dict] | None, paper: dict) -> bool:
    """Check if citation results are suspiciously few/empty.

    0 forward citations for a paper with known citation count > 50 is anomalous.
    """
    if result is None:
        return True
    if len(result) == 0 and paper.get("citation_count", 0) > 50:
        return True
    return False


def _fetch_references(paper: dict, limit: int) -> list[dict]:
    """Fetch references with OA-first, anomaly-checked multi-API fallback.

    OA is primary because one /works/{id} call returns the full referenced_works
    list. With API Key, there's no daily budget concern.

    Fallback chain: OA → COCI (each level validates before accepting).
    """
    oa_id = paper.get("_oa_id", "")
    doi = paper.get("_doi", "")

    # 1. OA primary: single call returns referenced_works list
    if oa_id:
        cached = _cache_get("oa_references", oa_id)
        if cached:
            return cached[:limit]
        try:
            refs = _openalex_get_references(oa_id, limit=limit)
            if not _refs_anomalous(refs, paper):
                _cache_put("oa_references", oa_id, refs)
                return refs
        except Exception:
            pass

    # 2. COCI fallback (DOI-based, returns DOIs only → needs resolution)
    if doi:
        try:
            dois = _coci_get_references(doi)
            if dois:
                resolved = _resolve_dois(dois, limit)
                if resolved:
                    return resolved
        except Exception:
            pass

    # 3. SS fallback (via arXiv ID — covers CS/AI papers OA misses)
    arxiv_id = paper.get("arxiv_id", "")
    if arxiv_id:
        try:
            ss_id = f"ArXiv:{arxiv_id}"
            refs = _ss_get_references(ss_id, limit=limit)
            if refs:
                return refs
        except Exception:
            pass

    return []


def _fetch_citations(paper: dict, limit: int) -> list[dict]:
    """Fetch forward citations with OA-first, anomaly-checked multi-API fallback.

    Fallback chain: OA → COCI (each level validates before accepting).
    """
    oa_id = paper.get("_oa_id", "")
    doi = paper.get("_doi", "")

    # 1. OA primary: filter=cites:{oa_id} returns citing papers
    if oa_id:
        cached = _cache_get("oa_citations", oa_id)
        if cached:
            return cached[:limit]
        try:
            cites = _openalex_get_citations(oa_id, limit=limit)
            if not _cites_anomalous(cites, paper):
                _cache_put("oa_citations", oa_id, cites)
                return cites
        except Exception:
            pass

    # 2. COCI fallback (DOI-based)
    if doi:
        try:
            dois = _coci_get_citations(doi)
            if dois:
                cites = _resolve_dois(dois, limit)
                if cites:
                    for c in cites:
                        c["source"] = "forward_citation"
                    return cites
        except Exception:
            pass

    # 3. SS fallback (via arXiv ID)
    arxiv_id = paper.get("arxiv_id", "")
    if arxiv_id:
        try:
            ss_id = f"ArXiv:{arxiv_id}"
            cites = _ss_get_citations(ss_id, limit=limit)
            if cites:
                for c in cites:
                    c["source"] = "forward_citation"
                return cites
        except Exception:
            pass

    return []


def _compute_bibliographic_coupling(
    graph: nx.DiGraph,
    node_data: dict[str, dict],
    seed_ids: set[str],
    l1_nodes: set[str],
    min_shared: int = 3,
    threshold: float = 0.15,
) -> set[tuple]:
    """Step 3b: Compute bibliographic coupling edges between L1 nodes.

    Uses Ochiai coefficient: shared_refs / sqrt(|refs_A| * |refs_B|).

    Only computes coupling among L1 nodes (seeds + direct neighbors).
    This keeps computation O(n²) on the smaller L1 set, not the full graph.

    Returns set of (node_a, node_b) tuples representing coupling edges.
    Both directions are included for the DiGraph.
    """
    # Collect reference sets for each L1 node (from node_data and graph edges)
    l1_refs: dict[str, set[str]] = {}
    for nid in l1_nodes:
        # References are outgoing edges (nid → ref)
        refs = set()
        for _, target in graph.out_edges(nid):
            refs.add(target)
        l1_refs[nid] = refs

    l1_list = list(l1_nodes)
    n_l1 = len(l1_list)
    if n_l1 < 2:
        return set()

    coupling_edges: set[tuple] = set()
    coupling_count = 0

    for i in range(n_l1):
        a = l1_list[i]
        refs_a = l1_refs.get(a, set())
        if len(refs_a) < min_shared:
            continue

        for j in range(i + 1, n_l1):
            b = l1_list[j]
            refs_b = l1_refs.get(b, set())
            if len(refs_b) < min_shared:
                continue

            shared = len(refs_a & refs_b)
            if shared < min_shared:
                continue

            ochiai = shared / math.sqrt(len(refs_a) * len(refs_b))
            if ochiai >= threshold:
                coupling_edges.add((a, b))
                coupling_edges.add((b, a))  # bidirectional for DiGraph
                coupling_count += 1

    if coupling_count:
        print(f"  Bibliographic coupling: {coupling_count} edges"
              f" among {n_l1} L1 nodes"
              f" (min_shared={min_shared}, threshold={threshold})")

    return coupling_edges


def _build_citation_graph(
    resolved_seeds: list[dict],
    *,
    output_dir: Optional[Path] = None,
) -> tuple[nx.DiGraph, dict[str, dict], set[str], set[tuple]]:
    """Step 3: Build citation graph with bibliographic coupling.

    Level 1: Expand each seed (backward + forward citations).
    Level 2: Expand top-cited L1 papers.
    Level 3: Expand recent high-citation L2 papers (year >= 2024, cit > 50).
    Step 3b: Bibliographic coupling among L1 nodes.

    When output_dir is given, saves per-seed raw citation data for debugging.

    Returns:
        (graph, node_data, seed_ids, coupling_edges) — graph is a DiGraph where
        edges point from citer to cited. node_data maps node_id → paper dict.
    """
    graph = nx.DiGraph()
    node_data: dict[str, dict] = {}
    seed_ids: set[str] = set()
    _TITLE_NODE_MAP.clear()

    L1_BACKWARD = config.V3_L1_BACKWARD_LIMIT
    L1_FORWARD = config.V3_L1_FORWARD_LIMIT
    L2_SEEDS_N = config.V3_L2_SEEDS
    L2_BACKWARD = config.V3_L2_BACKWARD_LIMIT
    L2_FORWARD = config.V3_L2_FORWARD_LIMIT

    # Per-seed citation data for debugging
    per_seed_data: dict[str, dict] = {}

    # Index seeds
    for seed in resolved_seeds:
        sid = _add_node(graph, node_data, seed)
        seed_ids.add(sid)

    processed_refs: set[str] = set()
    processed_cites: set[str] = set()

    workers = max(1, config.V3_API_WORKERS)

    # ── Level 1: Seed expansion ──
    print(f"\n  [Citation Graph] Level 1: expanding {len(seed_ids)} seeds"
          f" (OA primary, COCI fallback, {workers} workers)")
    l1_tasks = []
    for i, sid in enumerate(list(seed_ids)):
        sdata = node_data.get(sid, {})
        # V3.3.5: Skip venue supplements in L1 expansion. They are matched by
        # keyword+venue, often low-citation, and inflate graph noise when expanded.
        # They remain graph nodes (for ranking) but are not further crawled.
        if (sdata.get("_contribution") or "").startswith("venue supplement"):
            print(f"    seed {i+1}/{len(seed_ids)}: skip venue supplement "
                  f"'{sdata.get('title', sid)[:50]}' (no expansion)")
            continue
        seed_cit_count = sdata.get("citation_count", 0)
        # Adaptive forward citation limit: low-cit seeds get larger limit
        adaptive_forward = L1_FORWARD
        if seed_cit_count < 100:
            adaptive_forward = min(int(L1_FORWARD * 100 / max(seed_cit_count, 1)), 100)
        l1_tasks.append((i, sid, sdata, adaptive_forward))

    def _expand_l1(task):
        i, sid, sdata, adaptive_forward = task
        refs = _fetch_references(sdata, limit=L1_BACKWARD)
        cites = _fetch_citations(sdata, limit=adaptive_forward)
        # V3.3.10: also crawl the arXiv-hosted alternate records of this seed
        # (published-version twins) for forward citations — papers citing the
        # preprint version are mostly new papers the published record misses.
        alts = sdata.get("_alt_oa_ids", [])
        alt_cites: list[dict] = []
        alt_limit = max(1, adaptive_forward // (1 + len(alts)))
        for alt in alts:
            alt_paper = {"_oa_id": alt, "title": sdata.get("title", "")}
            alt_cites.extend(_fetch_citations(alt_paper, limit=alt_limit))
        return (i, sid, refs, cites, alt_cites, adaptive_forward)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
        l1_results = list(ex.map(_expand_l1, l1_tasks))

    for i, sid, refs, cites, alt_cites, adaptive_forward in l1_results:
        sdata = node_data.get(sid, {})
        title_snip = sdata.get("title", sid)[:50]
        seed_cit_count = sdata.get("citation_count", 0)

        for ref in refs:
            rid = _add_node(graph, node_data, ref)
            graph.add_edge(sid, rid)  # seed cites reference
        for cite_paper in cites + alt_cites:
            cid = _add_node(graph, node_data, cite_paper)
            graph.add_edge(cid, sid)  # citation cites seed
        processed_refs.add(sid)
        processed_cites.add(sid)

        if refs:
            print(f"    seed {i+1}/{len(seed_ids)}: +{len(refs)} refs from '{title_snip}'")
        if cites or alt_cites:
            print(f"    seed {i+1}/{len(seed_ids)}: +{len(cites)} citations of '{title_snip}'"
                  f" (adaptive limit={adaptive_forward}, alt +{len(alt_cites)})")

        per_seed_data[sid] = {
            "title": sdata.get("title", ""),
            "n_refs": len(refs),
            "n_cites": len(cites) + len(alt_cites),
            "n_alt_cites": len(alt_cites),
            "citation_count": seed_cit_count,
            "adaptive_forward_limit": adaptive_forward,
            "ref_titles": [r.get("title", "?")[:80] for r in refs],
            "cite_titles": [c.get("title", "?")[:80] for c in cites],
            "ref_ids": [_add_node(graph, node_data, r) for r in refs],
            "cite_ids": [_add_node(graph, node_data, c) for c in cites],
        }

    n_l1 = len(graph.nodes())
    l1_node_set = set(graph.nodes())  # snapshot for coupling

    # ── Level 2: Expand top-cited L1 papers ──
    l1_candidates = []
    for nid, ndata in node_data.items():
        if nid not in seed_ids:
            l1_candidates.append((nid, ndata.get("citation_count", 0)))
    l1_candidates.sort(key=lambda x: -x[1])
    l2_seed_ids = {nid for nid, _ in l1_candidates[:L2_SEEDS_N]}

    if l2_seed_ids:
        print(f"\n  [Citation Graph] Level 2: expanding {len(l2_seed_ids)} top-cited L1 papers")

        def _expand_l2(lid):
            ldata = node_data.get(lid, {})
            refs = _fetch_references(ldata, limit=L2_BACKWARD) if lid not in processed_refs else []
            cites = _fetch_citations(ldata, limit=L2_FORWARD) if lid not in processed_cites else []
            return lid, refs, cites

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            l2_results = list(ex.map(_expand_l2, list(l2_seed_ids)))

        for lid, refs, cites in l2_results:
            for ref in refs:
                rid = _add_node(graph, node_data, ref)
                graph.add_edge(lid, rid)
            for cite_paper in cites:
                cid = _add_node(graph, node_data, cite_paper)
                graph.add_edge(cid, lid)
            processed_refs.add(lid)
            processed_cites.add(lid)

    # ── Level 3: Expand recent high-citation L2 papers ──
    # Capture papers that cite these recent works (which L1 expansion may miss)
    l3_candidates = []
    for nid, ndata in node_data.items():
        year = ndata.get("year", 0)
        cit = ndata.get("citation_count", 0)
        if nid not in seed_ids and nid not in l2_seed_ids and nid not in l1_node_set:
            # These were added during L2 expansion
            if isinstance(year, int) and year >= 2024 and cit > 50:
                l3_candidates.append((nid, cit))
    # Also check L2 seeds themselves that are recent + high-cit
    for nid in l2_seed_ids:
        ndata = node_data.get(nid, {})
        year = ndata.get("year", 0)
        cit = ndata.get("citation_count", 0)
        if isinstance(year, int) and year >= 2024 and cit > 50:
            l3_candidates.append((nid, cit))

    if l3_candidates:
        # Deduplicate and limit
        seen_l3 = set()
        unique_l3 = []
        for nid, cit in l3_candidates:
            if nid not in seen_l3:
                seen_l3.add(nid)
                unique_l3.append((nid, cit))
        l3_limit = min(len(unique_l3), 10)
        print(f"\n  [Citation Graph] Level 3: expanding {l3_limit} recent high-cit L1/L2 papers"
              f" (year >= 2024, cit > 50)")

        def _expand_l3(nid):
            ldata = node_data.get(nid, {})
            cites = _fetch_citations(ldata, limit=50) if nid not in processed_cites else []
            return nid, cites

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            l3_results = list(ex.map(_expand_l3, [nid for nid, _ in unique_l3[:l3_limit]]))

        for nid, cites in l3_results:
            for cite_paper in cites:
                cid = _add_node(graph, node_data, cite_paper)
                graph.add_edge(cid, nid)
            processed_cites.add(nid)

    # ── Step 3b: Bibliographic coupling ──
    coupling_edges = _compute_bibliographic_coupling(
        graph, node_data, seed_ids, l1_node_set,
        min_shared=config.V3_COUPLING_MIN_SHARED,
        threshold=config.V3_COUPLING_THRESHOLD,
    )

    # Add coupling edges to graph (with weight 0.5 vs citation weight 1.0)
    for a, b in coupling_edges:
        if graph.has_node(a) and graph.has_node(b):
            graph.add_edge(a, b, weight=0.5, edge_type="bibliographic_coupling")

    # V3.3.11: SS citation enrichment for arXiv-hosted nodes (before ranking)
    _enrich_ss_citations(node_data)

    n_total = len(graph.nodes())
    n_edges = len(graph.edges())
    n_coupling = len(coupling_edges) // 2  # bidirectional, count unique pairs
    print(f"\n  Citation graph: {n_total} nodes, {n_edges} edges"
          f" ({len(seed_ids)} seeds, {n_l1} after L1, {n_coupling} coupling pairs)")

    # Save per-seed citation data if output_dir is given
    if output_dir and per_seed_data:
        path = output_dir / "step_3_per_seed_citations.json"
        try:
            path.write_text(json.dumps(
                _make_json_safe(per_seed_data), ensure_ascii=False, indent=2
            ), encoding="utf-8")
            print(f"  Saved per-seed citation data -> {path.name}")
        except Exception as e:
            print(f"  Warning: failed to save per-seed citation data: {e}")

    return graph, node_data, seed_ids, coupling_edges


def _make_json_safe(obj):
    """Convert numpy/non-serializable objects for JSON dump."""
    import numpy as np
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_make_json_safe(v) for v in obj]
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    return obj


# ── Step 4: Graph ranking ──


def _compute_seed_proximity(
    graph: nx.DiGraph,
    seed_ids: set[str],
) -> dict[str, float]:
    """Compute min distance from each node to any seed (BFS on undirected version)."""
    ug = graph.to_undirected()
    proximity = {}
    for node in graph.nodes():
        min_dist = float('inf')
        for sid in seed_ids:
            try:
                dist = nx.shortest_path_length(ug, node, sid)
                min_dist = min(min_dist, dist)
            except nx.NetworkXNoPath:
                continue
        proximity[node] = 1.0 / (1.0 + min_dist) if min_dist < float('inf') else 0.0
    return proximity


def _rank_papers(
    graph: nx.DiGraph,
    node_data: dict[str, dict],
    seed_ids: set[str],
    beta: float | None = None,
    gamma: float | None = None,
) -> list[dict]:
    """Step 4: Rank papers by graph_score.

    graph_score = γ·citation_rate + β·seed_proximity（+ seed boost）

    V3.3.14: PageRank and coupling_degree removed. PageRank on the
    seed-expanded snowball graph measured in-graph citation coverage
    (67% of nodes had no in-graph in-edges), not importance. Citation
    signal upgraded to age-adjusted rate (citations per year) — the same
    "same-year relative strength" principle as the venue admission gate.
    V3.3.8: field relevance multiplier removed — LLM classification (Step 5)
    is the sole field-relevance gate.
    Default weights: γ=0.80, β=0.20
    """
    if beta is None:
        beta = config.V3_BETA_PROXIMITY
    if gamma is None:
        gamma = config.V3_GAMMA_CITATION

    n_nodes = len(graph.nodes())
    print(f"\n  [Graph Ranking] {n_nodes} nodes (γ={gamma:.2f}, β={beta:.2f})")

    # V3.3.12: Anchor set = LLM seeds only. Venue supplements are bulk
    # retrievals with lower trust than LLM-curated seeds; proximity and
    # PageRank personalization encode distance to the curated anchors.
    llm_seed_ids = {
        sid for sid in seed_ids
        if not (node_data.get(sid, {}).get("_contribution") or "").startswith("venue supplement")
    }

    # V3.3.12: Auto seed promotion — a non-seed cited directly by >= N seeds
    # with high post-enrichment citations is field-critical by structural
    # evidence. Gets boost + Step 5 admission, but NOT anchor status.
    promoted_ids: set[str] = set()
    if seed_ids:
        for node in graph.nodes():
            if node in seed_ids:
                continue
            # Citation edges only — coupling edges (edge_type set) are
            # reference-sharing, not "the seed cites this paper".
            n_citing_seeds = sum(
                1 for s in seed_ids
                if graph.has_edge(s, node) and "edge_type" not in graph.edges[s, node]
            )
            if (n_citing_seeds >= config.V3_PROMOTE_SEED_IN_EDGES
                    and node_data.get(node, {}).get("citation_count", 0) >= config.V3_PROMOTE_MIN_CIT):
                promoted_ids.add(node)
    if promoted_ids:
        print(f"  Auto promotion: {len(promoted_ids)} non-seed papers"
              f" (>= {config.V3_PROMOTE_SEED_IN_EDGES} seed in-edges,"
              f" >= {config.V3_PROMOTE_MIN_CIT} cit)")

    # 1. Citation rate (age-adjusted, log-scale normalized)
    current_year = datetime.datetime.now().year
    cit_rate = {}
    for node, d in node_data.items():
        cc = d.get("citation_count", 0)
        age = max(current_year - int(d.get("year") or current_year), 1)
        cit_rate[node] = math.log1p(cc / age)
    max_rate = max(cit_rate.values()) if cit_rate else 1.0
    cit_rate_norm = {n: v / max_rate for n, v in cit_rate.items()}

    # 2. Seed proximity
    proximity = _compute_seed_proximity(graph, llm_seed_ids)

    # 3. Normalize proximity to [0, 1]
    prox_vals = list(proximity.values())
    prox_max = max(prox_vals) + 1e-10
    prox_norm = {n: v / prox_max for n, v in proximity.items()}

    # 4. Weighted combination
    scored = []
    seed_boost = config.V3_SEED_BOOST
    for node in graph.nodes():
        score = (
            gamma * cit_rate_norm.get(node, 0)
            + beta * prox_norm.get(node, 0)
        )
        # Seed boost: LLM seeds and V3.3.12-promoted papers get full boost;
        # venue supplements get none (V3.3.11: SS enrichment fixed the
        # citation undercount this compensated for).
        if node in seed_ids:
            contrib = node_data.get(node, {}).get("_contribution", "")
            if not contrib.startswith("venue supplement"):
                score += seed_boost
        elif node in promoted_ids:
            score += seed_boost
        entry = dict(node_data.get(node, {}))
        entry["_node_id"] = node
        entry["graph_score"] = round(float(score), 4)
        entry["citation_rate"] = round(float(cit_rate_norm.get(node, 0)), 4)
        entry["seed_proximity"] = round(float(proximity.get(node, 0)), 4)
        entry["is_seed"] = node in seed_ids
        entry["_promoted_seed"] = node in promoted_ids
        scored.append(entry)

    scored.sort(key=lambda p: -p["graph_score"])

    if scored:
        print(f"  Top by graph_score:")
        for i, p in enumerate(scored[:5]):
            seed_flag = " [S]" if p["is_seed"] else ""
            print(f"    {i+1}. ({p['graph_score']:.4f}) {p['title'][:70]}"
                  f" ({p.get('year','?')}, {p.get('citation_count',0)} cit){seed_flag}")

    return scored


# ── Step 5: Diversified selection ──

# ── LLM-based paper classification ──

_CLASSIFY_SYSTEM = """\
You are a research area specialist. Given a list of papers and a research field,
classify each paper into one of five categories based on its ROLE in that field.

Categories:
- CORE: A technical method paper that solves a core problem of THIS SPECIFIC
  FIELD. Researchers in this field would directly compare against or build upon
  this paper as related work. The paper's primary contribution lies within the
  boundaries of the field.
- ADJACENT: A technical method paper from a broader or adjacent field that is
  relevant context but NOT specific to this field. The field references it as
  prior art, background, or baseline, but it belongs to a different sub-area.
- DATASET: A dataset, benchmark, infrastructure, or evaluation framework. These
  enable research but are not methods themselves.
- FOUNDATION: A foundational paper from a broader or adjacent field that this
  field builds upon (backbone architectures, general frameworks, generic
  techniques from parent fields). Important but not specific to this field.
- NOISE: A paper that is not about this field, not a foundation, and not a
  dataset. It appears in the citation graph by accident.
- REVIEW: A survey, review, benchmark recipe, or tutorial that summarizes or
  evaluates the field but contributes no new method. Not CORE even if
  published in the field's own venues.

Return a JSON array of objects."""

_CLASSIFY_PROMPT = """\
Research field: **{field_name}**

Below are candidate papers. For each, classify its role in the field above.

Key distinction:
- CORE = this paper is a method within the field. It solves the field's problem.
- ADJACENT = this paper is a method from a different sub-area, cited as context.
  The field's papers cite it as background or baseline, not as direct related work.

{paper_entries}

Return JSON:
```json
[
  {{
    "index": 0,
    "category": "CORE|ADJACENT|DATASET|FOUNDATION|NOISE|REVIEW",
    "reason": "one-sentence explanation"
  }}
]
```"""

# V3.3.6: Venue supplements were discovered by searching the field's own
# primary venues with field keywords — that provenance is evidence of field
# membership. Classifying them with this context removes the CORE/ADJACENT
# flip observed for sparse-direction papers (e.g. Sparse4D) at temperature 0.
# V3.3.12: moved from system-prompt extension to user-prompt prefix — venue
# provenance is batch-level information, not a classification rule.
_CLASSIFY_VENUE_CONTEXT = """\
Note: All papers below were discovered by searching the research field's own
primary publication venues (top conferences/journals of this field) with the
field's keywords. Publication in the field's own venues is evidence of field
membership — it helps distinguish CORE (in-field method) from ADJACENT
(out-of-field method). It does NOT override the method requirement: a paper
whose primary contribution is a dataset, benchmark, or evaluation is still
DATASET; a survey, review, or recipe is still REVIEW; a foundation is still
FOUNDATION."""


def _llm_classify_papers(
    papers: list[dict],
    field_name: str,
    client: OpenAI,
    model: Optional[str] = None,
    *,
    system: Optional[str] = None,
    context: Optional[str] = None,
) -> dict[str, str]:
    """Classify papers into CORE/DATASET/FOUNDATION/NOISE via LLM.

    Returns dict mapping title -> category.
    Empty dict on failure (caller falls back to heuristic).

    Papers are classified in chunks: reasoning models spend hidden tokens on
    reasoning, so one giant batch can exhaust max_tokens (empty content,
    finish_reason=length) or exceed the request timeout.
    """
    if not papers or not client:
        return {}

    model = _resolve_model(model)
    if not model:
        return {}

    system = system or _CLASSIFY_SYSTEM

    result: dict[str, str] = {}
    chunk_size = 40

    for start in range(0, len(papers), chunk_size):
        chunk = papers[start:start + chunk_size]

        entries = []
        for i, p in enumerate(chunk):
            title = p.get("title", "?")[:120]
            year = p.get("year", "?")
            cit = p.get("citation_count", 0)
            abstract = (p.get("abstract") or "")[:300]
            entries.append(f"[{i}] {title} ({year}, {cit} citations)\n    Abstract: {abstract}")

        prompt = _CLASSIFY_PROMPT.format(
            field_name=field_name,
            paper_entries="\n\n".join(entries),
        )
        # V3.3.12: batch-level context (e.g. venue provenance) goes in the user
        # prompt, keeping the system classification standard identical.
        if context:
            prompt = context + "\n\n" + prompt

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                temperature=0,
                max_tokens=config.LLM_ANALYZER_MAX_TOKENS,
                timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
            )
            raw = resp.choices[0].message.content or ""
            if not raw:
                print(f"  LLM classification: empty response "
                      f"(finish_reason={resp.choices[0].finish_reason}, chunk {start//chunk_size + 1})")
                continue
            data = _extract_json_array(raw)
            if not data or not isinstance(data, list):
                print(f"  LLM classification: unparseable JSON (chunk {start//chunk_size + 1})")
                continue

            for item in data:
                idx = item.get("index")
                cat = item.get("category", "").upper()
                if 0 <= idx < len(chunk) and cat in ("CORE", "ADJACENT", "DATASET", "FOUNDATION", "NOISE", "REVIEW"):
                    result[chunk[idx]["title"]] = cat

        except Exception as e:
            print(f"  LLM classification failed (chunk {start//chunk_size + 1}): {e}")
            continue

    n_core = sum(1 for v in result.values() if v == "CORE")
    n_adj = sum(1 for v in result.values() if v == "ADJACENT")
    n_ds = sum(1 for v in result.values() if v == "DATASET")
    n_fd = sum(1 for v in result.values() if v == "FOUNDATION")
    n_ns = sum(1 for v in result.values() if v == "NOISE")
    n_rv = sum(1 for v in result.values() if v == "REVIEW")
    print(f"  LLM classification: {len(result)} papers "
          f"(CORE={n_core}, ADJACENT={n_adj}, DATASET={n_ds}, FOUNDATION={n_fd}, "
          f"NOISE={n_ns}, REVIEW={n_rv})")
    return result


def _diversified_select(
    ranked_papers: list[dict],
    seed_ids: set[str],
    max_papers: int = 20,
    *,
    client: Optional[OpenAI] = None,
    field_name: str = "",
    model: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> list[dict]:
    """Step 5: Select max_papers with temporal and sub-direction diversity.

    Strategy:
    0. LLM classification — classify top candidates into CORE/ADJACENT/DATASET/FOUNDATION/NOISE/REVIEW
    0a. Filter: keep only CORE papers (field methods), exclude adjacent, datasets, foundations, noise
    1. Year bucketing — ensure each major period has representation
    2. Sub-direction dedup — avoid too many papers from the same author group
    3. Seed protection — ensure resolved seeds are represented
    4. Non-seed quota (V3.3.12) — guarantee graph-discovered papers are represented
    """
    if len(ranked_papers) <= max_papers:
        return ranked_papers

    # ── Step 0: LLM classification ──
    # Take top-K by graph_score, PLUS all seeds and promoted papers (regardless of rank)
    top_k = min(config.V3_CLASSIFY_TOP_K, len(ranked_papers))
    candidates = ranked_papers[:top_k]
    # Include all seeds that aren't already in top-K.
    # V3.3.13: venue supplements gated by same-year citation percentile —
    # their unconditional admission dated from V3.3.6 (OA citation undercount
    # on arXiv papers), which V3.3.11 SS enrichment fixed. LLM seeds remain
    # unconditional (curated prior).
    seen_titles = {p["title"].lower().strip() for p in candidates}
    # V3.3.13: percentile computed within the venue-supplement pool per year —
    # the venue pool is the field's venue-paper elite (top-5 cited per venue
    # per year); comparing against the whole graph cohort would be unfair
    # (graph population is the citation neighborhood of high-cited seeds).
    year_cits: dict[int, list[int]] = defaultdict(list)
    for p in ranked_papers:
        if (p.get("_contribution") or "").startswith("venue supplement"):
            year_cits[p.get("year", 0)].append(p.get("citation_count", 0))
    year_pctl: dict[int, int] = {}
    pctl = config.V3_VENUE_ADMIT_YEAR_PCTL
    for y, cits in year_cits.items():
        if len(cits) >= 3:
            s = sorted(cits)
            year_pctl[y] = s[min(len(s) - 1, int(pctl / 100 * (len(s) - 1)))]
    n_venue_pass = n_venue_fail = 0
    for p in ranked_papers:
        if not p.get("is_seed") or p["title"].lower().strip() in seen_titles:
            continue
        if (p.get("_contribution") or "").startswith("venue supplement"):
            bar = year_pctl.get(p.get("year", 0))
            if bar is not None and p.get("citation_count", 0) < bar:
                n_venue_fail += 1
                continue
            n_venue_pass += 1
        candidates.append(p)
        seen_titles.add(p["title"].lower().strip())
    if n_venue_pass or n_venue_fail:
        print(f"  Venue admission: {n_venue_pass}/{n_venue_pass + n_venue_fail}"
              f" venue supplements >= year-P{pctl} of venue pool"
              f" (LLM seeds unconditional)")
    # V3.3.12: promoted seeds also get admission (structural evidence: cited
    # by multiple seeds + high citations) regardless of rank
    for p in ranked_papers:
        if p.get("_promoted_seed") and p["title"].lower().strip() not in seen_titles:
            candidates.append(p)
            seen_titles.add(p["title"].lower().strip())
    # V3.3.6: venue supplements are classified with venue-provenance context —
    # publication in the field's own venues stabilizes CORE/ADJACENT flips that
    # happen at temperature=0 for reasoning models (e.g. Sparse4D).
    if client and field_name:
        venue_candidates = [p for p in candidates
                            if (p.get("_contribution") or "").startswith("venue supplement")]
        other_candidates = [p for p in candidates if p not in venue_candidates]
        classified = _llm_classify_papers(other_candidates, field_name, client, model)
        if venue_candidates:
            classified.update(_llm_classify_papers(venue_candidates, field_name, client,
                                                   model, context=_CLASSIFY_VENUE_CONTEXT))
    else:
        classified = {}

    # Store category on each paper
    for p in ranked_papers:
        p["_llm_category"] = classified.get(p["title"], "")

    # V3.3.12: persist classification results for debugging — previously
    # _llm_category lived only in memory and step_4 was saved before
    # classification ran, leaving the debug JSON category column always empty.
    if output_dir:
        try:
            cls_out = sorted(
                ({"title": p["title"],
                  "category": p.get("_llm_category", ""),
                  "citation_count": p.get("citation_count", 0),
                  "graph_score": p.get("graph_score", 0),
                  "year": p.get("year", 0),
                  "is_seed": bool(p.get("is_seed")),
                  "promoted": bool(p.get("_promoted_seed")),
                  "venue_supplement": (p.get("_contribution") or "").startswith("venue supplement")}
                 for p in candidates),
                key=lambda x: (x["category"], -x["citation_count"]),
            )
            with open(os.path.join(output_dir, "step_5_classification.json"),
                      "w", encoding="utf-8") as f:
                json.dump(cls_out, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"  Failed to save classification debug JSON: {e}")

    # ── Step 0a: Filter — keep only CORE papers ──
    # V3.3.6: LLM seeds classified CORE/ADJACENT are protected — the seed list
    # is a stronger curation signal than the stochastic classification, and a
    # seed is a field method by construction (ADJACENT = classifier under-rating).
    # DATASET/FOUNDATION/NOISE/REVIEW vetoes stand; venue supplements are weak signals.
    core_pool = []
    protected_papers = []
    for p in ranked_papers:
        cat = p.get("_llm_category", "")
        is_llm_seed = p.get("is_seed") and not (p.get("_contribution") or "").startswith("venue supplement")
        if cat == "CORE":
            core_pool.append(p)
            if is_llm_seed:
                protected_papers.append(p)
        elif cat == "ADJACENT" and is_llm_seed:
            core_pool.append(p)
            protected_papers.append(p)

    # If LLM classification failed or no CORE found, fall back to all papers
    if not classified or not core_pool:
        core_pool = ranked_papers

    # V3.3.4: Deduplicate by normalized title before size check — same paper
    # can enter the graph via two different OA entries (seed resolution vs
    # citation expansion), and both get the same _llm_category by title match.
    # Keep the one with higher graph_score.
    core_pool.sort(key=lambda p: (-p["graph_score"], -p.get("citation_count", 0)))
    core_pool = _dedup_by_title(core_pool)

    if len(core_pool) <= max_papers:
        return core_pool

    # ── Year bucketing (on core pool) ──
    # V3.3.8: field_relevance mechanism deleted — LLM classification is the sole
    # relevance gate; core_pool is already CORE-only and sorted by graph_score.
    year_groups: dict[int, list[dict]] = defaultdict(list)
    for p in core_pool:
        y = p.get("year", 0)
        year_groups[y].append(p)

    years = sorted(year_groups.keys())
    n_years = len(years)
    if n_years == 0:
        return ranked_papers[:max_papers]

    min_per_year = max(1, max_papers // n_years)
    allocated: dict[int, int] = {}
    for y in years:
        allocated[y] = min(min_per_year, len(year_groups[y]))

    used = sum(allocated.values())
    remaining = max_papers - used
    if remaining > 0:
        total_papers = sum(len(v) for v in year_groups.values())
        for y in years:
            if remaining <= 0:
                break
            extra = max(1, int(remaining * len(year_groups[y]) / total_papers))
            extra = min(extra, len(year_groups[y]) - allocated[y])
            allocated[y] += extra
            remaining -= extra

    # Select per bucket (top by graph_score within each year)
    selected: list[dict] = []
    selected_titles: set[str] = set()
    for y in years:
        bucket = year_groups[y]
        bucket.sort(key=lambda p: -p["graph_score"])
        for p in bucket[:allocated[y]]:
            t = p["title"].lower().strip()
            if t not in selected_titles:
                selected.append(p)
                selected_titles.add(t)

    # ── Sub-direction dedup ──
    author_groups: dict[str, list[dict]] = defaultdict(list)
    for p in selected:
        authors = p.get("authors", []) or []
        first = authors[0] if authors else "unknown"
        author_groups[first].append(p)

    for author, group in author_groups.items():
        if len(group) > 2:
            group.sort(key=lambda p: -p["graph_score"])
            to_remove = group[2:]
            for p in to_remove:
                selected.remove(p)
                selected_titles.discard(p["title"].lower().strip())

    # ── Fill from remaining core papers ──
    if len(selected) < max_papers:
        for p in core_pool:
            if len(selected) >= max_papers:
                break
            t = p["title"].lower().strip()
            if t not in selected_titles:
                selected.append(p)
                selected_titles.add(t)

    selected.sort(key=lambda p: (-p["graph_score"], -p.get("citation_count", 0)))

    # V3.3.4: Deduplicate by normalized title — same paper can enter the graph
    # via two different OA entries (seed resolution vs citation expansion).
    # Keep the one with higher graph_score.
    deduped = _dedup_by_title(selected)
    n_removed = len(selected) - len(deduped)
    if n_removed:
        print(f"  Deduplication: removed {n_removed} duplicates by title")
    deduped.sort(key=lambda p: (-p["graph_score"], -p.get("citation_count", 0)))

    # V3.3.6: Seed protection — LLM seeds classified CORE/ADJACENT must be in
    # the final output. Missing ones replace the lowest-scored non-protected
    # papers (design doc Step 5: 种子保护).
    final = deduped[:max_papers]
    if protected_papers:
        protected_titles = {p["title"].lower().strip() for p in protected_papers}
        final_titles = {p["title"].lower().strip() for p in final}
        n_restored = 0
        for m in protected_papers:
            m_t = m["title"].lower().strip()
            if m_t in final_titles:
                continue
            victim = None
            for p in reversed(final):
                if p["title"].lower().strip() not in protected_titles:
                    victim = p
                    break
            if victim is None:
                break
            final[final.index(victim)] = m
            final_titles.add(m_t)
            final_titles.discard(victim["title"].lower().strip())
            n_restored += 1
        if n_restored:
            print(f"  Seed protection: restored {n_restored} LLM seeds into final selection")
        final.sort(key=lambda p: (-p["graph_score"], -p.get("citation_count", 0)))

    # V3.3.12: Non-seed quota — guarantee graph-discovered papers get in.
    # Replace lowest-scored unprotected seeds with highest-scored non-seed
    # CORE papers until quota met (or candidates exhausted).
    quota = min(config.V3_NONSEED_QUOTA, max_papers)
    if quota > 0:
        protected_titles = {p["title"].lower().strip() for p in protected_papers}
        n_nonseed = sum(1 for p in final if not p.get("is_seed"))
        if n_nonseed < quota:
            final_titles = {p["title"].lower().strip() for p in final}
            nonseed_pool = [p for p in core_pool
                            if not p.get("is_seed")
                            and p["title"].lower().strip() not in final_titles]
            n_added = 0
            for cand in nonseed_pool:
                if n_nonseed + n_added >= quota:
                    break
                victim = None
                for p in reversed(final):  # final sorted desc — lowest first
                    t = p["title"].lower().strip()
                    if p.get("is_seed") and t not in protected_titles:
                        victim = p
                        break
                if victim is None:
                    break
                final[final.index(victim)] = cand
                n_added += 1
            if n_added:
                print(f"  Non-seed quota: added {n_added} non-seed papers"
                      f" (quota={quota}, total non-seed={n_nonseed + n_added})")
            else:
                print(f"  Non-seed quota: unmet (quota={quota}, non-seed={n_nonseed},"
                      f" CORE non-seed supply exhausted)")
            final.sort(key=lambda p: (-p["graph_score"], -p.get("citation_count", 0)))

    print(f"  Diversified selection: {len(final)} papers"
          f" (years={sorted({p.get('year',0) for p in final})})")
    return final
