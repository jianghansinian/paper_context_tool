"""V3 Paper Retrieval — seed-driven citation graph architecture.

Usage:
    from paper_retriever_v3 import retrieve_field_papers_v3

    papers, report = retrieve_field_papers_v3("BEV Perception", client)
"""

from __future__ import annotations

from typing import Optional
from datetime import datetime
from pathlib import Path

import numpy as np
from openai import OpenAI

import config
from seed_generator import _generate_seeds, _resolve_seeds, _fallback_seeds
from citation_graph_builder import _build_citation_graph, _rank_papers, _diversified_select


def _make_json_safe(obj):
    """Convert numpy arrays and non-serializable objects to JSON-safe types."""
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


def retrieve_field_papers_v3(
    field_name: str,
    client: Optional[OpenAI] = None,
    *,
    max_papers: int = 20,
    fast: bool = False,
    model: Optional[str] = None,
    save_intermediates: bool = False,
) -> tuple[list[dict], dict]:
    """Retrieve representative papers for a research field (V3: seed → citation graph).

    Args:
        field_name: e.g. "BEV Perception".
        client: LLM client for seed generation.
        max_papers: maximum papers to return.
        fast: if True, skip LLM, use OpenAlex top-15 as seeds.
        model: optional model override.
        save_intermediates: if True, save step results to output dir.

    Returns:
        (papers, report) — papers: list[dict], report: dict with metadata.
    """
    print(f"\n{'='*60}")
    print(f"  Paper Retrieval V3: {field_name}")
    print(f"{'='*60}")

    out_dir = None
    if save_intermediates:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        import json
        out_dir = Path(f"output/v3_test_{ts}")
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output dir: {out_dir}")

    def _save(name, data):
        if out_dir:
            import json
            with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(_make_json_safe(data), f, ensure_ascii=False, indent=2)

    # ── Fast mode ──
    if fast or not client:
        if fast:
            print("  --fast mode: OpenAlex top-15 seeds, no LLM")
        print(f"  Searching OpenAlex for: {field_name}")
        from paper_retriever import _openalex_search
        candidates = _openalex_search(field_name, limit=100)
        from paper_retriever import _deduplicate
        candidates = _deduplicate(candidates)
        if not candidates:
            print("  No papers found.")
            return [], {"confirmed_missing": [], "total_graph_nodes": 0}
        candidates.sort(key=lambda c: (-c.get("citation_count", 0), -c.get("year", 0)))
        result = candidates[:max_papers]
        for c in result:
            c["rationale"] = "top by citations (fast mode)"
        _save("step_6_selected", result)
        return result, {
            "confirmed_missing": [],
            "total_graph_nodes": len(candidates),
            "resolved_seeds": 0,
            "unresolved_seeds": [],
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: Seed Generation (LLM call)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 1: Seed Generation")
    print(f"{'='*60}")

    seeds = _generate_seeds(field_name, client, model=model)
    _save("step_1_seeds_raw", seeds)

    if not seeds:
        print("  No seeds from LLM, using fallback")
        fallback = _fallback_seeds(field_name, count=15)
        # Convert resolved fallback papers to seed format
        seeds = [
            {
                "title": p["title"],
                "first_author": (p.get("authors", [""]) or [""])[0],
                "year": p.get("year", 0),
                "contribution": "",
            }
            for p in fallback
        ]
        # Skip resolution — fallback papers are already resolved
        resolved = fallback
        unresolved = []
        _save("step_1_seeds_raw", seeds)
    else:
        # ═══════════════════════════════════════════════════════════════════
        # Step 2: Seed Resolution (API calls)
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print("Step 2: Seed Resolution")
        print(f"{'='*60}")

        resolved, unresolved = _resolve_seeds(seeds)

        _save("step_2_resolved_seeds", resolved)
        _save("step_2_unresolved_seeds", unresolved)

        if not resolved:
            print("  WARNING: 0 seeds resolved. Falling back to OpenAlex top-15.")
            fallback = _fallback_seeds(field_name, count=15)
            resolved = fallback

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: Citation Graph Construction (API calls)
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 3: Citation Graph Construction")
    print(f"{'='*60}")

    graph, node_data, seed_ids, coupling_edges = _build_citation_graph(resolved)

    n_nodes = len(graph.nodes())
    n_edges = len(graph.edges())
    n_coupling = len(coupling_edges) // 2  # bidirectional pairs
    print(f"  Graph: {n_nodes} nodes, {n_edges} edges, {len(seed_ids)} seeds, {n_coupling} coupling pairs")

    _save("step_3_graph_stats", {
        "nodes": n_nodes,
        "edges": n_edges,
        "seeds": len(seed_ids),
    })

    if n_nodes == 0:
        print("  ERROR: Citation graph is empty. Falling back to OpenAlex top-20.")
        from paper_retriever import _openalex_search, _deduplicate
        candidates = _openalex_search(field_name, limit=100)
        candidates = _deduplicate(candidates)
        candidates.sort(key=lambda c: (-c.get("citation_count", 0), -c.get("year", 0)))
        return candidates[:max_papers], {
            "confirmed_missing": [],
            "total_graph_nodes": 0,
            "unresolved_seeds": unresolved,
            "error": "empty_graph",
        }

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4: Graph Ranking
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 4: Graph Ranking")
    print(f"{'='*60}")

    ranked = _rank_papers(graph, node_data, seed_ids)

    _save("step_4_ranked_top50", [
        {"title": p["title"], "year": p.get("year", 0),
         "citation_count": p.get("citation_count", 0),
         "graph_score": p["graph_score"],
         "citation_rate": p["citation_rate"],
         "seed_proximity": p["seed_proximity"],
         "is_seed": p["is_seed"],
         "source": p.get("source", "unknown")}
        for p in ranked[:50]
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5: Diversified Selection
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 5: Diversified Selection")
    print(f"{'='*60}")

    selected = _diversified_select(ranked, seed_ids, max_papers=max_papers,
                                   client=client, field_name=field_name, model=model,
                                   output_dir=str(out_dir) if out_dir else None)

    # Clean internal keys from output
    _internal_keys = {"_oa_id", "_oa_short", "_oa_s2_id", "_seed_title", "_contribution"}
    for p in selected:
        for k in _internal_keys:
            p.pop(k, None)

    _save("step_5_selected", selected)

    # ═══════════════════════════════════════════════════════════════════════
    # Step 6: Final Output
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")

    report = {
        "total_graph_nodes": n_nodes,
        "total_graph_edges": n_edges,
        "coupling_pairs": n_coupling,
        "resolved_seeds": len(resolved),
        "unresolved_seeds": unresolved,
        "total_selected": len(selected),
    }

    # Print result
    print(f"\nSelected ({len(selected)}):")
    for i, p in enumerate(selected):
        seed_flag = " [S]" if p.get("is_seed") else ""
        year = p.get("year", "?")
        cit = p.get("citation_count", 0)
        gs = p.get("graph_score", "?")
        print(f"  {i+1:2d}. ({gs:.4f}) ({year}) [{cit:4d} cit] {p['title'][:90]}{seed_flag}")

    _save("step_6_report", report)

    if unresolved:
        print(f"\nUnresolved seeds ({len(unresolved)}):")
        for us in unresolved:
            print(f"  ✗ {us.get('title','?')[:80]}"
                  f" ({us.get('first_author','?')}, {us.get('year','?')})"
                  f" — {us.get('reason','?')}")

    return selected, report
