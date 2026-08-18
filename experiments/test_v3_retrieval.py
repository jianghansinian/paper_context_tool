"""Step-by-step V3 retrieval test — saves every intermediate result to JSON."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

# Add src/ to path
SRC = Path(__file__).resolve().parent.parent / "src"
sys.path.insert(0, str(SRC))

from openai import OpenAI
import config
from embedding import build_embedding_client
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


def _save_step(out_dir: Path, step_name: str, data, extra_label: str = ""):
    """Save intermediate result as JSON."""
    suffix = f"_{extra_label}" if extra_label else ""
    path = out_dir / f"{step_name}{suffix}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(data), f, ensure_ascii=False, indent=2)
    if isinstance(data, list):
        print(f"  -> Saved {len(data)} items -> {path.name}")
    elif isinstance(data, dict):
        keys_summary = ", ".join(
            f"{k}={len(v) if isinstance(v, (list, dict)) else v}"
            for k, v in list(data.items())[:5]
        )
        print(f"  -> Saved -> {path.name}  ({keys_summary})")


def main():
    field_name = sys.argv[1] if len(sys.argv) > 1 else "BEV Perception"
    max_papers = config.V3_MAX_PAPERS

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"output/v3_test_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── Build clients ──
    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    model = config.LLM_ANALYZER_MODEL
    emb_client = build_embedding_client()
    print(f"LLM: {model} @ {config.LLM_BASE_URL}")

    # ═══════════════════════════════════════════════════════════════════════
    # Step 1: Seed Generation
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 1: Seed Generation (1 LLM call)")
    print(f"{'='*60}")

    seeds = _generate_seeds(field_name, client, model=model, output_dir=out_dir)
    _save_step(out_dir, "step_1_seeds_raw", seeds)

    if not seeds:
        print("No seeds from LLM, using fallback")
        fallback = _fallback_seeds(field_name, count=15)
        seeds = [
            {
                "title": p["title"],
                "first_author": (p.get("authors", [""]) or [""])[0],
                "year": p.get("year", 0),
                "contribution": "",
            }
            for p in fallback
        ]
        resolved = fallback
        unresolved = []
        _save_step(out_dir, "step_1_seeds_raw", seeds)
    else:
        # ═══════════════════════════════════════════════════════════════════
        # Step 2: Seed Resolution
        # ═══════════════════════════════════════════════════════════════════
        print(f"\n{'='*60}")
        print("Step 2: Seed Resolution (API calls)")
        print(f"{'='*60}")

        resolved, unresolved = _resolve_seeds(seeds)

        # Save simplified seeds info
        _save_step(out_dir, "step_2_resolved_seeds", [
            {"title": s["title"], "year": s.get("year", 0),
             "citation_count": s.get("citation_count", 0),
             "_oa_id": s.get("_oa_id", ""),
             "_contribution": s.get("_contribution", "")}
            for s in resolved
        ])
        _save_step(out_dir, "step_2_unresolved_seeds", unresolved)

        if not resolved:
            print("ERROR: 0 seeds resolved. Falling back to OpenAlex top-15.")
            fallback = _fallback_seeds(field_name, count=15)
            resolved = fallback
            _save_step(out_dir, "step_2_resolved_seeds_fallback", [
                {"title": s["title"], "year": s.get("year", 0),
                 "citation_count": s.get("citation_count", 0)}
                for s in resolved
            ])

    # ═══════════════════════════════════════════════════════════════════════
    # Step 3: Citation Graph Construction
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 3: Citation Graph Construction (API calls)")
    print(f"{'='*60}")

    graph, node_data, seed_ids, coupling_edges = _build_citation_graph(
        resolved, output_dir=out_dir,
    )

    n_nodes = len(graph.nodes())
    n_edges = len(graph.edges())
    n_coupling = len(coupling_edges) // 2
    _save_step(out_dir, "step_3_graph_stats", {
        "nodes": n_nodes,
        "edges": n_edges,
        "seeds": len(seed_ids),
        "coupling_pairs": n_coupling,
    })

    # Save graph edges for visualization (node_id → title mapping)
    edges_data = []
    for u, v in graph.edges():
        u_title = node_data.get(u, {}).get("title", u)
        v_title = node_data.get(v, {}).get("title", v)
        edges_data.append({"source": u, "target": v, "source_title": u_title, "target_title": v_title})
    _save_step(out_dir, "step_3_graph_edges", edges_data)

    if n_nodes == 0:
        print("ERROR: Empty citation graph. Exiting.")
        return

    # ═══════════════════════════════════════════════════════════════════════
    # Step 4: Graph Ranking
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 4: Graph Ranking (deterministic)")
    print(f"{'='*60}")

    ranked = _rank_papers(graph, node_data, seed_ids)
    _save_step(out_dir, "step_4_ranked_all", [
        {"title": p["title"], "year": p.get("year", 0),
         "citation_count": p.get("citation_count", 0),
         "graph_score": p["graph_score"],
         "citation_rate": p["citation_rate"],
         "seed_proximity": p["seed_proximity"],
         "is_seed": p["is_seed"],
         "_llm_category": p.get("_llm_category", ""),
         "source": p.get("source", "unknown")}
        for p in ranked
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # Step 5: Diversified Selection
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 5: Diversified Selection (deterministic)")
    print(f"{'='*60}")

    selected = _diversified_select(ranked, seed_ids, max_papers=max_papers,
                                    client=client, field_name=field_name, model=model,
                                    output_dir=str(out_dir))

    # Clean internal keys before saving
    _internal_keys = {"_oa_id", "_oa_short", "_oa_s2_id", "_seed_title", "_contribution"}
    for p in selected:
        for k in _internal_keys:
            p.pop(k, None)

    _save_step(out_dir, "step_5_selected", [
        {"title": p["title"], "year": p.get("year", 0),
         "citation_count": p.get("citation_count", 0),
         "graph_score": p["graph_score"],
         "is_seed": p["is_seed"],
         "_llm_category": p.get("_llm_category", ""),
         "source": p.get("source", "unknown")}
        for p in selected
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # Final Summary
    # ═══════════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")

    final = {
        "field_name": field_name,
        "total_seeds_generated": len(seeds),
        "seeds_resolved": len(resolved),
        "seeds_unresolved": len(unresolved),
        "total_graph_nodes": n_nodes,
        "total_graph_edges": n_edges,
        "coupling_pairs": n_coupling,
        "selected_count": len(selected),
        "selected_papers": [
            {"title": p["title"], "year": p.get("year", 0),
             "citation_count": p.get("citation_count", 0),
             "graph_score": p.get("graph_score", 0),
             "is_seed": p.get("is_seed", False),
             "source": p.get("source", "unknown")}
            for p in selected
        ],
        "unresolved_seeds": unresolved,
    }
    _save_step(out_dir, "final_summary", final)

    # Print concise result
    print(f"\nSelected ({len(selected)}):")
    for i, p in enumerate(selected):
        seed_flag = " [S]" if p.get("is_seed") else ""
        year = p.get("year", "?")
        cit = p.get("citation_count", 0)
        gs = p.get("graph_score", "?")
        print(f"  {i+1:2d}. ({gs:.4f}) ({year}) [{cit:4d} cit] {p['title'][:90]}{seed_flag}")

    if unresolved:
        print(f"\nUnresolved seeds ({len(unresolved)}):")
        for us in unresolved:
            print(f"  ? {us.get('title','?')[:80]} ({us.get('first_author','?')}, {us.get('year','?')})"
                  f" — {us.get('reason','?')}")

    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    main()
