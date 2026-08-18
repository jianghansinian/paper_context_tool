"""Step-by-step V2 retrieval test — saves every intermediate result to JSON."""

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
from paper_retriever import (
    _generate_query_expansion,
    _broad_recall,
    _deduplicate,
    _openalex_enrich,
    _citation_expand,
    _survey_calibrate,
    _generate_milestone_descriptions,
    _milestone_guided_search,
    _embedding_match,
    _embedding_select,
    _llm_unified_select,
    _closed_loop_recover,
    _pre_rank,
    _PRE_RANK_TOP_N,
    _cosine_similarity,
    _text_for_embedding,
    _compute_embeddings,
)


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


def _save_step(out_dir: Path, step_name: str, data: dict | list, extra_label: str = ""):
    """Save intermediate result as JSON."""
    suffix = f"_{extra_label}" if extra_label else ""
    path = out_dir / f"{step_name}{suffix}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(data), f, ensure_ascii=False, indent=2)
    if isinstance(data, list):
        print(f"  → Saved {len(data)} items → {path.name}")
    elif isinstance(data, dict):
        keys_summary = ", ".join(f"{k}={len(v) if isinstance(v, (list, dict)) else v}"
                                 for k, v in list(data.items())[:5])
        print(f"  → Saved → {path.name}  ({keys_summary})")


def main():
    field_name = "BEV Perception"
    max_papers = 20

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"output/v2_test_{ts}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {out_dir}")

    # ── Build clients ──
    client = OpenAI(api_key=config.LLM_API_KEY, base_url=config.LLM_BASE_URL)
    model = config.LLM_ANALYZER_MODEL
    emb_client = build_embedding_client()
    print(f"LLM: {model} @ {config.LLM_BASE_URL}")
    print(f"Embedding: {'API' if emb_client else 'local fallback'}")

    # ═══════════════════════════════════════════════════════════════════
    # Step 1.1: Query Expansion
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 1.1: Query Expansion")
    print(f"{'='*60}")
    query_expansion = _generate_query_expansion(field_name, client, model=model)
    _save_step(out_dir, "step_1.1_query_expansion", query_expansion)

    # ═══════════════════════════════════════════════════════════════════
    # Step 1.2: Multi-source Broad Recall
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 1.2: Multi-source Broad Recall")
    print(f"{'='*60}")
    candidates = _broad_recall(query_expansion, field_name)
    _save_step(out_dir, "step_1.2_broad_recall_raw", candidates)

    if not candidates:
        print("ERROR: No candidates found. Exiting.")
        return

    # Dedup
    candidates = _deduplicate(candidates)
    print(f"Candidates after dedup: {len(candidates)}")
    _save_step(out_dir, "step_1.2_broad_recall_deduped", candidates)

    # ═══════════════════════════════════════════════════════════════════
    # Step 1.3: Citation Expansion
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 1.3: Citation Expansion")
    print(f"{'='*60}")

    # First enrich with citation counts
    candidates = _openalex_enrich(candidates, field_name)
    n_enriched = sum(1 for c in candidates if c.get("citation_count", 0) > 0)
    print(f"Enriched: {n_enriched}/{len(candidates)} have citation data")

    # Extract disambiguation terms
    disambiguation = query_expansion.get("disambiguation", {})
    core_field_markers = disambiguation.get("core_field_markers", [])
    field_specific_terms = disambiguation.get("field_specific_terms", [field_name])
    exclusion_terms = disambiguation.get("exclusion_terms", [])
    if not core_field_markers:
        core_field_markers = list(field_specific_terms)
    for variants in query_expansion.get("synonyms_and_variants", {}).values():
        field_specific_terms.extend(variants)
    print(f"  Disambiguation: {len(core_field_markers)} core markers, {len(field_specific_terms)} field terms, {len(exclusion_terms)} exclusion terms")

    snowballed = _citation_expand(
        candidates, field_name,
        core_field_markers=core_field_markers,
        exclusion_terms=exclusion_terms,
    )
    _save_step(out_dir, "step_1.3_citation_expansion_raw", snowballed)

    if snowballed:
        candidates = candidates + snowballed
        candidates = _deduplicate(candidates)
        print(f"After citation expansion: {len(candidates)} candidates")

    _save_step(out_dir, "step_1.3_after_citation_expansion", candidates)

    # ═══════════════════════════════════════════════════════════════════
    # Step 1.4: Survey Calibration — DISABLED in V2.2
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 1.4: Survey Calibration — DISABLED (V2.2)")
    print(f"{'='*60}")
    print("  Skipped — OpenAlex survey search returns irrelevant papers for ambiguous field names.")
    _save_step(out_dir, "step_1.4_survey_papers", [])
    _save_step(out_dir, "step_1.4_full_candidate_pool", candidates)

    # ═══════════════════════════════════════════════════════════════════
    # Step 1.2.5: Relevance-based Pre-rank
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 1.2.5: Relevance-based Pre-rank")
    print(f"{'='*60}")
    candidates = _pre_rank(candidates, core_field_markers, exclusion_terms)
    llm_candidates = candidates[:_PRE_RANK_TOP_N]
    _save_step(out_dir, "step_1.2.5_pre_rank_top150", [
        {"title": c["title"], "year": c.get("year", 0), "citation_count": c.get("citation_count", 0),
         "relevance_score": c.get("relevance_score", 0), "source": c.get("source", "unknown")}
        for c in llm_candidates
    ])

    # ═══════════════════════════════════════════════════════════════════
    # Step 1.5: Milestone Description + Milestone-guided Search + Embedding Matching
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 1.5: Milestone Descriptions + Milestone-guided Search + Embedding Matching")
    print(f"{'='*60}")

    milestones = _generate_milestone_descriptions(field_name, client, model=model)
    _save_step(out_dir, "step_1.5_milestone_descriptions", milestones)

    # V2.2: Milestone-guided search — find papers by known title keywords
    pool_titles = {c["title"].lower().strip() for c in candidates}
    mg_papers = _milestone_guided_search(milestones, pool_titles, emb_client=emb_client)
    _save_step(out_dir, "step_1.5_milestone_guided_search", mg_papers)

    if mg_papers:
        candidates = candidates + mg_papers
        candidates = _deduplicate(candidates)
        candidates = _pre_rank(candidates, core_field_markers, exclusion_terms)
        llm_candidates = candidates[:_PRE_RANK_TOP_N]
        _save_step(out_dir, "step_1.2.5_pre_rank_top150", [
            {"title": c["title"], "year": c.get("year", 0), "citation_count": c.get("citation_count", 0),
             "relevance_score": c.get("relevance_score", 0), "source": c.get("source", "unknown")}
            for c in llm_candidates
        ])
        print(f"After milestone-guided search: {len(candidates)} candidates, top-{len(llm_candidates)} for LLM")

    # ── V2.3.1: Promote milestone-discovered papers to top-150 ──
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

    # Save embedding match results
    embedding_results = {
        "matched": {str(k): v for k, v in matched_map.items()},
        "ambiguous": [
            {"ms_idx": ms_idx, "candidate_title": cand["title"],
             "similarity": float(sim), "candidate_year": cand.get("year", 0)}
            for ms_idx, cand, sim in ambiguous
        ],
        "unmatched_indices": unmatched_indices,
        "unmatched_details": [
            {"index": i, "description": milestones[i].get("description", ""),
             "keywords": milestones[i].get("known_title_keywords", "")}
            for i in unmatched_indices
        ],
    }
    _save_step(out_dir, "step_1.5_embedding_match", embedding_results)

    # Mark high-confidence matches
    matched_candidate_titles = set()
    for titles in matched_map.values():
        for t in titles:
            matched_candidate_titles.add(t.lower().strip())

    for c in candidates:
        if c["title"].lower().strip() in matched_candidate_titles:
            c["is_seminal"] = True

    n_seminal = sum(1 for c in candidates if c.get("is_seminal"))
    print(f"Seminal-marked candidates: {n_seminal}")

    # Also save candidate summaries (title, year, citation, is_seminal) for quick review
    candidate_summaries = [
        {
            "title": c["title"],
            "year": c.get("year", 0),
            "citation_count": c.get("citation_count", 0),
            "is_seminal": c.get("is_seminal", False),
            "source": c.get("source", "unknown"),
        }
        for c in sorted(candidates, key=lambda c: (-c.get("citation_count", 0), -c.get("year", 0)))
    ]
    _save_step(out_dir, "step_1.5_candidate_summaries", candidate_summaries)

    # ═══════════════════════════════════════════════════════════════════
    # Step 1.6: LLM Unified Selection
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 1.6: LLM Unified Selection")
    print(f"{'='*60}")

    selected, rejected, missing_from_llm = _llm_unified_select(
        llm_candidates, field_name, ambiguous, client,
        max_papers=max_papers, model=model,
    )

    # V2.2: Embedding-based fallback if LLM used citation stratification
    if all(c.get("classification", "") == "unknown" for c in selected):
        print("  LLM selection used citation fallback, switching to embedding selection")
        selected = _embedding_select(milestones, llm_candidates, emb_client, max_papers)
        rejected = []
        missing_from_llm = []
        for c in selected:
            c["_fallback"] = "embedding"

    _save_step(out_dir, "step_1.6_selected", selected)
    _save_step(out_dir, "step_1.6_rejected", rejected)
    _save_step(out_dir, "step_1.6_missing_from_llm", missing_from_llm)

    # ═══════════════════════════════════════════════════════════════════
    # Step 1.7: Closed-loop Recovery
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Step 1.7: Closed-loop Recovery")
    print(f"{'='*60}")

    unmatched_milestones = [milestones[i] for i in unmatched_indices]
    recovered, confirmed_missing = _closed_loop_recover(
        missing_from_llm, unmatched_milestones, selected, field_name,
    )
    _save_step(out_dir, "step_1.7_recovered", recovered)
    _save_step(out_dir, "step_1.7_confirmed_missing", confirmed_missing)

    # ═══════════════════════════════════════════════════════════════════
    # Final
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print("Final Summary")
    print(f"{'='*60}")

    # Merge recovered into selection
    if recovered:
        for r in recovered:
            if len(selected) < max_papers:
                selected.append(r)

    # Save final output
    final_output = {
        "field_name": field_name,
        "total_candidates": len(candidates),
        "selected_count": len(selected),
        "recovered_count": len(recovered),
        "confirmed_missing_count": len(confirmed_missing),
        "selected_papers": [
            {
                "title": p["title"],
                "year": p.get("year", 0),
                "citation_count": p.get("citation_count", 0),
                "classification": p.get("classification", "unknown"),
                "rationale": p.get("rationale", ""),
                "source": p.get("source", "unknown"),
            }
            for p in selected
        ],
        "confirmed_missing": confirmed_missing,
    }
    _save_step(out_dir, "final_summary", final_output)

    # Print concise result
    print(f"\nSelected ({len(selected)}):")
    for i, p in enumerate(selected):
        cls = p.get("classification", "?")
        year = p.get("year", "?")
        cit = p.get("citation_count", 0)
        print(f"  {i+1:2d}. [{cls:12s}] ({year}) [{cit:4d} cit] {p['title'][:90]}")

    if confirmed_missing:
        print(f"\nCONFIRMED MISSING ({len(confirmed_missing)}):")
        for cm in confirmed_missing:
            info = cm.get("description", "") or cm.get("known_title_keywords", "")
            print(f"  ✗ {info[:100]} ({cm.get('first_author','?')}, {cm.get('year','?')})")

    print(f"\nAll results saved to: {out_dir}")


if __name__ == "__main__":
    main()
