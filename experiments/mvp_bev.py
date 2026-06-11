"""MVP experiment: BEV perception evolution narrative from 12 key papers.

Usage:
    cd /home/pnc/ws/paper_context_tool
    python experiments/mvp_bev.py

Flow:
    1. Resolve 12 papers (arXiv metadata + abstract)
    2. Extract claims from each paper
    3. Propose phase structure (LLM discovers paradigm boundaries)
    4. Build claim relations (within-phase + cross-phase paradigm)
    5. Detect research tensions
    6. Detect paradigm shifts (multi-level, within and across phases)
    7. Generate field-centric narrative (with relations, tensions, shifts)
    8. Output markdown report
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# Ensure src/ is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from paper import Paper, Claim
from claim_extractor import extract_claims
from narrative_builder import build_narrative_from_claims, propose_structure
from claim_relation_builder import build_paper_chain_relations, classify_paradigm_relation
from tension_detector import detect_tensions, tensions_to_markdown
from paradigm_shift_detector import detect_paradigm_shifts, shifts_to_markdown
from llm_analyzer import build_analyzer_client
from paper_resolver import _fetch_arxiv_metadata, _extract_arxiv_id

# ── 12 MVP papers ─────────────────────────────────────────────────────
MVP_PAPERS = [
    {"arxiv_id": "2008.05711", "title": "LSS"},
    {"arxiv_id": "2112.11790", "title": "BEVDet"},
    {"arxiv_id": "2206.10092", "title": "BEVDepth"},
    {"arxiv_id": "2203.17054", "title": "BEVDet4D"},
    {"arxiv_id": "2203.17270", "title": "BEVFormer"},
    {"arxiv_id": "2211.10439", "title": "BEVFormerV2"},
    {"arxiv_id": "2308.09244", "title": "SparseBEV"},
    {"arxiv_id": "2211.10581", "title": "Sparse4D"},
    {"arxiv_id": "2305.14018", "title": "Sparse4Dv2"},
    {"arxiv_id": "2212.10156", "title": "UniAD"},
    {"arxiv_id": "2303.12077", "title": "VAD"},
    {"arxiv_id": "2405.19620", "title": "SparseDrive"},
]

def main():
    print("=" * 60)
    print("  MVP: BEV Perception Evolution Narrative")
    print("=" * 60)

    # ── Build LLM client ──
    print("\n[1/4] Building LLM client...")
    client = build_analyzer_client()
    if not client:
        print("ERROR: Cannot build LLM client. Check LLM_API_KEY / LLM_BASE_URL.")
        sys.exit(1)
    print("  OK")

    # ── Phase 1: Resolve papers ──
    print(f"\n[2/4] Resolving {len(MVP_PAPERS)} papers...")
    papers: list[Paper] = []
    for i, mp in enumerate(MVP_PAPERS):
        arxiv_id = mp["arxiv_id"]
        arxiv_meta = _fetch_arxiv_metadata(arxiv_id)
        if arxiv_meta:
            paper = Paper(
                id=f"arxiv:{arxiv_id}",
                arxiv_id=arxiv_id,
                title=arxiv_meta.get("title", mp["title"]),
                authors=arxiv_meta.get("authors", []),
                year=arxiv_meta.get("year", 0),
                abstract=arxiv_meta.get("abstract", ""),
                url=arxiv_meta.get("url", f"https://arxiv.org/abs/{arxiv_id}"),
                source="arxiv",
            )
        else:
            # Fallback: metadata from our table
            paper = Paper(
                id=f"arxiv:{arxiv_id}",
                arxiv_id=arxiv_id,
                title=mp["title"],
                abstract="",
                source="arxiv",
            )
        papers.append(paper)
        status = "OK" if paper.abstract else "NO ABSTRACT"
        print(f"  [{i + 1}/{len(MVP_PAPERS)}] {paper.title[:50]} — {status}")

    # ── Phase 2: Extract claims (flat pool, no manual branches) ──
    print(f"\n[3/5] Extracting claims from {len(papers)} papers...")
    all_claims: list[Claim] = []
    for i, paper in enumerate(papers):
        claims = extract_claims(paper, client=client)
        all_claims.extend(claims)
        print(f"  [{i + 1}/{len(papers)}] {paper.title[:50]} — {len(claims)} claims")

    total_claims = len(all_claims)
    print(f"\n  Total: {total_claims} claims")

    # ── Phase 3: Propose phase structure (LLM discovers paradigm shifts) ──
    print(f"\n[4/5] Proposing phase structure from claims...")
    phases = propose_structure(all_claims, field_name="BEV Perception", client=client)

    if not phases:
        print("  WARNING: Structure proposal failed, using single phase fallback")
        phases = [{
            "name": "BEV Perception",
            "time_range": "2020-2024",
            "problem_statement": "How to build unified BEV perception from multi-camera images?",
            "core_paradigm": "Camera-only BEV perception",
            "paper_arxiv_ids": [mp["arxiv_id"] for mp in MVP_PAPERS],
            "key_evolution": "Evolution of BEV perception methods from 2020 to 2024",
        }]

    print(f"\n  LLM proposed {len(phases)} phase(s):")
    for p in phases:
        papers_in_phase = [mp["title"] for mp in MVP_PAPERS
                          if mp["arxiv_id"] in p.get("paper_arxiv_ids", [])]
        print(f"    {p['name']} ({p.get('time_range', '?')})")
        print(f"      Paradigm: {p.get('core_paradigm', '?')}")
        print(f"      Papers ({len(papers_in_phase)}): {', '.join(papers_in_phase)}")
        print()

    # Group claims into phases based on LLM proposal
    claims_by_phase: dict[str, list[Claim]] = {}
    problem_statements: dict[str, str] = {}
    for phase in phases:
        name = phase["name"]
        arxiv_ids = phase.get("paper_arxiv_ids", [])
        phase_claims = [c for c in all_claims
                        if any(aid in c.paper_id for aid in arxiv_ids)]
        claims_by_phase[name] = phase_claims
        problem_statements[name] = phase.get("problem_statement", "")
        problem_statements[f"{name}__time_range"] = phase.get("time_range", "")
        problem_statements[f"{name}__core_paradigm"] = phase.get("core_paradigm", "")

    # ── Phase 3.5: Build claim relations within and across phases ──
    print(f"\n[4.5] Building claim relations...")
    all_phase_relations: dict[str, list[dict]] = {}

    # Phase order list for cross-phase paradigm comparison
    phase_names = list(claims_by_phase.keys())
    prev_phase_paradigm: str = ""
    prev_phase_name: str = ""

    for phase_name in phase_names:
        phase_claims = claims_by_phase[phase_name]

        # Group claims by paper within this phase
        claims_by_paper: dict[str, list[Claim]] = {}
        for c in phase_claims:
            claims_by_paper.setdefault(c.paper_id, []).append(c)

        # Sort papers by year
        sorted_papers = sorted(claims_by_paper.items(), key=lambda x: x[1][0].year)

        # Within-phase relations: consecutive paper pairs
        relations = build_paper_chain_relations(claims_by_paper, client=client)

        # Cross-phase paradigm relation: compare core paradigms
        if prev_phase_paradigm and phase_name in problem_statements:
            paradigm_result = classify_paradigm_relation(
                earlier_paradigm=prev_phase_paradigm,
                later_paradigm=problem_statements.get(f"{phase_name}__core_paradigm", ""),
                earlier_phase_name=prev_phase_name,
                later_phase_name=phase_name,
                client=client,
            )
            if paradigm_result:
                relations.insert(0, {
                    "source_paper": f"[Paradigm] {prev_phase_name}",
                    "target_paper": f"[Paradigm] {phase_name}",
                    "source_claim": prev_phase_paradigm,
                    "target_claim": problem_statements.get(f"{phase_name}__core_paradigm", ""),
                    "source_year": 0,
                    "target_year": 0,
                    "relation": paradigm_result["relation"],
                    "explanation": f"[PARADIGM SHIFT] {paradigm_result['explanation']}",
                })

        all_phase_relations[phase_name] = relations

        # Remember this phase's paradigm for next cross-phase comparison
        prev_phase_paradigm = problem_statements.get(f"{phase_name}__core_paradigm", "")
        prev_phase_name = phase_name

        attack_count = sum(1 for r in relations if r["relation"] in ("attack", "replace"))
        print(f"  {phase_name}: {len(relations)} relations "
              f"({attack_count} attack/replace)")

    # ── Phase 3.7: Detect research tensions ──
    print(f"\n[4.7] Detecting research tensions...")
    tensions = detect_tensions(
        claims=all_claims,
        relations=[r for rels in all_phase_relations.values() for r in rels],
        phases=phases,
        field_name="BEV Perception",
        client=client,
    )
    if tensions:
        print(f"  Detected {len(tensions)} tensions:")
        for t in tensions:
            print(f"    [{t.get('status', '?')}] {t['tension']}")
    else:
        print("  WARNING: Tension detection failed")

    # ── Phase 3.8: Detect paradigm shifts ──
    print(f"\n[4.8] Detecting paradigm shifts...")
    all_relations_flat = [r for rels in all_phase_relations.values() for r in rels]
    paradigm_shifts = detect_paradigm_shifts(
        claims=all_claims,
        relations=all_relations_flat,
        phases=phases,
        tensions=tensions or [],
        field_name="BEV Perception",
        client=client,
    )
    if paradigm_shifts:
        print(f"  Detected {len(paradigm_shifts)} paradigm shifts:")
        for s in paradigm_shifts:
            print(f"    [{s.get('magnitude', '?').upper()}] {s['shift_name']} "
                  f"({s.get('level', '?')})")
    else:
        print("  WARNING: Paradigm shift detection failed")

    # ── Phase 4: Generate field-centric narrative ──
    print(f"\n[5/6] Generating field-centric narrative...")
    narrative = build_narrative_from_claims(
        claims_by_branch=claims_by_phase,
        problem_statements=problem_statements,
        field_name="BEV Perception",
        client=client,
        claim_relations=all_phase_relations,
        tensions=tensions or [],
        paradigm_shifts=paradigm_shifts or [],
    )

    if not narrative:
        print("ERROR: Narrative generation failed.")
        sys.exit(1)

    # ── Export ──
    output_dir = Path("output/v4_mvp")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full data
    data_path = output_dir / "narrative.json"
    data_path.write_text(
        json.dumps(narrative.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Generate markdown
    md_path = output_dir / "bev_evolution.md"
    md_text = _render_markdown(narrative, all_claims, tensions, paradigm_shifts)
    md_path.write_text(md_text, encoding="utf-8")
    print(f"\nMarkdown report: {md_path}")
    print(f"JSON data: {data_path}")

    # Quick preview
    print("\n" + "=" * 60)
    print("  PREVIEW")
    print("=" * 60)
    print(f"\n{narrative.overview[:500]}...")
    for b in narrative.branches:
        print(f"\n--- {b.name} ---")
        print(b.narrative[:300])
        print("...")
    print(f"\n{narrative.synthesis[:300]}...")


def _render_markdown(narrative, all_claims, tensions=None, paradigm_shifts=None) -> str:
    """Render narrative to markdown with idea-centric organization.

    Key design decisions:
    - Papers appear ONCE each in the claims table (primary claim only)
    - Claim relations shown as Mermaid graph (not table)
    - Paradigm shifts grouped by dimension (independent evolution threads)
    """
    lines = [
        f"# {narrative.field_name} — 技术发展叙事",
        "",
        "## 1. 领域全景",
        "",
        narrative.overview,
        "",
        "---",
    ]

    for i, branch in enumerate(narrative.branches, 1):
        header = f"## {i + 1}. {branch.name}"
        if branch.time_range:
            header += f" ({branch.time_range})"
        lines.append(header)
        lines.append("")
        if branch.core_paradigm:
            lines.append(f"**核心范式**: {branch.core_paradigm}")
        lines.extend([
            f"**核心问题**: {branch.problem_statement}",
            "",
            "### 技术发展故事",
            "",
            branch.narrative,
            "",
        ])

        # ── Idea Evolution Graph (Mermaid) ──
        relations = branch.claim_relations or []
        if relations:
            lines.extend([
                "### 思想演化图",
                "",
                "```mermaid",
                "graph TD",
            ])
            # Assign node IDs and build graph
            node_ids = {}
            node_idx = 0
            for r in relations:
                src = r["source_paper"]
                tgt = r["target_paper"]
                for paper in [src, tgt]:
                    if paper not in node_ids:
                        node_ids[paper] = f"P{node_idx}"
                        node_idx += 1
                        # Truncate label for readability
                        label = paper[:50] + ("..." if len(paper) > 50 else "")
                        year = r.get("source_year", "") if paper == src else r.get("target_year", "")
                        lines.append(f'    {node_ids[paper]}["{label}"]')

            # Add edges with relation labels
            rel_styles = {
                "attack": " ==>|⚔️ ATTACK| ",
                "replace": " ==>|🔄 REPLACE| ",
                "improve": " -->|🔧 IMPROVE| ",
                "support": " -->|✅ SUPPORT| ",
                "extend": " -->|➡️ EXTEND| ",
                "parallel": " -.->|∥ PARALLEL| ",
                "unknown": " -->|?| ",
            }
            for r in relations:
                src_id = node_ids.get(r["source_paper"], "")
                tgt_id = node_ids.get(r["target_paper"], "")
                if src_id and tgt_id:
                    edge = rel_styles.get(r["relation"], f" -->|{r['relation'].upper()}| ")
                    lines.append(f"    {src_id}{edge}{tgt_id}")

            lines.append("```")
            lines.append("")
            lines.append("> **图例**: 实线=因果关联 · 粗线=范式替代 · 虚线=并行发展(不同研究线)")
            lines.append("")

        # ── Papers Table (one row per paper, primary claim only) ──
        # Deduplicate by paper_id, keep first claim as primary
        seen_papers: dict[str, Claim] = {}
        for claim in branch.claims:
            if claim.paper_id not in seen_papers:
                seen_papers[claim.paper_id] = claim

        sorted_papers = sorted(seen_papers.values(), key=lambda c: c.year)

        lines.extend([
            "### 关键论文与核心主张",
            "",
            "| 论文 | 年份 | 核心主张 | 证据 |",
            "|------|------|---------|------|",
        ])
        for c in sorted_papers:
            evidence = c.evidence[:100] + "..." if len(c.evidence) > 100 else c.evidence
            title = c.paper_title[:60] + ("..." if len(c.paper_title) > 60 else "")
            lines.append(
                f"| {title} | {c.year} | "
                f"{c.statement[:100]} | {evidence} |"
            )
        lines.append("")

    # Paradigm shifts section (dimension-grouped from P0)
    if paradigm_shifts:
        lines.extend([
            "---",
            "",
            shifts_to_markdown(paradigm_shifts),
        ])

    # Research tensions section
    if tensions:
        lines.extend([
            "---",
            "",
            "## 核心研究张力",
            "",
            "> 技术发展史的真正主角不是论文，而是矛盾。以下是驱动该领域演化的核心张力。",
            "",
            tensions_to_markdown(tensions),
        ])

    lines.extend([
        "---",
        "",
        "## 领域趋势与展望",
        "",
        narrative.synthesis,
        "",
        "---",
        "",
        f"*Generated by Research Narrative Engine V4 (MVP)*",
    ])

    return "\n".join(lines)


if __name__ == "__main__":
    main()
