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


def _paper_label(title: str) -> str:
    """Extract a short readable label from a paper title."""
    if ":" in title:
        prefix = title.split(":")[0].strip()
        if len(prefix) < 50:
            return prefix
    return title[:45] + "..." if len(title) > 45 else title


def _render_markdown(narrative, all_claims, tensions=None, paradigm_shifts=None) -> str:
    """Render narrative to markdown.

    Design:
    - Claims in compact table format (multi-row per paper, no repetition)
    - Evolution shown as Mermaid graph (primary), text tree as collapsible fallback
    - Mermaid kept as collapsible alternative
    - Paradigm shifts grouped by dimension
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

    # ── Pass 1: All phase narratives ──────────────────────────────────
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

    # ── Paradigm shifts (right after narratives, before paper details) ──
    if paradigm_shifts:
        lines.extend([
            "---",
            "",
            shifts_to_markdown(paradigm_shifts),
        ])

    # ── Pass 2: All branch details (graph + path + claims) ────────────
    for i, branch in enumerate(narrative.branches, 1):
        lines.extend([
            f"### {branch.name} — 演化细节",
            "",
        ])

        relations = branch.claim_relations or []
        if relations:
            paper_labels: dict[str, str] = {}
            for r in relations:
                for paper in [r["source_paper"], r["target_paper"]]:
                    if paper not in paper_labels:
                        paper_labels[paper] = _paper_label(paper)

            # Build edge map for annotated chain
            edges: dict[str, tuple[str, str, str]] = {}
            for r in relations:
                edges[r["source_paper"]] = (r["target_paper"], r["relation"], r["explanation"])

            # Collect paper order and years
            papers_in_order: list[str] = []
            seen = set()
            for r in relations:
                for p in [r["source_paper"], r["target_paper"]]:
                    if p not in seen:
                        papers_in_order.append(p)
                        seen.add(p)

            paper_years: dict[str, str] = {}
            for c in branch.claims:
                if c.paper_title not in paper_years:
                    paper_years[c.paper_title] = str(c.year)

            # ── Mermaid graph ──
            lines.extend([
                "### 思想演化图",
                "",
                "```mermaid",
                "flowchart LR",
            ])
            node_ids = {}
            node_idx = 0
            for r in relations:
                for paper in [r["source_paper"], r["target_paper"]]:
                    if paper not in node_ids:
                        nid = f"P{node_idx}"
                        node_ids[paper] = nid
                        node_idx += 1
                        label = paper_labels[paper].replace('"', "'")
                        lines.append(f'    {nid}["{label}"]')

            for r in relations:
                src_id = node_ids.get(r["source_paper"], "")
                tgt_id = node_ids.get(r["target_paper"], "")
                if src_id and tgt_id:
                    rel_map = {
                        "improve": " -->|IMPROVE| ",
                        "extend": " -->|EXTEND| ",
                        "support": " -->|SUPPORT| ",
                        "replace": " ==>|REPLACE| ",
                        "attack": " ==>|ATTACK| ",
                        "parallel": " -.->|PARALLEL| ",
                    }
                    edge = rel_map.get(r["relation"], f" -->|{r['relation'].upper()}| ")
                    lines.append(f"    {src_id}{edge}{tgt_id}")

            lines.extend([
                "```",
                "",
                "*实线=因果演化 · 粗线=范式替代 · 虚线=并行发展*",
                "",
            ])

            # ── Annotated evolution path ──
            rel_icons = {
                "improve": "🔧 IMPROVE",
                "extend": "➡️ EXTEND",
                "support": "✅ SUPPORT",
                "replace": "🔄 REPLACE",
                "attack": "⚔️ ATTACK",
                "parallel": "∥ PARALLEL",
            }

            lines.extend([
                "### 演化路径",
                "",
            ])

            for paper in papers_in_order:
                label = paper_labels.get(paper, paper[:50])
                year = paper_years.get(paper, "")
                year_str = f" ({year})" if year else ""
                lines.append(f"**{label}**{year_str}")
                lines.append("")

                if paper in edges:
                    tgt, rel, expl = edges[paper]
                    icon = rel_icons.get(rel, rel.upper())
                    tgt_label = paper_labels.get(tgt, tgt[:50])
                    if rel == "parallel":
                        lines.append(f"> ∥ **PARALLEL** → {tgt_label}")
                        lines.append(f"> 不同研究线 — {expl}")
                    else:
                        lines.append(f"> {icon} → **{tgt_label}**")
                        lines.append(f"> {expl}")
                    lines.append("")

        # ── Claims Table ──
        claims_by_paper: dict[str, list[Claim]] = {}
        paper_meta: dict[str, tuple[str, int]] = {}
        for claim in branch.claims:
            pid = claim.paper_id
            if pid not in claims_by_paper:
                claims_by_paper[pid] = []
                paper_meta[pid] = (claim.paper_title, claim.year)
            claims_by_paper[pid].append(claim)

        sorted_pids = sorted(claims_by_paper.keys(),
                            key=lambda pid: paper_meta[pid][1])

        lines.extend([
            "### 关键论文与核心主张",
            "",
            "| 论文 | 年份 | 主张 | 证据 |",
            "|------|------|------|------|",
        ])

        for pid in sorted_pids:
            title, year = paper_meta[pid]
            p_claims = claims_by_paper[pid]
            short_title = _paper_label(title)

            for j, c in enumerate(p_claims):
                paper_cell = f"**{short_title}**" if j == 0 else ""
                year_cell = str(year) if j == 0 else ""
                claim_text = f"{j + 1}. {c.statement}"
                evidence = c.evidence
                lines.append(
                    f"| {paper_cell} | {year_cell} | {claim_text} | {evidence} |"
                )

        lines.append("")

    # ── Research tensions ──
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

    # ── Synthesis ──
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
