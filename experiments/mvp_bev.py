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

from paper import Paper, Claim, Phase
from claim_extractor import extract_claims
from narrative_builder import build_narrative
from claim_relation_builder import build_paper_chain_relations
from tension_detector import (
    detect_all_tensions, merge_tensions_into_phases,
    tensions_to_markdown, phases_to_markdown,
)
from paradigm_shift_detector import detect_paradigm_shifts, shifts_to_markdown
from research_question_detector import detect_research_questions
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

    # ── Phase 2: Extract claims ──
    print(f"\n[3/7] Extracting claims from {len(papers)} papers...")
    all_claims: list[Claim] = []
    for i, paper in enumerate(papers):
        claims = extract_claims(paper, client=client)
        all_claims.extend(claims)
        print(f"  [{i + 1}/{len(papers)}] {paper.title[:50]} — {len(claims)} claims")

    total_claims = len(all_claims)
    print(f"\n  Total: {total_claims} claims")

    # ── Phase 3: Build claim relations (V6: still needed for edges) ──
    print(f"\n[4/7] Building claim relations...")

    # Sort claims by year, then build relations between consecutive papers
    claims_sorted = sorted(all_claims, key=lambda c: c.year)

    # Group claims by paper for chain building
    claims_by_paper: dict[str, list[Claim]] = {}
    for c in claims_sorted:
        claims_by_paper.setdefault(c.paper_id, []).append(c)

    all_relations = build_paper_chain_relations(claims_by_paper, client=client)

    attack_count = sum(1 for r in all_relations
                       if (hasattr(r, 'relation') and r.relation in ("attack", "replace"))
                       or (isinstance(r, dict) and r.get("relation") in ("attack", "replace")))
    print(f"  {len(all_relations)} relations ({attack_count} attack/replace)")

    # ── Phase 3.5: Detect research questions (V7: RQ->Tension->Direction nested) ──
    print(f"\n[4.5] Detecting research questions (with nested tensions + direction)...")
    research_questions = detect_research_questions(
        claims=all_claims,
        field_name="BEV Perception",
        client=client,
    )
    if research_questions:
        print(f"  Detected {len(research_questions)} research questions:")
        for rq in research_questions:
            t_count = len(rq.tensions) if rq.tensions else 0
            d_info = ""
            if rq.direction:
                d_info = f" | direction: {rq.direction.confidence}"
            print(f"    [{rq.level.upper()}] {rq.question} — {t_count} tensions{d_info}")

        # Collect tensions from all RQs
        rq_tensions = []
        for rq in research_questions:
            if rq.tensions:
                rq_tensions.extend(rq.tensions)
        print(f"  Total RQ-nested tensions: {len(rq_tensions)}")
    else:
        print("  WARNING: RQ detection failed")
        rq_tensions = []
        research_questions = []

    # ── Phase 3.6: Two-stage tension -> phase detection (V8) ──
    print(f"\n[4.6] Stage 1: Detecting ALL tensions (no limit)...")
    all_tensions = detect_all_tensions(
        claims=all_claims,
        relations=all_relations,
        field_name="BEV Perception",
        client=client,
    )
    if all_tensions:
        print(f"  Detected {len(all_tensions)} tensions (Stage 1, fine-grained):")
        for t in all_tensions:
            print(f"    [{t.dimension}] {t.tension} — {t.status}")
    else:
        print("  WARNING: Stage 1 tension detection failed, using RQ-nested tensions")
        all_tensions = rq_tensions or []

    print(f"\n[4.7] Stage 2: Merging tensions into phases...")
    phases = None
    if all_tensions:
        phases = merge_tensions_into_phases(
            tensions=all_tensions,
            claims=all_claims,
            field_name="BEV Perception",
            client=client,
        )
    if phases:
        print(f"  Merged into {len(phases)} phases:")
        for i, p in enumerate(phases):
            print(f"    Phase {i + 1}: {p.name} ({p.time_range})")
            print(f"      Core contradiction: {p.core_contradiction[:100]}...")
            print(f"      Unresolved: {p.unresolved_problem[:100]}...")
            print(f"      Papers: {', '.join(p.key_papers[:4])}")
    else:
        print("  WARNING: Phase merging failed, narrative will fall back to RQ-based")

    # ── Phase 3.7: Detect paradigm shifts ──
    print(f"\n[4.8] Detecting paradigm shifts...")
    paradigm_shifts = detect_paradigm_shifts(
        claims=all_claims,
        relations=all_relations,
        phases=[{"name": "BEV Perception", "time_range": "2020-2024",
                 "core_paradigm": "Camera-only BEV perception",
                 "paper_arxiv_ids": [mp["arxiv_id"] for mp in MVP_PAPERS]}],
        tensions=all_tensions or [],
        field_name="BEV Perception",
        client=client,
    )
    if paradigm_shifts:
        print(f"  Detected {len(paradigm_shifts)} paradigm shifts:")
        for s in paradigm_shifts:
            mag = s.magnitude if hasattr(s, 'magnitude') else s.get('magnitude', '?')
            name = s.shift_name if hasattr(s, 'shift_name') else s['shift_name']
            lvl = s.level if hasattr(s, 'level') else s.get('level', '?')
            print(f"    [{mag.upper()}] {name} ({lvl})")
    else:
        print("  WARNING: Paradigm shift detection failed")

    # ── Phase 4: Generate Phase-centric narrative (V8) ──
    print(f"\n[5/7] Generating phase-centric narrative...")
    narrative = build_narrative(
        claims=all_claims,
        claim_relations=all_relations,
        research_questions=research_questions,
        tensions=all_tensions,
        paradigm_shifts=paradigm_shifts or [],
        phases=phases,
        field_name="BEV Perception",
        client=client,
    )

    if not narrative:
        print("ERROR: Narrative generation failed.")
        sys.exit(1)

    # ── Export ──
    # Auto-increment run number so each run gets a unique directory
    output_root = Path("output")
    existing = sorted(output_root.glob("v8_mvp_*"))
    if existing:
        last_num = int(existing[-1].name.rsplit("_", 1)[-1])
        run_num = last_num + 1
    else:
        run_num = 1
    output_dir = output_root / f"v8_mvp_{run_num:03d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full data
    data_path = output_dir / "narrative.json"
    data_path.write_text(
        json.dumps(narrative.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    # Generate markdown
    md_path = output_dir / "bev_evolution.md"
    md_text = _render_markdown(narrative, all_claims, all_tensions, paradigm_shifts, phases)
    md_path.write_text(md_text, encoding="utf-8")
    print(f"\nMarkdown report: {md_path}")
    print(f"JSON data: {data_path}")

    # Quick preview
    print("\n" + "=" * 60)
    print("  PREVIEW")
    print("=" * 60)
    print(f"\n{narrative.overview[:500]}...")
    for s in narrative.sections:
        print(f"\n--- {s.title} ---")
        print(s.narrative[:300])
        print("...")
    print(f"\n{narrative.synthesis[:300]}...")


def _paper_label(title: str) -> str:
    """Extract a short readable label from a paper title."""
    if ":" in title:
        prefix = title.split(":")[0].strip()
        if len(prefix) < 50:
            return prefix
    return title[:45] + "..." if len(title) > 45 else title


def _render_markdown(narrative, all_claims, tensions=None, paradigm_shifts=None, phases=None) -> str:
    """Render narrative to markdown — V8.1: idea-centric, compact output.

    Structure (§2 Overview → §3 Paradigm Shifts → §4 Phase Evolution
    → §5 Open Questions → §6 Reading List).

    Removed vs V8: Direction blocks, Evolution path annotations, Tension table,
    Paradigm Mermaid, Evidence-Backed/Speculative split.
    """
    lines = [
        f"# {narrative.field_name} — 技术发展叙事",
        "",
        "## 1. 领域全景",
        "",
        narrative.overview,
        "",
    ]

    # ── §2 Paradigm Shifts (0-5 compact bullets) ──────────────────────────
    shifts = paradigm_shifts or []
    if shifts:
        lines.extend([
            "---",
            "",
            "## 2. 范式转移",
            "",
            shifts_to_markdown(shifts),
        ])

    # ── §3 Phase Evolution ──────────────────────────────────────────────
    lines.extend([
        "---",
        "",
        "## 3. 阶段演化",
        "",
    ])

    for i, section in enumerate(narrative.sections, 1):
        phase_name = section.title
        lines.append(f"### 3.{i} {phase_name}")
        lines.append("")

        # Phase metadata (compact blockquote)
        if hasattr(section, 'phase') and section.phase:
            p = section.phase
            lines.append(f"> **核心矛盾**: {p.core_contradiction}")
            lines.append(f"> **核心辩论**: {p.core_debate}")
            if i > 1 and hasattr(narrative.sections[i - 2], 'phase') and narrative.sections[i - 2].phase:
                prev_p = narrative.sections[i - 2].phase
                lines.append(f"> **承接上一阶段**: {prev_p.unresolved_problem}")
            lines.append("")

        # Narrative
        lines.append(section.narrative)
        lines.append("")

        # Mermaid graph — nodes from phase key_papers, edges from within-phase relations.
        # Use key_papers (LLM-assigned phase membership) to avoid time-range overlap leaks.
        key_papers: list[str] = []
        if hasattr(section, 'phase') and section.phase:
            key_papers = section.phase.key_papers or []
        relations = section.claim_relations or []

        if key_papers:
            paper_labels: dict[str, str] = {}
            for title in key_papers:
                paper_labels[title] = _paper_label(title)

            lines.extend([
                "**思想演化图**",
                "",
                "```mermaid",
                "flowchart LR",
            ])
            node_ids = {}
            node_idx = 0
            for paper_title in paper_labels:
                nid = f"P{node_idx}"
                node_ids[paper_title] = nid
                node_idx += 1
                label = paper_labels[paper_title].replace('"', "'")
                lines.append(f'    {nid}["{label}"]')

            for r in relations:
                src = r.source_paper if hasattr(r, 'source_paper') else r["source_paper"]
                tgt = r.target_paper if hasattr(r, 'target_paper') else r["target_paper"]
                rel = r.relation if hasattr(r, 'relation') else r["relation"]
                src_id = node_ids.get(src, "")
                tgt_id = node_ids.get(tgt, "")
                if src_id and tgt_id:
                    rel_map = {
                        "improve": " -->|改进| ",
                        "extend": " -->|扩展| ",
                        "support": " -->|支持| ",
                        "replace": " ==>|替代| ",
                        "attack": " ==>|挑战| ",
                        "parallel": " -.->|并行| ",
                    }
                    edge = rel_map.get(rel, f" -->|{rel}| ")
                    lines.append(f"    {src_id}{edge}{tgt_id}")

            lines.extend([
                "```",
                "",
            ])

            # Cross-phase outgoing transitions (source in this phase, target in next phase)
            if i < len(narrative.sections):
                next_phase = narrative.sections[i]
                next_key_papers = set()
                if hasattr(next_phase, 'phase') and next_phase.phase:
                    next_key_papers = set(next_phase.phase.key_papers or [])
                all_relations = narrative.claim_relations or []
                cross = []
                for r in all_relations:
                    src = r.source_paper if hasattr(r, 'source_paper') else r["source_paper"]
                    tgt = r.target_paper if hasattr(r, 'target_paper') else r["target_paper"]
                    if src in paper_labels and tgt in next_key_papers:
                        rel = r.relation if hasattr(r, 'relation') else r["relation"]
                        if rel in ("replace", "attack"):
                            cross.append((src, tgt, rel))
                if cross:
                    items = []
                    for src, tgt, rel in cross:
                        src_label = paper_labels.get(src, src[:30])
                        tgt_label = _paper_label(tgt)
                        tag = "替代" if rel == "replace" else "挑战"
                        items.append(f"{src_label} → {tgt_label} [{tag}]")
                    lines.append(f"*范式转折: {'; '.join(items)}*")
                    lines.append("")

        # Claims table — filtered to phase key_papers to avoid time-range overlap
        all_section_claims = section.claims or []
        key_paper_set = set(key_papers)
        section_claims = [c for c in all_section_claims if c.paper_title in key_paper_set]
        if section_claims:
            claims_by_paper: dict[str, list[Claim]] = {}
            paper_meta: dict[str, tuple[str, int]] = {}
            for claim in section_claims:
                pid = claim.paper_id
                if pid not in claims_by_paper:
                    claims_by_paper[pid] = []
                    paper_meta[pid] = (claim.paper_title, claim.year)
                claims_by_paper[pid].append(claim)

            sorted_pids = sorted(claims_by_paper.keys(),
                                key=lambda pid: paper_meta[pid][1])

            lines.extend([
                "**关键论文与核心主张**",
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

        # Causal chain hook to next phase
        if hasattr(section, 'phase') and section.phase and i < len(narrative.sections):
            lines.append(f"> **→ 遗留问题**: {section.phase.unresolved_problem}")
            lines.append("")

        lines.append("---")
        lines.append("")

    # ── §4 Open Questions (0-3) ─────────────────────────────────────────
    open_qs = getattr(narrative, 'open_questions', None) or []
    if open_qs:
        lines.extend([
            "## 4. 开放问题",
            "",
        ])
        for q in open_qs:
            lines.append(f"- {q}")
        lines.append("")

    # ── §5 Reading List ──────────────────────────────────────────────────
    reading = getattr(narrative, 'reading_list', None) or []
    if reading:
        lines.extend([
            "---",
            "",
            "## 5. 推荐阅读",
            "",
        ])
        # Group by phase
        by_phase: dict[str, list[dict]] = {}
        for item in reading:
            p = item.get("phase", "") or "其它"
            by_phase.setdefault(p, []).append(item)

        for phase_name, items in by_phase.items():
            lines.append(f"### {phase_name}")
            lines.append("")
            for item in items:
                title = item.get("title", "")
                year = item.get("year", "")
                contrib = item.get("contribution", "")
                year_str = f" ({year})" if year else ""
                lines.append(f"- **{title}**{year_str} — {contrib}")
            lines.append("")

    # ── §6 Synthesis ─────────────────────────────────────────────────────
    if narrative.synthesis:
        lines.extend([
            "---",
            "",
            "## 6. 领域趋势与展望",
            "",
            narrative.synthesis,
            "",
        ])

    lines.append(f"*Generated by Research Narrative Engine V8.1 (MVP)*")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    main()
