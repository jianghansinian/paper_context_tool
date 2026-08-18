"""Structured field evolution narrative from one-shot analysis result — Scheme B.

Consumes the one-shot output {phases, shifts, claims, tensions} and generates
a 6-section structured markdown document:

  1. Field overview + phase summary table
  2. Paradigm shifts list
  3. Per-phase evolution (narrative + mermaid diagram + key paper claims)
  4. Open questions
  5. Recommended reading
  6. Field trends & outlook

LLM calls: N+2 (1 overview + N per-phase + 1 synthesis).
Design: docs/design_stage_boundary.md §6.6 (v4.5).

Usage:
    from one_shot_narrative import generate_evolution_md

    md = generate_evolution_md(result, field_name, client)
"""
from __future__ import annotations

import json
import os
import sys
import re
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

from llm_analyzer import build_analyzer_client, _resolve_model
import config


# ═══════════════════════════════════════════════════════════════════════
# Phase Narrative Prompt
# ═══════════════════════════════════════════════════════════════════════

_PHASE_SYSTEM = """\
You are a historian of science writing for an audience that wants to understand \
HOW ideas evolved — not a chronological catalog of papers.

WRITING MODEL: Idea-Centric History
- Lead with the INSIGHT, then name the paper that crystallized it.
- Papers are evidence for the story, not the story itself.
- Each section must be concise and tightly written.

TIME DISCIPLINE: ONLY reference papers that exist WITHIN or BEFORE this phase's \
time range. Do NOT mention future papers by name.

OUTPUT: Return a JSON object with these fields:
- narrative: The full narrative text following the structure below
- mermaid: A mermaid flowchart LR diagram showing paper relationships within this phase
- key_insight: ONE bold memorable sentence (an aphorism)
- unresolved: A brief statement of what problem remains unsolved

NARRATIVE STRUCTURE — use these EXACT sub-headings on their own lines:

**背景** — Brief setup of why this phase started.

**核心发现** — A few short paragraphs. Lead with the conceptual advance. \
Mention papers inline: "X showed Y (Paper, YEAR)." Use bullet points \
(- Approach A: ...) only for genuinely parallel approaches.

**转折点** — What result shifted the debate?

**关键认知** — ONE bold sentence.

**遗留问题** — Brief: "But this created a new problem: ..."

MERMAID DIAGRAM RULES:
- Use `flowchart LR` (left-to-right)
- Node IDs: P0, P1, P2, ... in chronological order
- Node labels: short paper names, use abbreviations
- Edge types: -->|改进| for improvement/extension, -.->|并行| for parallel/independent, ==>|替代| for replacement
- Example:
  ```mermaid
  flowchart LR
      P0["Paper A"]
      P1["Paper B"]
      P2["Paper C"]
      P0 -->|改进| P1
      P1 -->|改进| P2
  ```"""

_PHASE_PROMPT = """\
Write a concise historical account of this phase in {field_name}.

PHASE: **{phase_name}** ({time_range})

PREVIOUS UNSOLVED PROBLEM: "{prev_unresolved}"

DOMINANT QUESTION: "{dominant_question}"

CORE TENSION: {core_tension}

PAPERS in this phase (chronological):
{phase_papers}

CLAIMS from these papers:
{phase_claims}

TENSIONS (debates within this phase):
{tensions_text}

Write the narrative following the structure exactly. Generate a mermaid diagram \
showing the relationships between these papers. Return ONLY a JSON object."""


# ═══════════════════════════════════════════════════════════════════════
# Field Overview Prompt
# ═══════════════════════════════════════════════════════════════════════

_OVERVIEW_SYSTEM = """\
You are a historian of science. Write a concise field overview.
Use **bold** for key concepts. A few sentences covering the overall trajectory.
Do NOT list individual phases, papers, or metrics.
Return ONLY: {"overview": "..."}"""

_OVERVIEW_PROMPT = """\
Write a BRIEF field overview for {field_name}.

PHASES (for context only — do NOT enumerate them):
{phases_summary}

PARADIGM SHIFTS:
{shifts_summary}

Describe the starting point, the key paradigm shifts, and where the field ended up."""


# ═══════════════════════════════════════════════════════════════════════
# Synthesis Prompt
# ═══════════════════════════════════════════════════════════════════════

_SYNTHESIS_SYSTEM = """\
You are a historian of science writing concluding remarks.
Return ONLY a JSON object with: synthesis, open_questions, reading_list."""

_SYNTHESIS_PROMPT = """\
Write the concluding content for {field_name}.

PHASE SUMMARIES:
{phase_summaries}

ALL PAPERS with claims:
{all_claims}

1. SYNTHESIS — ONE paragraph summarizing the overall trajectory. \
   What changed and why? Use **bold** for key concepts.

2. OPEN QUESTIONS — The questions the field still hasn't resolved. \
   Return as a list of strings.

3. READING LIST — Group the most impactful papers by phase. Return as:
   [{{"phase": "Phase Name", "title": "Paper Title", "year": 2022, "contribution": "one sentence"}}]
   Pick the most impactful ones.

Return JSON:
```json
{{
  "synthesis": "...",
  "open_questions": ["q1?", "q2?"],
  "reading_list": [...]
}}
```"""


# ═══════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════

def _call_llm(client, system: str, prompt: str, *, model=None,
              max_tokens=None) -> str:
    model = _resolve_model(model)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=max_tokens or config.LLM_ANALYZER_MAX_TOKENS,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC * 2,
        )
        return resp.choices[0].message.content or ""
    except Exception as exc:
        print(f"  LLM call failed: {exc}")
        return ""


def _parse_json_field(raw: str, field: str) -> Optional[str]:
    if not raw or not raw.strip():
        return None
    raw_clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw_clean = re.sub(r"\s*```$", "", raw_clean)
    try:
        data = json.loads(raw_clean)
        if isinstance(data, dict):
            return data.get(field, "").strip() or None
    except json.JSONDecodeError:
        pass
    return None


def _parse_json_list(raw: str, field: str) -> list:
    if not raw or not raw.strip():
        return []
    raw_clean = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw_clean = re.sub(r"\s*```$", "", raw_clean)
    try:
        data = json.loads(raw_clean)
        if isinstance(data, dict):
            val = data.get(field, [])
            return val if isinstance(val, list) else []
    except json.JSONDecodeError:
        pass
    return []


def _load_paper_years() -> dict[str, int]:
    """Build title -> year lookup from the paper cache index."""
    index_path = PROJECT_ROOT / "data" / "paper_cache" / "index.json"
    lookup: dict[str, int] = {}
    if not index_path.exists():
        return lookup
    index = json.loads(index_path.read_text())
    for entry in index.values():
        title = entry.get("title", "")
        year = entry.get("year", 0)
        if title and year:
            lookup[title] = year
    return lookup


def _fuzzy_match(title: str, lookup: dict[str, int]) -> int:
    """Find year for a paper title, with fuzzy matching."""
    if title in lookup:
        return lookup[title]
    title_lower = title.lower().strip()
    for key, year in lookup.items():
        if title_lower == key.lower().strip():
            return year
        if title_lower in key.lower() or key.lower() in title_lower:
            return year
        # Match first 40 chars
        if title_lower[:40] == key.lower()[:40]:
            return year
    return 0


def _short_name(title: str, max_len: int = 25) -> str:
    """Create a short display name for mermaid nodes."""
    if len(title) <= max_len:
        return title
    return title[:max_len - 3] + "..."


# ═══════════════════════════════════════════════════════════════════════
# Main generator
# ═══════════════════════════════════════════════════════════════════════

def generate_evolution_md(
    result: dict,
    field_name: str,
    client,
    *,
    model=None,
) -> str:
    """Generate the 6-section structured evolution document from a one-shot result.

    Returns markdown text, or "" on failure.
    """
    phases = result.get("phases", [])
    shifts = result.get("shifts", [])
    claims = result.get("claims", [])
    tensions = result.get("tensions", [])

    if not phases:
        print("ERROR: no phases in result")
        return ""

    # Load paper year lookup
    year_lookup = _load_paper_years()

    # Index claims by paper title
    claims_by_paper: dict[str, list[dict]] = {}
    for c in claims:
        claims_by_paper.setdefault(c["paper"], []).append(c)

    # Index tensions by phase
    tensions_by_phase: dict[int, list[dict]] = {}
    for t in tensions:
        tensions_by_phase.setdefault(t.get("phase", -1), []).append(t)

    # Build phases summary for overview
    phases_lines = []
    for p in phases:
        papers_str = ", ".join(p.get("papers", [])[:3])
        if len(p.get("papers", [])) > 3:
            papers_str += f" (+{len(p['papers']) - 3})"
        phases_lines.append(
            f"  Phase {p['index']}: {p['name']} ({p.get('year_range', '?')}) — {papers_str}"
        )
    phases_summary = "\n".join(phases_lines)

    # Build shifts summary for overview
    shifts_lines = []
    for s in shifts:
        shifts_lines.append(f"  {s['shift_name']}: {s.get('trigger', '')[:120]}")
    shifts_summary = "\n".join(shifts_lines)

    # ── 1. Generate Field Overview ──
    print("\nGenerating field overview...")
    overview_raw = _call_llm(
        client, _OVERVIEW_SYSTEM,
        _OVERVIEW_PROMPT.format(
            field_name=field_name,
            phases_summary=phases_summary,
            shifts_summary=shifts_summary,
        ),
        model=model, max_tokens=1024,
    )
    overview = _parse_json_field(overview_raw, "overview") or ""

    # ── 2. Generate Phase Narratives ──
    print(f"Generating narratives for {len(phases)} phases...")
    phase_narratives = []  # list of dicts: narrative, mermaid, key_insight, unresolved

    for i, p in enumerate(phases):
        pidx = p.get("index", i)
        phase_name = p.get("name", f"Phase {pidx}")
        time_range = p.get("year_range", "?")
        dominant_q = p.get("dominant_question", "")
        core_tension = p.get("core_tension", "")
        phase_papers = p.get("papers", [])

        # Previous unresolved problem
        prev_unresolved = "NONE — this is the FIRST phase. Set up the field's original motivation."
        if i > 0 and phase_narratives:
            prev_unresolved = phase_narratives[-1].get("unresolved", "")

        # Paper list with years
        paper_lines = []
        for title in phase_papers:
            y = _fuzzy_match(title, year_lookup)
            y_str = f" ({y})" if y else ""
            paper_lines.append(f"  - {title}{y_str}")

        # Claims for these papers
        claim_lines = []
        for title in phase_papers:
            for c in claims_by_paper.get(title, []):
                claim_lines.append(
                    f"  [{c.get('claim_level', '?')}] {c.get('statement', '')}"
                )
                if c.get("evidence"):
                    claim_lines.append(f"      Evidence: {c['evidence'][:150]}")

        # Tensions for this phase
        phase_tensions = tensions_by_phase.get(pidx, [])
        tensions_lines = []
        for t in phase_tensions:
            tensions_lines.append(f"  - {t.get('name', '')}: {t.get('description', '')}")
            for pos in t.get("positions", []):
                tensions_lines.append(
                    f"    {pos.get('paper', '')}: {pos.get('position', '')}"
                )
        tensions_text = "\n".join(tensions_lines) if tensions_lines else "(No tensions identified)"

        print(f"  Phase {pidx}: {phase_name[:60]}...")

        raw = _call_llm(
            client, _PHASE_SYSTEM,
            _PHASE_PROMPT.format(
                field_name=field_name,
                phase_name=phase_name,
                time_range=time_range,
                prev_unresolved=prev_unresolved,
                dominant_question=dominant_q,
                core_tension=core_tension,
                phase_papers="\n".join(paper_lines),
                phase_claims="\n".join(claim_lines[:20]),  # limit claims to avoid bloat
                tensions_text=tensions_text,
            ),
            model=model,
        )

        narrative = _parse_json_field(raw, "narrative") or ""
        mermaid = _parse_json_field(raw, "mermaid") or ""
        key_insight = _parse_json_field(raw, "key_insight") or ""
        unresolved = _parse_json_field(raw, "unresolved") or ""

        phase_narratives.append({
            "narrative": narrative,
            "mermaid": mermaid,
            "key_insight": key_insight,
            "unresolved": unresolved,
        })

    # ── 3. Generate Synthesis ──
    print("Generating synthesis + open questions + reading list...")
    phase_summaries_text = "\n\n".join(
        f"Phase {phases[i].get('index', i)}: {phases[i].get('name', '')} "
        f"({phases[i].get('year_range', '')})\n"
        f"Key insight: {phase_narratives[i].get('key_insight', '')}\n"
        f"Unresolved: {phase_narratives[i].get('unresolved', '')}"
        for i in range(len(phases))
    )

    all_claims_text = "\n".join(
        f"[{_fuzzy_match(c.get('paper', ''), year_lookup) or '?'}] "
        f"{c.get('paper', '')}: {c.get('statement', '')[:120]}"
        for c in claims
    )

    synth_raw = _call_llm(
        client, _SYNTHESIS_SYSTEM,
        _SYNTHESIS_PROMPT.format(
            field_name=field_name,
            phase_summaries=phase_summaries_text,
            all_claims=all_claims_text,
        ),
        model=model,
    )

    synthesis = _parse_json_field(synth_raw, "synthesis") or ""
    open_questions = _parse_json_list(synth_raw, "open_questions")
    reading_list = []
    raw_clean = re.sub(r"^```(?:json)?\s*", "", synth_raw.strip())
    raw_clean = re.sub(r"\s*```$", "", raw_clean)
    try:
        data = json.loads(raw_clean)
        if isinstance(data, dict):
            rl = data.get("reading_list", [])
            if isinstance(rl, list):
                reading_list = [item for item in rl if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass

    # ── 4. Assemble Markdown ──
    print("Assembling markdown...")
    md = _assemble_markdown(
        field_name=field_name,
        phases=phases,
        shifts=shifts,
        claims_by_paper=claims_by_paper,
        year_lookup=year_lookup,
        overview=overview,
        phase_narratives=phase_narratives,
        synthesis=synthesis,
        open_questions=open_questions,
        reading_list=reading_list,
    )

    return md


def _assemble_markdown(
    field_name: str,
    phases: list[dict],
    shifts: list[dict],
    claims_by_paper: dict[str, list[dict]],
    year_lookup: dict[str, int],
    overview: str,
    phase_narratives: list[dict],
    synthesis: str,
    open_questions: list[str],
    reading_list: list[dict],
) -> str:
    lines = []
    w = lines.append

    w(f"# {field_name} — 技术发展叙事")
    w("")

    # ── 1. 领域全景 ──
    w("## 1. 领域全景")
    w("")
    w(overview)
    w("")
    w("| Phase | Time | Key Papers |")
    w("|-------|------|------------|")
    for p in phases:
        papers = p.get("papers", [])
        papers_str = ", ".join(papers[:3])
        if len(papers) > 3:
            papers_str += f" (+{len(papers) - 3})"
        w(f"| {p.get('name', '?')} | {p.get('year_range', '?')} | {papers_str} |")
    w("")
    w("---")
    w("")

    # ── 2. 范式转移 ──
    w("## 2. 范式转移")
    w("")
    for s in shifts:
        w(f"- **{s.get('shift_name', '?')}**")
    w("")
    w("---")
    w("")

    # ── 3. 阶段演化 ──
    w("## 3. 阶段演化")
    w("")

    for i, p in enumerate(phases):
        phase_name = p.get("name", f"Phase {i}")
        time_range = p.get("year_range", "?")
        core_tension = p.get("core_tension", "")
        phase_papers = p.get("papers", [])
        pn = phase_narratives[i] if i < len(phase_narratives) else {}

        w(f"### 3.{i + 1} Phase {i + 1}: {phase_name} ({time_range})")
        w("")

        # Metadata block
        w(f"> **核心矛盾**: {core_tension}")
        if i > 0:
            prev_unresolved = phase_narratives[i - 1].get("unresolved", "")
            if prev_unresolved:
                w(f"> **承接上一阶段**: {prev_unresolved}")
        w("")

        # Narrative
        narrative = pn.get("narrative", "")
        if narrative:
            w(narrative)
            w("")

        # Mermaid diagram
        mermaid = pn.get("mermaid", "")
        if mermaid:
            mermaid = mermaid.strip()
            if mermaid.startswith("```"):
                w(mermaid)
            else:
                w("```mermaid")
                w(mermaid)
                w("```")
            w("")

        # Key papers table
        w("**关键论文与核心主张**")
        w("")
        w("| 论文 | 年份 | 主张 | 证据 |")
        w("|------|------|------|------|")
        for title in phase_papers:
            year = _fuzzy_match(title, year_lookup)
            year_str = str(year) if year else "?"
            paper_claims = claims_by_paper.get(title, [])
            if paper_claims:
                c = paper_claims[0]
                statement = c.get("statement", "")[:120]
                evidence = c.get("evidence", "")[:120]
                w(f"| **{title}** | {year_str} | {statement} | {evidence} |")
            else:
                w(f"| **{title}** | {year_str} | — | — |")
        w("")

        w("---")
        w("")

    # ── 4. 开放问题 ──
    if open_questions:
        w("## 4. 开放问题")
        w("")
        for q in open_questions:
            w(f"- {q}")
        w("")
        w("---")
        w("")

    # ── 5. 推荐阅读 ──
    if reading_list:
        w("## 5. 推荐阅读")
        w("")
        # Group by phase
        phase_groups: dict[str, list[dict]] = {}
        for item in reading_list:
            phase_name = item.get("phase", "Other")
            phase_groups.setdefault(phase_name, []).append(item)

        for phase_name, items in phase_groups.items():
            # Normalize "Phase N: ..." → "Phase N+1: ..." (display as 1-indexed)
            display_name = phase_name
            m = re.match(r"Phase (\d+):(.+)", phase_name)
            if m:
                display_name = f"Phase {int(m.group(1)) + 1}:{m.group(2)}"
            w(f"### {display_name}")
            w("")
            for item in items:
                title = item.get("title", "?")
                year = item.get("year", "?")
                contrib = item.get("contribution", "")
                w(f"- **{title}** ({year}) — {contrib}")
            w("")
        w("---")
        w("")

    # ── 6. 领域趋势与展望 ──
    if synthesis:
        w("## 6. 领域趋势与展望")
        w("")
        w(synthesis)
        w("")
        w("---")
        w("")

    w("*Generated by Paper Context Tool (E2E V4)*")
    w("")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Generate structured evolution markdown from one-shot result")
    ap.add_argument("--result", required=True,
                    help="Path to one-shot result JSON (e.g. output/v4/xxx/one_shot_result.json)")
    ap.add_argument("--domain", required=True,
                    help="Field/domain name")
    ap.add_argument("--out", default="",
                    help="Output path (default: <result_dir>/<domain_slug>_evolution.md)")
    args = ap.parse_args()

    result_path = PROJECT_ROOT / args.result
    if not result_path.exists():
        print(f"ERROR: result file not found: {result_path}")
        sys.exit(1)

    result = json.loads(result_path.read_text())
    print(f"Loaded result: {len(result.get('phases', []))} phases, "
          f"{len(result.get('shifts', []))} shifts, "
          f"{len(result.get('claims', []))} claims, "
          f"{len(result.get('tensions', []))} tensions")

    client = build_analyzer_client()
    if not client:
        print("ERROR: no LLM client available")
        sys.exit(1)

    md = generate_evolution_md(result, args.domain, client)

    if not md:
        print("ERROR: generation failed")
        sys.exit(1)

    if args.out:
        out_path = Path(args.out)
    else:
        slug = args.domain.lower().replace(" ", "_").replace("-", "_")
        out_path = result_path.parent / f"{slug}_evolution.md"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md)
    print(f"\nWritten: {out_path} ({len(md):,} chars)")


if __name__ == "__main__":
    main()
