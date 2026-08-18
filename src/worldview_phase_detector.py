"""Shift-driven phase detection — paradigm shifts define stage boundaries,
papers are assigned to stages deterministically.

Pipeline:
  1. Paradigm shift detection (1 LLM call — identify turning points)
  2. Stage building (deterministic — N shifts → N+1 stages)
  3. Paper-to-stage assignment (1 LLM call — classify, don't cluster)
  4. Phase building (1 LLM call — name + dominant_question per stage)
  5. Merge empty Stage 0 (deterministic post-processing)

Stability: shift detection has low variance (shifts are historical facts),
stage boundaries are deterministic, and paper assignment is classification
with fixed boundaries — three orders of magnitude stabler than "group papers" approaches.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _resolve_model, _extract_json_object
from paper import Claim, Tension, Phase, ClaimRelation, ParadigmShift


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — Paradigm Shift Detection (1 LLM call)
# ═══════════════════════════════════════════════════════════════════════

_SHIFT_SYSTEM = """\
You are a research historian. Your task is to find the PARADIGM SHIFTS in a field's
evolution — the moments when the community's CORE BELIEF about what problem matters
and how to solve it fundamentally changed.

You think like this:
  1. Tell the evolution STORY first — what happened, in what order, WHY each
     new approach emerged (what was the motivation, what gap did it fill?)
  2. In that story, find TURNING POINTS — moments where the field's CORE BELIEF
     changed. Not where a method improved, but where the field started believing
     something fundamentally different about what problem matters or how to solve it.
  3. From those turning points, extract the paradigm shifts.

LITMUS TEST: Would a researcher from before this shift find the new approach
UNINTELLIGIBLE or OBVIOUSLY WRONG without understanding the shift itself?
If yes → paradigm shift. If they'd see it as a natural improvement → technique evolution.

SHIFT (return these):
- Core belief overturned: "Dense BEV grids are NECESSARY" → "Sparse queries suffice"
- Research question changed: "How to fuse features?" → "How to reduce computation?"
- Success definition changed: "Detection accuracy" → "Planning safety"
- A direction was ABANDONED by the community

NOT A SHIFT (technique evolution, do NOT return):
- Adding a dimension: "Single-frame → Temporal fusion" — same belief
- Architecture swap: "CNN → Transformer" — same paradigm, different tool
- Efficiency improvement within same paradigm
- A paper achieving SOTA — that's a result, not a belief change

You must write your STORY and TURNING POINTS before the JSON, then output
the shifts as a JSON object at the end."""

_SHIFT_PROMPT = """\
Trace the evolution of {field_name} through these claims and field-level tensions,
then identify the paradigm shifts — the moments when the community's CORE BELIEF
about what problem matters and how to solve it fundamentally changed.

CLAIMS (chronological, each is a paper's core contribution):
{claims_text}

KNOWN TENSIONS (field-level contradictions between competing positions,
already identified — each is a DEBATE that divided the community):
{tensions_text}

STEP 1 — Tell the evolution STORY:
  Describe the narrative arc. Use the TENSIONS as signposts — each tension is
  a debate that the field engaged in. Which tensions were resolved? Which ones
  represent genuine changes in what the field BELIEVES, versus which are just
  normal competition between methods within the same belief system?

STEP 2 — For each tension, JUDGE whether it marks a paradigm shift:
  A paradigm shift means the field's CORE BELIEF changed. Not every tension is
  a paradigm shift — some are just engineering trade-offs within a shared paradigm.

  Ask for each tension: "Did the field start believing something fundamentally
  DIFFERENT after this debate resolved?" If yes → paradigm shift. If no (just
  a better implementation or architecture choice) → skip it.

  Also identify any paradigm shifts NOT captured by the tensions, if the claims
  reveal a belief change the tensions missed.

STEP 3 — Output the shifts as JSON:
{{
  "shifts": [
    {{
      "shift_name": "Dense BEV → Sparse Representation",
      "old_paradigm": "...",
      "new_paradigm": "...",
      "catalyst_papers": ["..."],
      "year_range": "..."
    }}
  ]
}}

Common FALSE POSITIVES to reject:
- Temporal fusion added to single-frame detector: improvement, not belief change
- Architecture swap (CNN→Transformer, ResNet→ViT): different tool, same paradigm
- Depth supervision added: refinement, not belief change
- A paper achieves SOTA: result, not paradigm shift
- "X method vs Y method for same problem": competition, not paradigm shift

Only include shifts where the field's belief GENUINELY changed. Most tensions are
internal debates within a paradigm, not paradigm boundaries."""


# ═══════════════════════════════════════════════════════════════════════
# Step 1-alt — Narrative-First Shift Detection (2 LLM calls)
# ═══════════════════════════════════════════════════════════════════════

_NARRATIVE_SYSTEM = """\
You are a research historian writing the evolution story of a research field.
Your task is to write a coherent, causal narrative tracing how the field's core
beliefs, research questions, and technical approaches changed over time.

Write like a historian, not a bibliographer:
- Lead with IDEAS, not paper names. Papers are footnotes to ideas.
- Every paragraph should answer "WHY did the field move from X to Y?"
- Be specific about turning points — name the papers that triggered change
- Distinguish between technique evolution (improvement within same belief) and
  paradigm shift (belief itself changed)

Your narrative MUST explicitly mark TURNING POINTS using the tag [TURNING POINT]
followed by a concise explanation of what belief changed and which paper(s)
triggered the change. These turning points will later be extracted as paradigm shifts.

A TURNING POINT meets this litmus test: would a researcher from before this point
find the new approach UNINTELLIGIBLE or OBVIOUSLY WRONG? If yes → mark it.
If they'd see it as a natural improvement → do NOT mark it.

Write 600-1200 words. Be concise but specific."""

_NARRATIVE_PROMPT = """\
Write the evolution story of {field_name} based on the claims, tensions, and
paper relationships below.

CLAIMS (chronological, each is a paper's core contribution to the field):
{claims_text}

KNOWN TENSIONS (field-level debates, each with which papers took which side):
{tensions_text}

PAPER RELATIONSHIPS (how later work relates to earlier work):
{relations_text}

RESEARCH QUESTIONS (the organizing questions of this field):
{rqs_text}

---
Write a narrative that:

1. Traces the causal chain — WHY did each new approach emerge? What gap or
   failure of the previous approach motivated it?

2. Identifies TURNING POINTS using the [TURNING POINT] tag. A turning point is
   where the field's CORE BELIEF about what problem matters or how to solve it
   fundamentally changed. Mark each concisely, naming the key papers.

   A turning point is a BELIEF REVERSAL: the field used to believe X, now it
   believes NOT-X (or Y instead of X). For a turning point to exist, there must
   be a BEFORE belief that was broadly held and an AFTER belief that replaced it.

   Turning points should be RARE. Most papers are technique evolution within
   a shared paradigm. When multiple candidate turning points exist, keep only
   those where the field's core belief genuinely reversed — the rest are
   tensions or method debates. If you have many candidates, merge the minor
   ones into the narrative without the [TURNING POINT] tag.

   WHAT IS NOT A TURNING POINT:
   - The founding paper(s) that establish the field's FIRST paradigm. A turning
     point requires a BEFORE belief to overturn — the first paradigm has none.
   - An internal debate between two methods that SHARE the same core belief
     (e.g. "should we use method A or B to achieve the same goal?"). These are
     TENSIONS, not turning points — describe them in the narrative without the tag.
   - A refinement or extension of an existing paradigm to a new task. If the core
     belief didn't change, it's technique evolution, not a turning point.

   WHAT IS A TURNING POINT:
   - The field abandons a core assumption (e.g. "we need X" → "we don't need X")
   - The field redefines what success means (e.g. "accuracy matters" → "safety
     and planning quality matter more than detection accuracy")
   - The field unifies two previously separate sub-problems under one framework

3. Distinguishes paradigm shifts from technique evolution. Not every new paper
   is a turning point — most are incremental improvements within a shared belief.
   A paper proposing a fundamentally different approach IS a turning point only if
   it caused the community to CHANGE what they believe, not just try a new method.

4. Uses specific paper names and years. Don't say "some researchers believed" —
   say "BEVDepth (2022) showed that explicit depth supervision..."

Do NOT invent turning points not supported by the data. Do NOT mark technique
evolution as turning points. Quality over quantity.

Return ONLY the narrative text, no JSON wrapper."""

_NARRATIVE_SHIFT_SYSTEM = """\
You are a research historian. Your task is to extract PARADIGM SHIFTS from a
field's evolution narrative.

The narrative below was written by a colleague who has identified key TURNING
POINTS (marked with [TURNING POINT] tags). Your job is to judge which of these
turning points represent genuine PARADIGM SHIFTS and extract only those.

A paradigm shift changes WHAT PROBLEM the field is solving or WHAT COUNTS as a
valid solution. A technique debate or method improvement changes HOW the same
problem is solved — these are TENSIONS, not paradigm shifts.

STRONG LITMUS TEST (apply to EVERY turning point):
1. Did this change WHAT GOAL the field pursues? (e.g., from "maximize detection
   accuracy" to "maximize planning safety") → paradigm shift
2. Did this change WHAT REPRESENTATION is considered valid? (e.g., from "dense
   grid is necessary" to "sparse features are sufficient") → paradigm shift
3. Did this only change HOW to achieve the same goal using the SAME TYPE of
   representation? → NOT a paradigm shift, skip it.

   KEY TEST for #3: Do both sides of the debate produce the SAME KIND of output?
   If both produce dense grids, or both produce bounding boxes, or both solve
   the same task with different computational mechanisms — the debate is a
   TENSION, not a paradigm shift. Two different algorithms for the same output
   format is a method debate. A paradigm shift changes the output format itself
   or what the output is used for.

CRITICAL RULES:
1. Only extract shifts marked as [TURNING POINT] in the narrative.
2. Apply the strong litmus test to each turning point. Skip any that only change
   HOW a problem is solved (even if the narrative calls it a belief reversal).
   A debate about the right METHOD is a tension. Different projection mechanisms
   for the same output format are tensions, not paradigm shifts.
3. Two turning points that represent the same paradigm shift applied to different
   tasks (e.g., detection vs planning) should be MERGED into one shift.
4. If in doubt after applying the litmus test, keep the shift.

Return ONLY a JSON object. No other text."""

_NARRATIVE_SHIFT_PROMPT = """\
Below is a narrative describing the evolution of {field_name}. Extract the
paradigm shifts that correspond to the turning points in this story.

NARRATIVE:
{narrative_text}

CLAIMS (for paper title reference — use these exact titles):
{claims_text}

For each GENUINE paradigm shift in the narrative, output:
- shift_name: Short label, e.g. "Dense BEV → Sparse Representation"
- old_paradigm: What the field believed before
- new_paradigm: What the field believed after
- catalyst_papers: Papers that triggered or crystallized this shift
- year_range: e.g. "2022-2023"

Only include shifts that:
1. Are explicitly described as turning points in the narrative
2. Represent a fundamental belief change, not a technique improvement
3. Pass the litmus test

Return JSON:
{{
  "shifts": [
    {{
      "shift_name": "...",
      "old_paradigm": "...",
      "new_paradigm": "...",
      "catalyst_papers": ["..."],
      "year_range": "..."
    }}
  ]
}}"""


def _generate_field_narrative(
    client: OpenAI,
    claims: list[Claim],
    field_name: str,
    tensions: list | None = None,
    relations: list | None = None,
    rqs: list | None = None,
) -> str:
    """Generate a field-level evolution narrative before phase detection.

    The narrative identifies turning points that will be extracted as paradigm shifts.
    """
    # Format claims chronologically
    claims_by_paper: dict[str, list[Claim]] = {}
    for c in claims:
        claims_by_paper.setdefault(c.paper_title, []).append(c)

    papers_sorted = sorted(
        claims_by_paper.items(),
        key=lambda x: (x[1][0].year, getattr(x[1][0], "month", 0)),
    )

    claims_lines = []
    for title, pclaims in papers_sorted:
        c0 = pclaims[0]
        m = getattr(c0, "month", 0)
        date_str = f"{c0.year}-{m:02d}" if m > 0 else str(c0.year)
        claims_lines.append(f"\n[{date_str}] {title}")
        for j, c in enumerate(pclaims, 1):
            level_tag = f" [{c.claim_level}]" if c.claim_level else ""
            claims_lines.append(f"  Claim {j}{level_tag}: {c.statement}")
            if c.problem_addressed:
                claims_lines.append(f"    Problem addressed: {c.problem_addressed}")

    claims_text = "\n".join(claims_lines)

    # Format tensions
    tensions_lines = []
    if tensions:
        for i, t in enumerate(tensions, 1):
            if hasattr(t, 'tension'):
                name, desc, status = t.tension, t.description, t.status
                introduced = t.introduced_by if hasattr(t, 'introduced_by') else []
                resolved = t.resolved_by if hasattr(t, 'resolved_by') else []
            else:
                name = t.get('tension', '?')
                desc = t.get('description', '')
                status = t.get('status', '?')
                introduced = t.get('introduced_by', [])
                resolved = t.get('resolved_by', [])
            tensions_lines.append(f"\nTension {i}: {name}")
            tensions_lines.append(f"  Description: {desc}")
            if introduced:
                tensions_lines.append(f"  Introduced by: {', '.join(introduced)}")
            if resolved:
                tensions_lines.append(f"  Resolved/advanced by: {', '.join(resolved)}")
            tensions_lines.append(f"  Status: {status}")
    tensions_text = "\n".join(tensions_lines) if tensions_lines else "(no tensions available)"

    # Format relations
    relations_lines = []
    if relations:
        for r in relations:
            if hasattr(r, 'source_paper'):
                src, tgt, rel, exp = r.source_paper, r.target_paper, r.relation, r.explanation
            else:
                src = r.get('source_paper', '?')
                tgt = r.get('target_paper', '?')
                rel = r.get('relation', '?')
                exp = r.get('explanation', '')
            relations_lines.append(f"  {src} → {tgt}: {rel.upper()}")
            if exp:
                relations_lines.append(f"    {exp}")
    relations_text = "\n".join(relations_lines) if relations_lines else "(no relations available)"

    # Format RQs
    rqs_lines = []
    if rqs:
        for i, rq in enumerate(rqs, 1):
            if hasattr(rq, 'question'):
                q, desc, level = rq.question, rq.description, rq.level
            else:
                q = rq.get('question', '?')
                desc = rq.get('description', '')
                level = rq.get('level', '?')
            rqs_lines.append(f"\n  RQ{i} [{level}]: {q}")
            if desc:
                rqs_lines.append(f"    Context: {desc}")
    rqs_text = "\n".join(rqs_lines) if rqs_lines else "(no RQs available)"

    prompt = _NARRATIVE_PROMPT.format(
        field_name=field_name,
        claims_text=claims_text,
        tensions_text=tensions_text,
        relations_text=relations_text,
        rqs_text=rqs_text,
    )

    try:
        response = client.chat.completions.create(
            model=_resolve_model("deepseek-chat"),
            messages=[
                {"role": "system", "content": _NARRATIVE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        narrative = response.choices[0].message.content or ""
        return narrative.strip()
    except Exception as exc:
        print(f"Field narrative generation failed: {exc}")
        return ""


def _extract_shifts_from_narrative(
    client: OpenAI,
    narrative: str,
    claims: list[Claim],
    field_name: str,
) -> list[dict]:
    """Extract paradigm shifts from a pre-written field narrative.

    This is more constrained than _detect_shifts: the LLM reads the narrative's
    turning points rather than detecting shifts from raw claims.
    """
    if not narrative.strip():
        return []

    # Format claims concisely (just titles and years for reference)
    claims_lines = []
    seen = set()
    for c in sorted(claims, key=lambda c: (c.year, getattr(c, 'month', 0))):
        if c.paper_title not in seen:
            seen.add(c.paper_title)
            claims_lines.append(f"  [{c.year}] {c.paper_title}")
    claims_text = "\n".join(claims_lines)

    prompt = _NARRATIVE_SHIFT_PROMPT.format(
        field_name=field_name,
        narrative_text=narrative,
        claims_text=claims_text,
    )

    try:
        response = client.chat.completions.create(
            model=_resolve_model("deepseek-chat"),
            messages=[
                {"role": "system", "content": _NARRATIVE_SHIFT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        data = _extract_json_object(raw)
        if not data:
            return []
        return data.get("shifts", [])
    except Exception as exc:
        print(f"Narrative shift extraction failed: {exc}")
        return []


# ═══════════════════════════════════════════════════════════════════════
# Step 3 — Paper-to-Stage Assignment (1 LLM call)
# ═══════════════════════════════════════════════════════════════════════

_ASSIGN_SYSTEM = "You are precise and systematic. Assign each paper to the stage whose paradigm it follows."

_ASSIGN_PROMPT = """Below are {n_papers} papers in {field_name}.

{claims_text}

The field evolved through {n_stages} stages separated by {n_shifts} paradigm shifts.

STAGE DEFINITIONS:
{stages_text}

CRITICAL RULES:
1. Assign each paper to exactly ONE stage.
2. A paper belongs to the stage whose paradigm it follows.
3. If a paper TRIGGERS a shift, it belongs to the stage AFTER that shift.
4. "Sparse detection" and "end-to-end planning" are DIFFERENT stages — do NOT merge them.
5. A paper about sparse object detection belongs to the sparse stage, NOT the
   end-to-end stage, even if both use sparse representations.

Return ONLY a JSON object:
{{
  "assignments": [
    {{"paper": "full paper title", "stage_index": 0, "reason": "one sentence"}}
  ]
}}
"""


# ═══════════════════════════════════════════════════════════════════════
# Step 4 — Phase Building (1 LLM call, adapted from existing prompt)
# ═══════════════════════════════════════════════════════════════════════

_PHASE_BUILDING_SYSTEM = """\
You build coherent research phases from paper groups defined by paradigm shift boundaries.

Each group contains papers that belong to the same stage — a period between two
paradigm shifts. The stage is defined by a dominant paradigm. Your job: name the
phase, identify the dominant question, detect internal tensions, and connect
phases into a causal chain.

A Phase is a CHAPTER in the story. It has:
- A DOMINANT QUESTION that anchors all papers in this phase
- Internal TENSIONS (debates between papers within the phase, if any)
- An UNRESOLVED PROBLEM that logically leads to the next phase

Return ONLY a JSON object. No other text."""

_PHASE_BUILDING_PROMPT = """\
Build phases from these shift-defined paper groups in the field of {field_name}.

STAGES (each defined by a paradigm, separated by paradigm shifts):
{stages_text}

For each stage, output a Phase:
- name: question-based phase name, e.g. "How to Build a Reliable BEV from Multi-Camera Images?"
- dominant_question: Core research question driving this phase
- core_contradiction: The central tension/contradiction within this phase (1 sentence)
- core_debate: What was the field debating? (1 sentence)
- key_papers: List of paper titles in this phase
- time_range: e.g. "2020-08—2022-11"
- internal_tensions: List of {{"tension": "short label", "description": "contradiction between positions", "status": "direction_clear|direction_forming|open"}}
- unresolved_problem: The problem that remains → seeds the next phase (1 sentence)
- status: "direction_clear" | "direction_forming" | "open"

PHASE COUNT MUST MATCH STAGE COUNT: {n_stages}
Each paper MUST appear in exactly one phase (COVERAGE: {n_papers} papers total)
unresolved_problem should logically motivate the NEXT phase

Return JSON:
```json
{{"phases": [
  {{
    "name": "...",
    "dominant_question": "...",
    "core_contradiction": "...",
    "core_debate": "...",
    "key_papers": [...],
    "time_range": "...",
    "internal_tensions": [...],
    "unresolved_problem": "...",
    "status": "..."
  }}
]}}
```"""


# ═══════════════════════════════════════════════════════════════════════
# Step 2 — Deterministic Stage Building
# ═══════════════════════════════════════════════════════════════════════

def _shifts_to_stages(shifts: list[dict]) -> list[dict]:
    """N shifts → N+1 stages. Each shift is a boundary between two stages."""
    if not shifts:
        return [{"index": 0, "name": "Single Stage", "paradigm": "All papers in one stage"}]

    stages = []
    # Stage 0: paradigm before the first shift
    shift0 = shifts[0]
    stages.append({
        "index": 0,
        "name": shift0.get("old_paradigm", ""),
        "paradigm": shift0.get("old_paradigm", ""),
    })
    # Stages 1..N: paradigm after each shift
    for i, s in enumerate(shifts):
        stages.append({
            "index": i + 1,
            "name": s.get("new_paradigm", ""),
            "paradigm": s.get("new_paradigm", ""),
        })
    return stages


def _merge_empty_stage_0(stages: list[dict],
                         stage_papers: dict[int, list[str]],
                         shifts: list[dict]) -> tuple[list[dict], dict[int, list[str]], list[dict]]:
    """If Stage 0 has no papers, merge it into Stage 1.

    Also drops the first shift (which separates Stage 0 from Stage 1) since
    Stage 0 no longer exists.
    """
    stage0_empty = 0 not in stage_papers or len(stage_papers.get(0, [])) == 0
    if not stage0_empty:
        return stages, stage_papers, shifts

    if len(stages) <= 1:
        return stages, stage_papers, shifts

    # Remove Stage 0
    stages = stages[1:]
    # Drop the first shift
    shifts = shifts[1:]

    # Re-index stages
    for i, s in enumerate(stages):
        s["index"] = i

    # Re-index paper assignments: all stage indices shift down by 1.
    # Skip Stage 0 — it's empty (checked above) and would overwrite Stage 1's
    # papers at new_idx=0 if it iterates after them (dict order can vary).
    new_stage_papers: dict[int, list[str]] = {}
    for idx, papers in stage_papers.items():
        if idx == 0:
            continue
        new_stage_papers[idx - 1] = papers

    return stages, new_stage_papers, shifts


def _merge_thin_stages(
    stages: list[dict],
    stage_papers: dict[int, list[str]],
    shifts: list[dict],
    min_papers: int = 3,
) -> tuple[list[dict], dict[int, list[str]], list[dict]]:
    """Merge stages with fewer than min_papers papers into adjacent stages.

    Thin stages are usually technique debates misidentified as paradigm shifts.
    This deterministic post-processing catches what prompt tuning cannot.
    Merges backward (into previous stage) to preserve semantic grouping.
    """
    if len(stages) <= 1:
        return stages, stage_papers, shifts

    stages = list(stages)
    stage_papers = dict(stage_papers)
    shifts = list(shifts)

    # Iterate right-to-left so removals don't shift later indices
    for i in range(len(stages) - 1, -1, -1):
        count = len(stage_papers.get(i, []))
        if count == 0 or count >= min_papers:
            continue

        # Find merge target: previous non-empty stage, or next if first stage
        target = i - 1 if i > 0 else i + 1
        if target not in stage_papers:
            continue

        # Move papers
        stage_papers[target].extend(stage_papers.pop(i, []))

        # Remove the shift at min(i, target) — the shift that separated these stages
        shift_idx = min(i, target)
        if shift_idx < len(shifts):
            shifts.pop(shift_idx)

        # Remove the stage
        stages.pop(i)

        print(f"  Merged thin Stage {i} ({count} papers) into Stage {target}")

    # Re-index stages
    for j, s in enumerate(stages):
        s["index"] = j

    # Re-index paper assignments
    new_mapping: dict[int, list[str]] = {}
    old_indices = sorted(stage_papers.keys())
    for new_idx, old_idx in enumerate(old_indices):
        new_mapping[new_idx] = stage_papers[old_idx]

    return stages, new_mapping, shifts


# ═══════════════════════════════════════════════════════════════════════
# Pipeline Steps
# ═══════════════════════════════════════════════════════════════════════

def _detect_shifts(
    client: OpenAI,
    claims: list[Claim],
    field_name: str,
    tensions: list | None = None,
) -> list[dict]:
    """Detect paradigm shifts. Single LLM call."""
    claims_by_paper: dict[str, list[Claim]] = {}
    for c in claims:
        claims_by_paper.setdefault(c.paper_title, []).append(c)

    papers_sorted = sorted(
        claims_by_paper.items(),
        key=lambda x: (x[1][0].year, getattr(x[1][0], "month", 0)),
    )

    claims_lines = []
    for title, pclaims in papers_sorted:
        c0 = pclaims[0]
        m = getattr(c0, "month", 0)
        date_str = f"{c0.year}-{m:02d}" if m > 0 else str(c0.year)
        claims_lines.append(f"\n[{date_str}] {title}")
        for j, c in enumerate(pclaims, 1):
            claims_lines.append(f"  Claim {j}: {c.statement}")
            if c.problem_addressed:
                claims_lines.append(f"    Problem: {c.problem_addressed}")

    claims_text = "\n".join(claims_lines)

    # Format tensions as known field-level debates
    if tensions:
        t_lines = []
        for i, t in enumerate(tensions, 1):
            if hasattr(t, 'tension'):
                name, desc, status = t.tension, t.description, t.status
            else:
                name = t.get('tension', '?')
                desc = t.get('description', '')
                status = t.get('status', '?')
            t_lines.append(f"T{i}: {name}\n  {desc}\n  Status: {status}")
        tensions_text = "\n".join(t_lines)
    else:
        tensions_text = "(no tensions available)"

    prompt = _SHIFT_PROMPT.format(
        field_name=field_name,
        claims_text=claims_text,
        tensions_text=tensions_text,
    )

    try:
        response = client.chat.completions.create(
            model=_resolve_model("deepseek-chat"),
            messages=[
                {"role": "system", "content": _SHIFT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        data = _extract_json_object(raw)
        if not data:
            return []
        return data.get("shifts", [])
    except Exception as exc:
        print(f"  Shift detection failed: {exc}")
        return []


def _assign_papers_to_stages(
    client: OpenAI,
    claims: list[Claim],
    stages: list[dict],
    shifts: list[dict],
    field_name: str,
) -> dict[int, str]:
    """Assign each paper to a stage. Single LLM call."""
    if not stages or not claims:
        return {}

    # Format claims
    claims_by_paper: dict[str, list[Claim]] = {}
    for c in claims:
        claims_by_paper.setdefault(c.paper_title, []).append(c)

    papers_sorted = sorted(
        claims_by_paper.items(),
        key=lambda x: (x[1][0].year, getattr(x[1][0], "month", 0)),
    )

    claims_lines = []
    for title, pclaims in papers_sorted:
        c0 = pclaims[0]
        m = getattr(c0, "month", 0)
        date_str = f"{c0.year}-{m:02d}" if m > 0 else str(c0.year)
        claims_lines.append(f"\n[{date_str}] {title}")
        for j, c in enumerate(pclaims, 1):
            claims_lines.append(f"  Claim {j}: {c.statement}")

    claims_text = "\n".join(claims_lines)

    # Format stages with shift boundaries
    stages_lines = []
    for i, stage in enumerate(stages):
        stages_lines.append(f"Stage {i}: Paradigm — {stage['paradigm'][:120]}")
        if i < len(shifts):
            s = shifts[i]
            stages_lines.append(f"  → SHIFT: {s.get('shift_name', '??')}")
            stages_lines.append(f"    Trigger: {', '.join(s.get('catalyst_papers', []))}")
        stages_lines.append("")
    stages_text = "\n".join(stages_lines)

    prompt = _ASSIGN_PROMPT.format(
        n_papers=len(claims_by_paper),
        n_stages=len(stages),
        n_shifts=len(shifts),
        field_name=field_name,
        claims_text=claims_text,
        stages_text=stages_text,
    )

    try:
        response = client.chat.completions.create(
            model=_resolve_model("deepseek-chat"),
            messages=[
                {"role": "system", "content": _ASSIGN_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        data = _extract_json_object(raw)
        if not data:
            return {}

        assignments = data.get("assignments", [])
        result: dict[int, list[str]] = {}
        for a in assignments:
            idx = a.get("stage_index", -1)
            paper = a.get("paper", "")
            if 0 <= idx < len(stages) and paper:
                result.setdefault(idx, []).append(paper)
        return result
    except Exception as exc:
        print(f"  Paper assignment failed: {exc}")
        return {}


def _build_phases_from_stages(
    client: OpenAI,
    stages: list[dict],
    stage_papers: dict[int, list[str]],
    shifts: list[dict],
    field_name: str,
) -> list[Phase]:
    """Build Phase objects from stages. Single LLM call."""
    if not stages:
        return []

    # Format stages with papers
    stages_lines = []
    all_paper_titles: list[str] = []
    for i, stage in enumerate(stages):
        papers = stage_papers.get(i, [])
        all_paper_titles.extend(papers)
        stages_lines.append(f"\nStage {i}: {stage.get('name', '??')[:120]}")
        stages_lines.append(f"  Paradigm: {stage.get('paradigm', '??')[:120]}")
        if i < len(shifts):
            s = shifts[i]
            stages_lines.append(f"  Shift → {s.get('shift_name', '??')}: {s.get('old_paradigm', '')} → {s.get('new_paradigm', '')}")
        stages_lines.append(f"  Papers ({len(papers)}): {', '.join(papers[:8])}")

    stages_text = "\n".join(stages_lines)
    n_stages = len(stages)

    prompt = _PHASE_BUILDING_PROMPT.format(
        field_name=field_name,
        stages_text=stages_text,
        n_stages=n_stages,
        n_papers=len(all_paper_titles),
    )

    try:
        response = client.chat.completions.create(
            model=_resolve_model("deepseek-chat"),
            messages=[
                {"role": "system", "content": _PHASE_BUILDING_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        phases_data = data.get("phases", [])

        phases: list[Phase] = []
        for idx, pd in enumerate(phases_data):
            tensions: list[Tension] = []
            for t in pd.get("internal_tensions", []):
                tensions.append(Tension(
                    tension=t.get("tension", ""),
                    description=t.get("description", ""),
                    introduced_by=[],
                    resolved_by=[],
                    status=t.get("status", "open"),
                    dimension="",
                    domain_scope="",
                ))
            phases.append(Phase(
                name=pd.get("name", ""),
                time_range=pd.get("time_range", ""),
                core_contradiction=pd.get("core_contradiction", ""),
                key_papers=stage_papers.get(idx, pd.get("key_papers", [])),
                core_debate=pd.get("core_debate", ""),
                unresolved_problem=pd.get("unresolved_problem", ""),
                dominant_question=pd.get("dominant_question", ""),
                tensions=tensions,
                status=pd.get("status", ""),
            ))

        return phases
    except Exception as exc:
        print(f"  Phase building failed: {exc}")
        return []


# ═══════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════

def detect_worldview_phases(
    claims: list[Claim],
    field_name: str,
    client: OpenAI,
    relations: Optional[list[ClaimRelation]] = None,
    tensions: list | None = None,
    *,
    narrative_first: bool = False,
    rqs: list | None = None,
    skip_merge: bool = False,
) -> tuple[list[Phase], list, list, list]:
    """Full shift-driven phase detection pipeline.

    When narrative_first=False (default): detects paradigm shifts directly from
    claims + tensions (1 LLM call for shift detection, then deterministic stages).

    When narrative_first=True: generates a field-level evolution narrative first,
    then extracts paradigm shifts from the narrative's turning points (2 LLM calls
    for shifts, more stable but slightly more expensive).

    Returns (phases, [], [], []) — worldview_beliefs, boundaries, worldview_groups
    are empty since this pipeline doesn't use those intermediate artifacts.
    """
    field_narrative = ""

    # ── Step 1: Detect paradigm shifts ──
    if narrative_first:
        print("\n[Shift-Driven Phase] Step 1a/5: Generating field narrative...")
        field_narrative = _generate_field_narrative(
            client, claims, field_name, tensions=tensions,
            relations=relations, rqs=rqs,
        )
        if field_narrative:
            print(f"  Narrative generated: {len(field_narrative):,} chars")
            # Preview first 200 chars
            preview = field_narrative[:200].replace('\n', ' ')
            print(f"  Preview: {preview}...")
        else:
            print("  Narrative generation failed — falling back to direct shift detection")

        if field_narrative:
            print("\n  Step 1b/5: Extracting shifts from narrative...")
            shifts = _extract_shifts_from_narrative(
                client, field_narrative, claims, field_name,
            )
        else:
            shifts = []

        if not shifts and field_narrative:
            # Narrative was generated but no shifts extracted — try direct detection
            print("  No shifts extracted from narrative — trying direct detection...")
            shifts = _detect_shifts(client, claims, field_name, tensions)
    else:
        print("\n[Shift-Driven Phase] Step 1/5: Detecting paradigm shifts...")
        shifts = _detect_shifts(client, claims, field_name, tensions)

    if not shifts:
        print("  No shifts detected — falling back to single-phase")
        shifts = [{
            "shift_name": "No paradigm shifts detected",
            "old_paradigm": "Field before",
            "new_paradigm": "Field after",
            "catalyst_papers": [],
            "year_range": "?",
        }]

    print(f"  Detected {len(shifts)} paradigm shifts:")
    for i, s in enumerate(shifts, 1):
        print(f"    {i}. {s.get('shift_name', '??')[:80]}")
        print(f"       {s.get('old_paradigm', '??')[:60]} → {s.get('new_paradigm', '??')[:60]}")

    # ── Step 2: Build stages (deterministic) ──
    print(f"\n  Step 2/5: Building stages from shifts (deterministic)...")
    stages = _shifts_to_stages(shifts)
    print(f"  {len(shifts)} shifts → {len(stages)} stages")

    # ── Step 3: Assign papers to stages ──
    print(f"\n  Step 3/5: Assigning papers to stages...")
    stage_papers = _assign_papers_to_stages(client, claims, stages, shifts, field_name)
    if not stage_papers:
        print("  Paper assignment failed — all papers go to Stage 0")
        stage_papers = {0: list({c.paper_title for c in claims})}  # fallback

    # Show assignment
    total_assigned = sum(len(v) for v in stage_papers.values())
    print(f"  {total_assigned} papers assigned to {len(stage_papers)} stages:")
    for idx in sorted(stage_papers.keys()):
        papers = stage_papers[idx]
        short_titles = [p.split(":")[0][:35] for p in papers]
        print(f"    Stage {idx} [{len(papers)}]: {', '.join(short_titles[:5])}")

    # Normalize: ensure all stages (0..n-1) have entries in stage_papers
    for idx in range(len(stages)):
        if idx not in stage_papers:
            stage_papers[idx] = []

    # ── Step 4: Merge empty Stage 0 ──
    print(f"\n  Step 4/5: Checking for empty Stage 0...")
    n_stages_before = len(stages)
    stages, stage_papers, shifts = _merge_empty_stage_0(stages, stage_papers, shifts)
    if len(stages) < n_stages_before:
        print(f"  Merged empty Stage 0 → {len(stages)} stages, {len(shifts)} shifts")
    else:
        print(f"  No empty Stage 0 to merge")

    # ── Step 4.5: Merge thin stages (≤2 papers usually = technique debate, not paradigm shift) ──
    if not skip_merge:
        n_stages_before = len(stages)
        stages, stage_papers, shifts = _merge_thin_stages(stages, stage_papers, shifts)
        if len(stages) < n_stages_before:
            print(f"  Merged thin stages → {len(stages)} stages, {len(shifts)} shifts")
    else:
        print(f"  Skipping thin-stage merge (skip_merge=True)")

    # ── Step 5: Build phases ──
    print(f"\n  Step 5/5: Building phases from stages...")
    phases = _build_phases_from_stages(client, stages, stage_papers, shifts, field_name)
    if phases:
        print(f"\n  Final: {len(phases)} phases")
        all_phase_tensions = []
        for i, p in enumerate(phases):
            all_phase_tensions.extend(p.tensions)
            print(f"    Phase {i + 1}: {p.name} ({p.time_range}) [{p.status}]")
            print(f"      Dominant question: {p.dominant_question[:120]}...")
            print(f"      Core contradiction: {p.core_contradiction[:120]}...")
            print(f"      Unresolved: {p.unresolved_problem[:120]}...")
            print(f"      Key papers: {', '.join(p.key_papers[:4])}")
            if p.tensions:
                print(f"      Tensions ({len(p.tensions)}):")
                for t in p.tensions:
                    print(f"        - {t.tension}: {t.description[:80]}...")
    else:
        print("  Phase building failed")

    return phases, shifts, stages, stage_papers, field_narrative


# ═══════════════════════════════════════════════════════════════════════
# Formatting helpers
# ═══════════════════════════════════════════════════════════════════════

def phases_to_text(phases: list[Phase]) -> str:
    """Compact text representation of phases for narrative builder."""
    lines = []
    for i, p in enumerate(phases, 1):
        lines.append(f"Phase {i}: {p.name} ({p.time_range})")
        lines.append(f"  Dominant question: {p.dominant_question}")
        lines.append(f"  Core contradiction: {p.core_contradiction}")
        lines.append(f"  Core debate: {p.core_debate}")
        lines.append(f"  Key papers: {', '.join(p.key_papers)}")
        lines.append(f"  Unresolved: {p.unresolved_problem}")
        if p.tensions:
            lines.append("  Internal tensions:")
            for t in p.tensions:
                lines.append(f"    - {t.tension}: {t.description}")
        lines.append("")
    return "\n".join(lines)


def tensions_to_markdown(phases: list[Phase]) -> str:
    """Format all tensions from phases as markdown."""
    all_tensions: list[Tension] = []
    for p in phases:
        all_tensions.extend(p.tensions)
    lines = ["| Tension | Phase | Status |", "|---------|-------|--------|"]
    for t in all_tensions:
        status = t.status or "open"
        lines.append(f"| {t.tension} | ? | {status} |")
    return "\n".join(lines)


def phases_to_markdown(phases: list[Phase]) -> str:
    """Format phases as markdown table."""
    lines = ["| Phase | Time | Key Papers |",
             "|-------|------|------------|"]
    for p in phases:
        papers = ", ".join(p.key_papers[:3])
        if len(p.key_papers) > 3:
            papers += f" (+{len(p.key_papers) - 3})"
        lines.append(f"| {p.name} | {p.time_range} | {papers} |")
    return "\n".join(lines)
