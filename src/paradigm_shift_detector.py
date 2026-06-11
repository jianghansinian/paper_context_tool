"""Paradigm shift detector — identifies when a field's fundamental assumptions changed.

Core V4 module. A Paradigm Shift is when the research community's COLLECTIVE BELIEF
about what problem matters and how to solve it fundamentally changes — not when
someone proposes a better method within the same belief system.

Key distinction (GPT comment insight):
  Method change:   "BEVDepth uses explicit depth supervision instead of implicit"
  Paradigm shift:  "The field stopped asking 'how to fuse cameras' and started
                    asking 'how to reduce cost'"

  Method change:   "BEVFormer uses temporal attention for feature aggregation"
  Paradigm shift:  "The field realized depth quality isn't the bottleneck —
                    temporal modeling determines performance"

Three levels of shift:
  research_question — What problem the field considers important changed
  method            — The fundamental approach category changed (not just a
                      better variant within the same category)
  evaluation        — What "success" means changed (new metrics, new priorities)

Magnitude classification:
  paradigm_shift — Fundamental assumption overturned; the field cannot go back
  optimization   — Same paradigm, but substantially better implementation that
                   changed what the field considers SOTA
  incremental    — Improvement within established paradigm, no belief change
  convergence    — Two previously separate paradigms merged into one
  dead_end       — Paradigm was pursued but abandoned due to insurmountable issues
"""
from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _resolve_model
from paper import Claim


# ── System prompt ─────────────────────────────────────────────────────
_PARADIGM_SHIFT_SYSTEM = """\
You are a research historian who identifies PARADIGM SHIFTS in a field's evolution. \
Your expertise is distinguishing between incremental method improvements and genuine \
changes in what the research community believes.

A paradigm shift is NOT:
- "Paper B uses attention instead of convolution" (method substitution)
- "Paper B achieves +3% NDS over Paper A" (incremental improvement)
- "Paper B proposes a novel module" (architectural novelty)

A paradigm shift IS:
- The field's CORE RESEARCH QUESTION changed (e.g., from "how to fuse cameras" \
  to "how to reduce computational cost")
- A fundamental ASSUMPTION was overturned (e.g., from "dense BEV is necessary" \
  to "sparse queries suffice")
- The DEFINITION OF SUCCESS changed (e.g., from "accuracy at any cost" to \
  "accuracy per FLOP")
- Two previously separate approaches MERGED into a unified framework
- A once-promising direction was ABANDONED by the community

To detect a paradigm shift, ask:
1. After this paper, did researchers ask DIFFERENT QUESTIONS?
2. Did this paper change what the field considers "obvious" or "given"?
3. Did the evaluation criteria fundamentally change after this work?
4. Would a researcher from before this shift find the new approach unintelligible
   without understanding the shift itself?

Return ONLY a JSON object. No other text."""

_PARADIGM_SHIFT_PROMPT = """\
Identify the paradigm shifts in this field's evolution. A paradigm shift is when \
the field's fundamental assumptions, research questions, or success criteria changed \
— NOT when a method incrementally improved.

FIELD: {field_name}

CLAIMS (chronological):
{claims_text}

CLAIM RELATIONS (how later claims relate to earlier ones):
{relations_text}

PHASES (LLM-identified paradigm eras):
{phases_text}

RESEARCH TENSIONS (field-level contradictions):
{tensions_text}

Identify 3-6 paradigm shifts. Each shift should represent a moment when the field's \
understanding fundamentally changed. Focus on shifts in the CLAIM SPACE — what \
researchers believed to be true — not just changes in method names.

IMPORTANT DISTINCTIONS:
- "Better depth estimation" within the same paradigm → NOT a shift
- "Depth doesn't need to be explicit; attention can learn geometry" → IS a shift \
  (assumption overturned)
- "Sparse queries can replace dense BEV grids" → IS a shift (research question changed \
  from "how to build better BEV" to "do we even need BEV?")
- "Planning and perception should be jointly optimized" → IS a shift (evaluation \
  criterion changed from task-specific metrics to open-loop planning safety)

For each shift, identify:
- The old paradigm (what the field believed before)
- The new paradigm (what the field believed after)
- Catalyst papers (which papers triggered or crystallized this shift)
- Magnitude (paradigm_shift, optimization, incremental, convergence, dead_end)
- Level (research_question, method, evaluation)
- Dimension: which ASPECT of the system this shift belongs to:
  - "representation" — how the scene/world is represented (dense grid, sparse query, vectorized)
  - "geometry" — how 3D geometry/depth is inferred (explicit depth, learned attention, hybrid)
  - "system" — how the pipeline is architected (modular, unified, end-to-end)
  - "evaluation" — what metrics/standards define success (accuracy, efficiency, safety)

MAGNITUDE GUIDE:
- paradigm_shift: A core assumption was overturned; the field cannot go back
- optimization: Same paradigm, but a substantially more efficient/effective \
  implementation that changed SOTA expectations
- convergence: Two previously separate approaches merged into one framework
- dead_end: A direction the field pursued but largely abandoned

Return JSON:
```json
{{
  "paradigm_shifts": [
    {{
      "shift_name": "string (short label, e.g. 'Explicit Depth → Learned Geometry')",
      "description": "string (2-3 sentences: what changed, why it mattered, how the field was different after)",
      "old_paradigm": "string (what the field believed before)",
      "new_paradigm": "string (what the field believed after)",
      "catalyst_papers": ["paper_title", ...],
      "magnitude": "paradigm_shift|optimization|convergence|dead_end",
      "level": "research_question|method|evaluation",
      "dimension": "representation|geometry|system|evaluation",
      "phase": "string (which LLM-identified phase this shift belongs to)",
      "year_range": "string (e.g. '2022-2022')"
    }}
  ]
}}
```"""


# ── Public API ────────────────────────────────────────────────────────

def detect_paradigm_shifts(
    claims: list[Claim],
    relations: list[dict],
    phases: list[dict],
    tensions: list[dict] | None = None,
    field_name: str = "BEV Perception",
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Optional[list[dict]]:
    """Detect paradigm shifts from claims, relations, phases, and tensions.

    Args:
        claims: All claims (flat list, chronologically ordered)
        relations: Claim relations from claim_relation_builder
        phases: Phase dicts from propose_structure
        tensions: Research tensions from tension_detector (optional)
        field_name: Field name
        client: LLM client
        model: Model override

    Returns:
        List of paradigm shift dicts, each with: shift_name, description,
        old_paradigm, new_paradigm, catalyst_papers, magnitude, level, phase, year_range.
        None on failure.
    """
    if not claims:
        return None

    if client is None:
        from llm_analyzer import build_analyzer_client
        client = build_analyzer_client()
    if not client:
        return None

    model = _resolve_model(model)
    if not model:
        return None

    # Format claims
    claims_sorted = sorted(claims, key=lambda c: c.year)
    claims_lines = []
    for c in claims_sorted:
        claims_lines.append(
            f"[{c.year}] {c.paper_title}\n"
            f"  Claim: {c.statement}\n"
            f"  Problem: {c.problem_addressed}\n"
            f"  Type: {c.claim_type}"
        )
    claims_text = "\n\n".join(claims_lines)

    # Format relations
    if relations:
        rel_lines = []
        for r in relations:
            rel_lines.append(
                f"{r['source_paper']} → {r['target_paper']}: "
                f"{r['relation'].upper()} — {r['explanation']}"
            )
        relations_text = "\n".join(rel_lines)
    else:
        relations_text = "(no relations available)"

    # Format phases
    phases_lines = []
    for p in phases:
        phases_lines.append(
            f"Phase: {p['name']} ({p.get('time_range', '?')})\n"
            f"  Paradigm: {p.get('core_paradigm', '?')}\n"
            f"  Papers: {', '.join(p.get('paper_arxiv_ids', []))}"
        )
    phases_text = "\n".join(phases_lines)

    # Format tensions
    if tensions:
        t_lines = []
        for t in tensions:
            t_lines.append(
                f"Tension: {t.get('tension', '?')}\n"
                f"  {t.get('description', '')}\n"
                f"  Status: {t.get('status', '?')}"
            )
        tensions_text = "\n".join(t_lines)
    else:
        tensions_text = "(no tensions available)"

    prompt = _PARADIGM_SHIFT_PROMPT.format(
        field_name=field_name,
        claims_text=claims_text,
        relations_text=relations_text,
        phases_text=phases_text,
        tensions_text=tensions_text,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _PARADIGM_SHIFT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=3072,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        shifts = data.get("paradigm_shifts", [])
        if shifts and isinstance(shifts, list):
            return shifts
    except Exception as exc:
        print(f"Paradigm shift detection failed: {exc}")

    return None


def format_shifts_for_narrative(
    shifts: list[dict],
    phase_name: str = "",
) -> str:
    """Format paradigm shifts as readable context for narrative prompt injection.

    Only includes shifts relevant to the given phase (or all if phase_name is empty).
    """
    if not shifts:
        return ""

    relevant = shifts
    if phase_name:
        relevant = [s for s in shifts if s.get("phase", "") == phase_name]

    if not relevant:
        return ""

    lines = [
        "PARADIGM SHIFTS (fundamental changes in the field's beliefs during this phase):",
        "These are the TURNING POINTS that the narrative must highlight. Each represents",
        "a moment when the field's understanding fundamentally changed.",
    ]
    for s in relevant:
        magnitude = s.get("magnitude", "?").upper()
        level = s.get("level", "?").replace("_", " ").upper()
        lines.append(f"\n  Shift: {s['shift_name']} [{magnitude}] [{level}]")
        lines.append(f"    From: {s['old_paradigm']}")
        lines.append(f"    To:   {s['new_paradigm']}")
        lines.append(f"    {s['description']}")
        lines.append(f"    Catalyst: {', '.join(s.get('catalyst_papers', []))}")
    return "\n".join(lines)


def shifts_to_markdown(shifts: list[dict]) -> str:
    """Render paradigm shifts as a markdown section, grouped by dimension.

    Each dimension represents an independent evolution thread:
    - representation: dense grid → sparse query → vectorized
    - geometry: explicit depth → learned attention → hybrid
    - system: modular → unified → end-to-end
    - evaluation: accuracy → efficiency → safety
    """
    if not shifts:
        return ""

    # Group shifts by dimension
    dimension_order = ["representation", "geometry", "system", "evaluation"]
    dimension_labels = {
        "representation": "表示范式 (Representation)",
        "geometry": "几何范式 (Geometry)",
        "system": "系统范式 (System)",
        "evaluation": "评估范式 (Evaluation)",
    }
    dimension_icons = {
        "representation": "🗺️",
        "geometry": "📐",
        "system": "⚙️",
        "evaluation": "📊",
    }

    grouped: dict[str, list[dict]] = {}
    for s in shifts:
        dim = s.get("dimension", "system")
        if dim not in grouped:
            grouped[dim] = []
        grouped[dim].append(s)

    magnitude_icons = {
        "paradigm_shift": "🔴 PARADIGM SHIFT",
        "optimization": "🟡 OPTIMIZATION",
        "convergence": "🔵 CONVERGENCE",
        "dead_end": "⚫ DEAD END",
        "incremental": "⚪ INCREMENTAL",
    }

    lines = [
        "## 核心范式转移",
        "",
        "> 技术发展史的本质是范式的更替。以下按维度分组展示该领域经历的根本性信念转变，",
        "> 每个维度是一条独立的思想演化线。",
        "",
    ]

    # Render Mermaid overview diagram of all dimensions
    lines.append("### 范式演化全景")
    lines.append("")
    lines.append("```mermaid")
    lines.append("graph LR")
    for dim in dimension_order:
        if dim in grouped:
            dim_shifts = grouped[dim]
            label = dimension_labels.get(dim, dim)
            # Show the chain: old → new for each shift in this dimension
            prev_node = None
            for s in dim_shifts:
                old = s['old_paradigm'][:30]
                new = s['new_paradigm'][:30]
                node_from = f'{dim}_old_{dim_shifts.index(s)}'
                node_to = f'{dim}_new_{dim_shifts.index(s)}'
                lines.append(f'    {node_from}["{old}"] -->|"{s["shift_name"][:25]}"| {node_to}["{new}"]')
    lines.append("```")
    lines.append("")

    # Render each dimension as a section
    for dim in dimension_order:
        if dim not in grouped:
            continue

        dim_shifts = grouped[dim]
        icon = dimension_icons.get(dim, "")
        label = dimension_labels.get(dim, dim)
        lines.append(f"### {icon} {label}")
        lines.append("")

        for s in dim_shifts:
            mag = magnitude_icons.get(s.get("magnitude", ""), s.get("magnitude", "?").upper())
            level = s.get("level", "?").replace("_", " ").title()
            period = s.get("year_range", "?")

            lines.append(f"**{s['shift_name']}** ({period})")
            lines.append(f"{mag} | {level}")
            lines.append("")
            lines.append(f"> **前**: *{s['old_paradigm']}*")
            lines.append("")
            lines.append(f"> **后**: *{s['new_paradigm']}*")
            lines.append("")
            lines.append(s["description"])
            lines.append("")

            catalysts = s.get("catalyst_papers", [])
            if catalysts:
                lines.append(f"**代表论文**: {' · '.join(catalysts)}")
            lines.append("")

    return "\n".join(lines) + "\n"
