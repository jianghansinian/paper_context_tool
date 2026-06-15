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
Your expertise is distinguishing between TECHNIQUE EVOLUTION (methods improve but \
the field's core beliefs stay the same) and PARADIGM SHIFTS (the field's consensus \
about what problem matters and how to solve it fundamentally changes).

CRITICAL DISTINCTION — Paradigm Shift vs Technique Evolution:

TECHNIQUE EVOLUTION (NOT a paradigm shift):
- "Single-frame perception → temporal fusion" — The core belief ("dense BEV is \
  necessary") didn't change; researchers just added a new input dimension.
- "Depth from lidar → depth from stereo" — The belief ("explicit depth labels \
  are needed") didn't change; the source of depth labels changed.
- "CNN backbone → Transformer backbone" — Architecture substitution within the \
  same paradigm.
- "Better temporal fusion via recurrence" — Improving a known technique.
- Any improvement where the field's RESEARCH QUESTION and SUCCESS CRITERIA \
  remain the same.

PARADIGM SHIFT (return these):
- The field's CORE RESEARCH QUESTION changed (e.g., from "how to build a better \
  dense BEV grid" to "do we even need a BEV grid?")
- A fundamental ASSUMPTION was overturned (e.g., from "dense BEV grid is \
  necessary" to "sparse queries can replace it entirely")
- The DEFINITION OF SUCCESS changed (e.g., from "accuracy at any cost" to \
  "planning safety is the ultimate metric")
- Two previously separate approaches MERGED into a unified framework
- A once-promising direction was ABANDONED by the community

LITMUS TEST: Would a researcher from before this shift find the new approach \
UNINTELLIGIBLE or OBVIOUSLY WRONG without understanding the shift itself? \
If yes → paradigm shift. If they'd see it as a natural improvement → technique \
evolution.

You MUST return AT MOST 3 paradigm shifts. If you find more than 3, keep only \
the 3 most fundamental ones (those that changed the research question or core \
assumption, not the method or evaluation). Quality over quantity.

Return ONLY a JSON object. No other text."""

_PARADIGM_SHIFT_PROMPT = """\
Identify the PARADIGM SHIFTS in this field's evolution. A paradigm shift is when \
the field's fundamental assumptions, research questions, or success criteria changed \
— NOT when a method incrementally improved or a technique evolved.

FIELD: {field_name}

CLAIMS (chronological):
{claims_text}

CLAIM RELATIONS (how later claims relate to earlier ones):
{relations_text}

PHASES (LLM-identified paradigm eras):
{phases_text}

RESEARCH TENSIONS (field-level contradictions):
{tensions_text}

Identify AT MOST 3 paradigm shifts. A paradigm shift must pass the LITMUS TEST: \
would a researcher from before this shift find the new approach fundamentally \
wrong or unintelligible? If they'd see it as a natural improvement, it's technique \
evolution, not a paradigm shift.

EXAMPLES OF WHAT NOT TO INCLUDE (these are technique evolution):
- "Single-frame → Temporal fusion" — Adding temporal input doesn't change the \
  core paradigm; the field still believes in the same representation and goals.
- "Fixed backbone → Backbone-agnostic" — Engineering improvement, not a belief change.
- "O(T) temporal storage → O(1) recurrent fusion" — Algorithmic efficiency gain \
  within the same paradigm.

EXAMPLES OF WHAT TO INCLUDE:
- "Dense BEV grid necessary → Sparse queries sufficient" — Fundamental assumption \
  overturned about what representation is needed.
- "Modular pipeline → End-to-end planning-oriented system" — The definition of \
  success changed from per-task metrics to planning safety.
- "Explicit depth supervision essential → Learned geometry via attention" — Core \
  belief about what information is needed changed.

For each shift, identify:
- The old paradigm (what the field believed before)
- The new paradigm (what the field believed after)
- Catalyst papers (which papers triggered or crystallized this shift)
- Magnitude: use "paradigm_shift" only for fundamental assumption overturns. \
  Use "optimization" for substantial efficiency gains that changed SOTA expectations \
  but not core beliefs. Use "convergence" for merging of separate paradigms.
- Level (research_question, method, evaluation)
- Dimension: representation, geometry, system, evaluation

MAGNITUDE GUIDE:
- paradigm_shift: A core assumption was overturned; the field cannot go back
- optimization: Same paradigm, substantially more efficient implementation
- convergence: Two separate approaches merged into one framework

YOUR RESPONSE MUST CONTAIN AT MOST 3 SHIFTS. If tempted to add more, keep only \
the most fundamental ones.

Return JSON:
```json
{{
  "paradigm_shifts": [
    {{
      "shift_name": "string (short label, e.g. 'Dense BEV → Sparse Representation')",
      "description": "string (2-3 sentences: what changed, why it mattered, how the field was different after)",
      "old_paradigm": "string (what the field believed before)",
      "new_paradigm": "string (what the field believed after)",
      "catalyst_papers": ["paper_title", ...],
      "magnitude": "paradigm_shift|optimization|convergence",
      "level": "research_question|method|evaluation",
      "dimension": "representation|geometry|system|evaluation",
      "phase": "string (which LLM-identified phase this shift belongs to)",
      "year_range": "string (e.g. '2022-2023')"
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
