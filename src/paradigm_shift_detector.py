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
from paper import Claim, ParadigmShift


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
    relations: list,  # list[ClaimRelation] or list[dict]
    phases: list[dict],
    tensions: list | None = None,  # list[Tension] or list[dict]
    field_name: str = "BEV Perception",
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Optional[list[ParadigmShift]]:
    """Detect paradigm shifts from claims, relations, phases, and tensions.

    Returns list of ParadigmShift objects. None on failure.
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

    # Format relations (handles both ClaimRelation dataclass and legacy dict)
    if relations:
        rel_lines = []
        for r in relations:
            if hasattr(r, 'source_paper'):
                src, tgt, rel, expl = r.source_paper, r.target_paper, r.relation, r.explanation
            else:
                src, tgt, rel, expl = r['source_paper'], r['target_paper'], r['relation'], r['explanation']
            rel_lines.append(f"{src} → {tgt}: {rel.upper()} — {expl}")
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

    # Format tensions (handles both Tension dataclass and legacy dict)
    if tensions:
        t_lines = []
        for t in tensions:
            if hasattr(t, 'tension'):
                name, desc, status = t.tension, t.description, t.status
            else:
                name, desc, status = t.get('tension', '?'), t.get('description', ''), t.get('status', '?')
            t_lines.append(
                f"Tension: {name}\n"
                f"  {desc}\n"
                f"  Status: {status}"
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
        raw_shifts = data.get("paradigm_shifts", [])
        if raw_shifts and isinstance(raw_shifts, list):
            return [
                ParadigmShift(
                    shift_name=s.get("shift_name", ""),
                    description=s.get("description", ""),
                    old_paradigm=s.get("old_paradigm", ""),
                    new_paradigm=s.get("new_paradigm", ""),
                    catalyst_papers=s.get("catalyst_papers", []),
                    magnitude=s.get("magnitude", "incremental"),
                    level=s.get("level", "method"),
                    dimension=s.get("dimension", "system"),
                    year_range=s.get("year_range", ""),
                )
                for s in raw_shifts
            ]
    except Exception as exc:
        print(f"Paradigm shift detection failed: {exc}")

    return None


def format_shifts_for_narrative(
    shifts: list,  # list[ParadigmShift] or list[dict]
    phase_name: str = "",
) -> str:
    """Format paradigm shifts as readable context for narrative prompt injection.

    Only includes shifts relevant to the given phase (or all if phase_name is empty).
    """
    if not shifts:
        return ""

    relevant = shifts
    if phase_name:
        relevant = [s for s in shifts if (
            s.phase if hasattr(s, 'phase') else s.get("phase", "")
        ) == phase_name]

    if not relevant:
        return ""

    lines = [
        "PARADIGM SHIFTS (fundamental changes in the field's beliefs during this phase):",
        "These are the TURNING POINTS that the narrative must highlight. Each represents",
        "a moment when the field's understanding fundamentally changed.",
    ]
    for s in relevant:
        if hasattr(s, 'shift_name'):
            name, old, new, desc, cat, mag, lvl = (
                s.shift_name, s.old_paradigm, s.new_paradigm,
                s.description, s.catalyst_papers, s.magnitude, s.level
            )
        else:
            name = s['shift_name']
            old, new = s['old_paradigm'], s['new_paradigm']
            desc = s['description']
            cat = s.get('catalyst_papers', [])
            mag = s.get('magnitude', '?')
            lvl = s.get('level', '?')
        magnitude = mag.upper()
        level = lvl.replace("_", " ").upper()
        lines.append(f"\n  Shift: {name} [{magnitude}] [{level}]")
        lines.append(f"    From: {old}")
        lines.append(f"    To:   {new}")
        lines.append(f"    {desc}")
        lines.append(f"    Catalyst: {', '.join(cat)}")
    return "\n".join(lines)


def shifts_to_markdown(shifts: list) -> str:
    """Render paradigm shifts as a markdown section, grouped by dimension.

    Each dimension represents an independent evolution thread:
    - representation: dense grid → sparse query → vectorized
    - geometry: explicit depth → learned attention → hybrid
    - system: modular → unified → end-to-end
    - evaluation: accuracy → efficiency → safety

    Accepts list[ParadigmShift] or list[dict].
    """
    if not shifts:
        return ""

    def _get(s, key, default=""):
        if hasattr(s, key):
            return getattr(s, key, default)
        return s.get(key, default)

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

    grouped: dict[str, list] = {}
    for s in shifts:
        dim = _get(s, "dimension", "system")
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
            for idx, s in enumerate(dim_shifts):
                old = _get(s, "old_paradigm", "")[:55]
                new = _get(s, "new_paradigm", "")[:55]
                shift_name = _get(s, "shift_name", "")[:35]
                node_from = f'{dim}_old_{idx}'
                node_to = f'{dim}_new_{idx}'
                lines.append(f'    {node_from}["{old}"] -->|"{shift_name}"| {node_to}["{new}"]')
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
            name = _get(s, "shift_name", "")
            mag = _get(s, "magnitude", "")
            lvl = _get(s, "level", "?")
            period = _get(s, "year_range", "?")
            old = _get(s, "old_paradigm", "")
            new = _get(s, "new_paradigm", "")
            desc = _get(s, "description", "")
            cat = _get(s, "catalyst_papers", [])

            mag_label = magnitude_icons.get(mag, mag.upper())
            level_label = lvl.replace("_", " ").title()

            lines.append(f"**{name}** ({period})")
            lines.append(f"{mag_label} | {level_label}")
            lines.append("")
            lines.append(f"> **前**: *{old}*")
            lines.append("")
            lines.append(f"> **后**: *{new}*")
            lines.append("")
            lines.append(desc)
            lines.append("")

            if cat:
                lines.append(f"**代表论文**: {' · '.join(cat)}")
            lines.append("")

    return "\n".join(lines) + "\n"
