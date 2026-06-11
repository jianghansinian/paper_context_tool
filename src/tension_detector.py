"""Tension detector — identifies the contradictions that drive field evolution.

Core V4 module. A Research Tension is a field-level contradiction, unsolved
problem, or trade-off that the research community wrestled with. Tensions are
the "why" behind claim relations — they are what make an evolution story
compelling.

Key distinction:
  Method limitation: "LSS depth estimation is coarse" (one paper's weakness)
  Research tension:  "Camera-only depth without LiDAR is unreliable" (field
                      problem that multiple papers grapple with)

A tension has introducers (papers that made it visible) and resolvers (papers
that addressed it). Unresolved tensions point to future research directions.
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
_TENSION_SYSTEM = """\
You are a research meta-analyst who identifies the KEY TENSIONS that drove \
a field's evolution. You understand that the most memorable part of any \
technical history is not the papers — it's the contradictions they wrestled with.

A Research Tension is a field-level contradiction, unsolved problem, or \
fundamental trade-off. It is NOT a single paper's method limitation.

Good tension: "Dense BEV grids cost O(H*W), making edge deployment impractical"
Bad tension:  "LSS produces blurry depth features" (too specific to one method)

Good tension: "Modular perception-prediction-planning pipelines accumulate errors"
Bad tension:  "UniAD's inference is slow" (one paper's engineering issue)

A tension must:
1. Involve at least two papers (introducers + resolvers from different papers)
2. Be at the level the FIELD cares about, not one lab's engineering problem
3. Have a clear contradiction: "the field wanted X but Y prevented it"
4. Be falsifiable — you could imagine a different resolution

Return ONLY a JSON object. No other text."""

_TENSION_PROMPT = """\
Identify the key research tensions in this field's evolution.

FIELD: {field_name}

CLAIMS (chronological):
{claims_text}

CLAIM RELATIONS (how later claims relate to earlier ones):
{relations_text}

PHASES:
{phases_text}

Identify 5-8 research tensions. Each tension should capture a contradiction \
or problem that the FIELD wrestled with — not one paper's specific limitation.

A tension has:
- introducers: papers that made the tension visible or acute
- resolvers: papers that addressed or resolved it
- status: "resolved" (definitively addressed), "partially_resolved" (improved \
  but not fully), or "unresolved" (still open)

IMPORTANT:
- Each paper may appear in multiple tensions (as introducer for one, resolver for another)
- Focus on tensions that changed the field's direction — not minor improvements
- Tensions should connect across papers: "Paper A revealed problem P, Papers B and C tried to solve it"

Return JSON:
```json
{{
  "tensions": [
    {{
      "tension": "string (short name, e.g. 'Depth reliability gap')",
      "description": "string (2-3 sentences: what was the contradiction, why did it matter, how was it addressed)",
      "introduced_by": ["paper_title", ...],
      "resolved_by": ["paper_title", ...],
      "status": "resolved|partially_resolved|unresolved"
    }}
  ]
}}
```"""


# ── Public API ────────────────────────────────────────────────────────

def detect_tensions(
    claims: list[Claim],
    relations: list[dict],
    phases: list[dict],
    field_name: str = "BEV Perception",
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Optional[list[dict]]:
    """Detect research tensions from claims, relations, and phase structure.

    Args:
        claims: All claims (flat list, chronologically ordered)
        relations: Claim relations from claim_relation_builder
        phases: Phase dicts from propose_structure
        field_name: Field name
        client: LLM client
        model: Model override

    Returns:
        List of tension dicts, each with: tension, description, introduced_by,
        resolved_by, status. None on failure.
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
            f"  Problem: {c.problem_addressed}"
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

    prompt = _TENSION_PROMPT.format(
        field_name=field_name,
        claims_text=claims_text,
        relations_text=relations_text,
        phases_text=phases_text,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _TENSION_SYSTEM},
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
        tensions = data.get("tensions", [])
        if tensions and isinstance(tensions, list):
            return tensions
    except Exception as exc:
        print(f"Tension detection failed: {exc}")

    return None


def format_tensions_for_narrative(
    tensions: list[dict],
    phase_name: str = "",
) -> str:
    """Format tensions as readable context for narrative prompt injection.

    Only includes tensions relevant to the given phase (or all if phase_name is empty).
    """
    if not tensions:
        return ""

    lines = ["KEY RESEARCH TENSIONS (the contradictions that drove this phase):"]
    for t in tensions:
        lines.append(
            f"  Tension: {t['tension']}"
        )
        lines.append(f"    {t['description']}")
        lines.append(f"    Introduced by: {', '.join(t.get('introduced_by', []))}")
        lines.append(f"    Resolved by: {', '.join(t.get('resolved_by', []))}")
        lines.append(f"    Status: {t.get('status', 'unknown')}")
    return "\n".join(lines)


def tensions_to_markdown(tensions: list[dict]) -> str:
    """Render tensions as a markdown section."""
    if not tensions:
        return ""

    lines = [
        "### 核心研究张力",
        "",
        "| 张力 | 描述 | 引入者 | 解决者 | 状态 |",
        "|------|------|--------|--------|------|",
    ]

    status_icons = {
        "resolved": "✅ resolved",
        "partially_resolved": "⚠️ partial",
        "unresolved": "❌ open",
    }

    for t in tensions:
        status = status_icons.get(t.get("status", ""), t.get("status", "?"))
        lines.append(
            f"| **{t['tension']}** | {t['description'][:150]} | "
            f"{', '.join(t.get('introduced_by', []))[:80]} | "
            f"{', '.join(t.get('resolved_by', []))[:80]} | "
            f"{status} |"
        )

    return "\n".join(lines) + "\n"
