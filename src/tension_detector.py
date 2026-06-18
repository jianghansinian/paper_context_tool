"""Tension detector — two-stage: detect all tensions, then merge into phases.

V8 architecture:
  Stage 1: detect_all_tensions() → 8-12 fine-grained tensions (no limit)
  Stage 2: merge_tensions_into_phases() → 2-4 phases (time+theme clustering)

Phase is the V8 narrative chapter unit. It emerges from tension clustering,
not a separate entity. Each phase has a causal chain link to the next.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _resolve_model
from paper import Claim, Tension, Phase


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

Identify ALL research tensions in this field's evolution. Do NOT limit the number —
output every tension you can identify (typically 8-12).

A tension has:
- introducers: papers that made the tension visible or acute
- resolvers: papers that advanced a direction or favored one side
- status: "direction_clear" (community converged on a dominant approach), \
  "direction_forming" (one approach gaining traction but not settled), \
  or "open" (no clear winner, actively debated)
- domain_scope: which sub-area this tension applies to \
  (e.g. "in detection/tracking", "in occupancy/world modeling", "in planning")

CRITICAL — Status semantics:
- "direction_clear" means the community FAVORS one direction, NOT that the tension \
  is permanently resolved. New evidence could reopen it. Be specific about the SCOPE \
  of agreement (which sub-area, which benchmark, which setting).
- "direction_forming" means competing approaches exist but one is gaining traction. \
  Name which direction is emerging.
- "open" means actively debated with no clear winner or the community is split.
- Avoid claiming anything is "resolved" — scientific tensions are rarely closed; \
  they evolve in scope and may re-emerge in new contexts.

IMPORTANT:
- Each paper may appear in multiple tensions (as introducer for one, resolver for another)
- Focus on tensions that changed the field's direction — not minor improvements
- Tensions should connect across papers: "Paper A revealed problem P, Papers B and C tried to solve it"
- For each tension, state its DOMAIN SCOPE explicitly

Return JSON:
```json
{{
  "tensions": [
    {{
      "tension": "string (short name, e.g. 'Depth reliability gap')",
      "description": "string (2-3 sentences: what was the contradiction, why did it matter, how was it addressed)",
      "introduced_by": ["paper_title", ...],
      "resolved_by": ["paper_title", ...],
      "status": "direction_clear|direction_forming|open",
      "dimension": "representation|geometry|system|evaluation",
      "domain_scope": "string (e.g. 'in detection/tracking on nuScenes benchmark')"
    }}
  ]
}}
```"""


# ── Public API ────────────────────────────────────────────────────────

def detect_all_tensions(
    claims: list[Claim],
    relations: list,  # list[ClaimRelation] or list[dict]
    field_name: str = "",
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Optional[list[Tension]]:
    """Stage 1: Detect ALL research tensions — no limit (typically 8-12).

    Args:
        claims: All claims (flat list, chronologically ordered)
        relations: ClaimRelation objects from claim_relation_builder
        field_name: Field name
        client: LLM client
        model: Model override

    Returns:
        List of Tension objects. None on failure.
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

    prompt = _TENSION_PROMPT.format(
        field_name=field_name or "this field",
        claims_text=claims_text,
        relations_text=relations_text,
        phases_text="(auto-detected from claim chronology)",
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

        # Parse with fallback: try direct JSON, then regex extraction
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise
            raw_extracted = m.group(0)
            # Try to fix unterminated strings by closing them
            raw_extracted = re.sub(r'(?<!\\)"((?:[^"\\]|\\.)*)$', r'"\1"', raw_extracted)
            try:
                data = json.loads(raw_extracted)
            except json.JSONDecodeError:
                # Last resort: try to close all open structures
                raw_extracted = _repair_truncated_json(m.group(0))
                data = json.loads(raw_extracted)

        raw_tensions = data.get("tensions", [])
        if raw_tensions and isinstance(raw_tensions, list):
            return [
                Tension(
                    tension=t.get("tension", ""),
                    description=t.get("description", ""),
                    introduced_by=t.get("introduced_by", []),
                    resolved_by=t.get("resolved_by", []),
                    status=t.get("status", "open"),
                    dimension=t.get("dimension", "system"),
                    domain_scope=t.get("domain_scope", ""),
                )
                for t in raw_tensions
            ]
    except Exception as exc:
        print(f"Tension detection failed: {exc}")

    return None


def _repair_truncated_json(raw: str) -> str:
    """Attempt to repair truncated/incomplete JSON by closing open structures."""
    open_braces = raw.count('{') - raw.count('}')
    open_brackets = raw.count('[') - raw.count(']')
    repaired = raw.rstrip()
    repaired = repaired.rstrip(',')
    repaired += ']' * open_brackets
    repaired += '}' * open_braces
    return repaired


# ── Stage 2: Merge tensions into phases ───────────────────────────────

_PHASE_MERGE_SYSTEM = """\
You are a research historian who organizes technical evolution into coherent \
time periods. Each Phase is a chapter in the story of how a field evolved.

A Phase is a TIME PERIOD with a CORE CONTRADICTION. It emerges from clustering \
related tensions that were active at the same time. Phases are linked by CAUSAL \
CHAINS: Phase N's unsolved problem becomes Phase N+1's motivation.

Key constraints:
- 2-4 phases total (merge overlapping tensions into the same phase)
- Each phase covers a distinct time range (phases are sequential, not parallel)
- Phase names should be memorable: "Dense BEV Era", "Sparse Revolution", etc.
- The causal chain MUST be explicit: each phase's unresolved_problem MUST be \
  the seed of the next phase's core_contradiction

Return ONLY a JSON object. No other text."""

_PHASE_MERGE_PROMPT = """\
Merge the following research tensions into 2-4 chronological Phases.

FIELD: {field_name}

ALL TENSIONS (detected from claims, fine-grained):
{tensions_text}

PAPERS (chronological, for time reference):
{papers_text}

---
INSTRUCTIONS:

1. Group tensions by TIME OVERLAP first, then by THEME similarity.
   - Tensions active in the same period → same Phase
   - Tensions about similar topics (e.g. depth, representation) → same Phase
   - A tension may span multiple phases if it evolved over time

2. For each Phase, identify:
   - name: Question-driven title (e.g. "How to Build a 3D View from 2D Images?" \
     not "Dense BEV Era"). Use the form "{{How/Can/Should/What}} ... ?" when possible.
   - dominant_question: 1 SENTENCE — the EXACT question that drove this phase. \
     This is a STRUCTURAL ANCHOR: it defines the phase independent of any particular \
     paper or tension. A reader should understand what this phase was about from \
     just this question. Be precise and scoped.
   - time_range: e.g. "2020-2022"
   - core_contradiction: 1 SENTENCE — the CENTRAL contradiction (keep under 100 chars)
   - key_papers: 3-5 papers most central to this phase (from paper list)
   - core_debate: 1 SENTENCE — What competing answers existed (keep under 80 chars)
   - unresolved_problem: 1 SENTENCE — What this phase COULD NOT solve (keep under 100 chars)

3. CRITICAL — Causal chain:
   - Phase 1's unresolved_problem must logically lead to Phase 2's core_contradiction
   - Phase 2's unresolved_problem → Phase 3, etc.
   - The story must flow: "They solved X, but that created problem Y..."

4. CRITICAL — Paper distribution:
   - Each paper appears PRIMARILY in ONE phase (its main contribution's phase)
   - Avoid the same paper dominating multiple phases
   - A paper may be mentioned in an adjacent phase if bridging
   - DISTRIBUTE PAPERS EVENLY: each phase should have 2-5 key papers. \
     Avoid putting 7 papers in one phase and 1 in another.
   - If a phase only has 1 natural paper, merge it with the adjacent phase. \
     A phase with 2 papers is acceptable if they form a coherent narrative arc.

Return JSON:
```json
{{
  "phases": [
    {{
      "name": "How to Build a 3D View from 2D Images?",
      "dominant_question": "How can 2D image features be reliably projected into 3D space for autonomous driving perception without relying on expensive depth labels?",
      "time_range": "2020-2022",
      "core_contradiction": "Camera-to-BEV needs depth but dense projection is expensive and depth labels constrain backbones",
      "key_papers": ["LSS: Lift-Splat-Shoot", "BEVDet: High-Performance Multi-Camera 3D Object Detection", "BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection"],
      "core_debate": "Is explicit depth supervision needed for accurate BEV perception?",
      "unresolved_problem": "Dense BEV grids waste computation on empty space and require complex view transformations",
      "tensions": ["depth-reliability-gap", "dense-computation-cost"]
    }}
  ]
}}
```"""


def merge_tensions_into_phases(
    tensions: list[Tension],
    claims: list[Claim],
    field_name: str = "",
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Optional[list[Phase]]:
    """Stage 2: Merge fine-grained tensions into 2-4 chronological Phases.

    Args:
        tensions: All tensions from Stage 1
        claims: All claims (for paper chronology reference)
        field_name: Field name
        client: LLM client
        model: Model override

    Returns:
        List of Phase objects (2-4, chronologically ordered). None on failure.
    """
    if not tensions:
        return None

    if client is None:
        from llm_analyzer import build_analyzer_client
        client = build_analyzer_client()
    if not client:
        return None

    model = _resolve_model(model)
    if not model:
        return None

    # Format tensions
    tensions_lines = []
    for i, t in enumerate(tensions, 1):
        tensions_lines.append(
            f"T{i}. [{t.dimension}] {t.tension}\n"
            f"    Description: {t.description}\n"
            f"    Introduced by: {', '.join(t.introduced_by)}\n"
            f"    Advanced by: {', '.join(t.resolved_by)}\n"
            f"    Status: {t.status}\n"
            f"    Domain: {t.domain_scope}"
        )
    tensions_text = "\n\n".join(tensions_lines)

    # Format papers (chronological, for time reference)
    claims_by_paper = {}
    for c in claims:
        if c.paper_title not in claims_by_paper:
            claims_by_paper[c.paper_title] = c.year
    papers_sorted = sorted(claims_by_paper.items(), key=lambda x: x[1])
    papers_lines = [f"[{y}] {t}" for t, y in papers_sorted]
    papers_text = "\n".join(papers_lines)

    prompt = _PHASE_MERGE_PROMPT.format(
        field_name=field_name or "this field",
        tensions_text=tensions_text,
        papers_text=papers_text,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _PHASE_MERGE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            m = re.search(r"\{.*\}", raw, re.DOTALL)
            if not m:
                raise
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                data = json.loads(_repair_truncated_json(m.group(0)))
        raw_phases = data.get("phases", [])
        if raw_phases and isinstance(raw_phases, list):
            phases = []
            for p in raw_phases:
                phase_tensions = []
                for tn in p.get("tensions", []):
                    # Match by tension name (short label)
                    matched = None
                    tn_lower = tn.lower().replace("-", " ").replace("_", " ")
                    for t in tensions:
                        t_lower = t.tension.lower().replace("-", " ").replace("_", " ")
                        if tn_lower in t_lower or t_lower in tn_lower:
                            matched = t
                            break
                    if matched:
                        phase_tensions.append(matched)
                phases.append(Phase(
                    name=p.get("name", ""),
                    time_range=p.get("time_range", ""),
                    dominant_question=p.get("dominant_question", ""),
                    core_contradiction=p.get("core_contradiction", ""),
                    key_papers=p.get("key_papers", []),
                    core_debate=p.get("core_debate", ""),
                    unresolved_problem=p.get("unresolved_problem", ""),
                    tensions=phase_tensions,
                ))
            return phases
    except Exception as exc:
        print(f"Phase merging failed: {exc}")

    return None


def phases_to_text(phases: list[Phase]) -> str:
    """Format phases for narrative prompt injection."""
    if not phases:
        return ""

    lines = ["NARRATIVE PHASES (time periods with causal chain):"]
    for i, p in enumerate(phases, 1):
        lines.append(f"\n  Phase {i}: {p.name} ({p.time_range})")
        lines.append(f"    Core contradiction: {p.core_contradiction}")
        lines.append(f"    Core debate: {p.core_debate}")
        lines.append(f"    Key papers: {', '.join(p.key_papers)}")
        lines.append(f"    Unresolved problem → Phase {i + 1}: {p.unresolved_problem}" if i < len(phases) else f"    Unresolved problem: {p.unresolved_problem}")
        if p.tensions:
            lines.append(f"    Tensions: {', '.join(t.tension for t in p.tensions)}")
    return "\n".join(lines)


def phases_to_markdown(phases: list[Phase]) -> str:
    """Render phases as markdown overview."""
    if not phases:
        return ""

    lines = [
        "## 技术发展阶段",
        "",
        "| 阶段 | 时间 | 核心矛盾 | 核心辩论 | 关键论文 | 遗留问题 |",
        "|------|------|----------|----------|----------|----------|",
    ]
    for p in phases:
        contradiction = p.core_contradiction if len(p.core_contradiction) <= 120 else p.core_contradiction[:117] + "..."
        debate = p.core_debate if len(p.core_debate) <= 80 else p.core_debate[:77] + "..."
        papers = ', '.join(p.key_papers[:3])
        if len(p.key_papers) > 3:
            papers += f" (+{len(p.key_papers) - 3})"
        unresolved = p.unresolved_problem if len(p.unresolved_problem) <= 80 else p.unresolved_problem[:77] + "..."
        lines.append(
            f"| **{p.name}** | {p.time_range} | {contradiction} | "
            f"{debate} | {papers} | "
            f"{unresolved} |"
        )
    return "\n".join(lines) + "\n"


def format_tensions_for_narrative(
    tensions: list[Tension],
    phase_name: str = "",
) -> str:
    """Format tensions as readable context for narrative prompt injection."""
    if not tensions:
        return ""

    lines = ["KEY RESEARCH TENSIONS (the contradictions that drove the field):"]
    for t in tensions:
        scope = f" [{t.domain_scope}]" if t.domain_scope else ""
        lines.append(
            f"  Tension [{t.dimension}]{scope}: {t.tension}"
        )
        lines.append(f"    {t.description}")
        lines.append(f"    Introduced by: {', '.join(t.introduced_by)}")
        lines.append(f"    Advanced by: {', '.join(t.resolved_by)}")
        lines.append(f"    Status: {t.status}")
    return "\n".join(lines)


def tensions_to_markdown(tensions: list[Tension]) -> str:
    """Render tensions as a markdown table section."""
    if not tensions:
        return ""

    lines = [
        "| 张力 | 适用域 | 描述 | 引入者 | 推进者 | 方向 |",
        "|------|--------|------|--------|--------|------|",
    ]

    status_labels = {
        "direction_clear": "✅ 方向明确",
        "direction_forming": "⚠️ 方向形成中",
        "open": "❌ 开放",
        "ongoing": "❌ 开放",
        "resolved": "✅ 方向明确",       # backward compat
        "partially_resolved": "⚠️ 方向形成中",  # backward compat
        "unresolved": "❌ 开放",          # backward compat
    }

    for t in tensions:
        status = status_labels.get(t.status, t.status)
        scope = t.domain_scope if t.domain_scope else "—"
        desc = t.description if len(t.description) <= 200 else t.description[:197] + "..."
        intro = ', '.join(t.introduced_by[:3])
        if len(t.introduced_by) > 3:
            intro += f" (+{len(t.introduced_by) - 3})"
        resolv = ', '.join(t.resolved_by[:3])
        if len(t.resolved_by) > 3:
            resolv += f" (+{len(t.resolved_by) - 3})"
        lines.append(
            f"| **{t.tension}** | {scope} | {desc} | "
            f"{intro} | "
            f"{resolv} | "
            f"{status} |"
        )

    return "\n".join(lines) + "\n"
