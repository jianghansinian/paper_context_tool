"""Problem-driven phase detector — single LLM call from problem_addressed to phases.

Replaces the 2-stage V8 pipeline (detect_all_tensions → merge_tensions_into_phases)
with a single LLM call that groups papers by the PROBLEM they solve.

Architecture:
    claims[problem_addressed] → [one LLM call] → Phase[]
                                                      ├── name, dominant_question
                                                      ├── key_papers, time_range
                                                      ├── tensions[] (internal, 1-3 per phase)
                                                      ├── core_contradiction, core_debate
                                                      └── unresolved_problem (→ next phase seed)

Stability comes from: (1) single LLM call instead of two, (2) grouping by
explicit problem_addressed field rather than LLM-invented tension abstractions,
(3) problem semantic similarity is more stable than tension clustering.
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
_PROBLEM_PHASE_SYSTEM = """\
You are a research historian who organizes the evolution of a technical field \
into coherent time periods. Each period (Phase) is defined by a CORE PROBLEM \
that the community was collectively trying to solve.

CRITICAL PRINCIPLE — Problem-driven grouping:
- Group papers by the PROBLEM they solve, NOT by the method they use.
- Papers with opposite approaches to the SAME problem belong in the SAME phase.
- Papers that solve DIFFERENT problems belong in DIFFERENT phases, even if \
  they use similar methods.

EXAMPLE of same problem (→ same phase):
  Problem: "How to construct an accurate BEV from multi-camera images?"
  Papers: LSS (implicit depth), BEVDepth (explicit depth supervision), \
  BEVFormer (attention-based, no depth)
  → All in one phase because they all address "how to build BEV."

EXAMPLE of different problems (→ different phases):
  Problem A: "How to construct BEV from cameras?" (construction)
  Problem B: "How to make BEV efficient for real-time deployment?" (efficiency)
  → Different phases — the core problem changed.

A Phase is a CHAPTER in the story. It has:
- A clear PROBLEM that unifies its papers
- A TIME RANGE from earliest to latest paper
- A DOMINANT QUESTION that defines what the field was trying to answer
- Internal TENSIONS (contradictions/debates between papers within this phase)
- An UNRESOLVED PROBLEM that logically leads to the next phase

Return ONLY a JSON object. No other text."""


# ── Per-field prompt ──────────────────────────────────────────────────
_PROBLEM_PHASE_PROMPT = """\
Organize the following papers into 4-6 chronological Phases based on the \
PROBLEM each paper addresses.

FIELD: {field_name}

PAPERS (chronological, with problems they address):
{papers_text}

---
INSTRUCTIONS:

STEP 1 — Identify 4-6 distinct PROBLEM CLUSTERS:
- Read through all papers' problems chronologically.
- Ask: "What changed in the PROBLEM the community was solving?"
- When the core problem shifts, a new phase begins.
- A phase must have 2-6 papers (merge adjacent singleton phases).
- 4-6 phases total. Fewer than 4 loses important transitions; more than 6 \
  fragments the story.

STEP 2 — For each Phase, determine:
- name: A question-driven title, preferably in the form "How/Can/Should/What ... ?"
  Examples: "How to Build a 3D View from 2D Images?" \
  "Can Sparse Representations Match Dense Performance?"
- dominant_question: 1 SENTENCE — the precise question driving this phase.
  This is the STRUCTURAL ANCHOR. A reader should understand the phase from \
  this question alone.
- time_range: "YYYY-MM—YYYY-MM" using the exact dates of the earliest and \
  latest key_papers in this phase.
- key_papers: 3-6 papers most central to this phase. Every paper MUST appear \
  in exactly ONE phase as a key_paper.
- core_contradiction: 1 SENTENCE (under 100 chars). The central contradiction \
  that made this problem hard.
- core_debate: 1 SENTENCE (under 80 chars). What competing answers existed?
- status: "direction_clear" (community converged), "direction_forming" \
  (emerging consensus), or "open" (actively debated).
- tensions: 1-3 internal tensions within this phase. Each tension is a \
  SPECIFIC debate between papers in THIS phase. Must involve at least 2 papers.

STEP 3 — Build the CAUSAL CHAIN:
- Phase N's unresolved_problem MUST logically seed Phase N+1's problem.
- Example: Phase 1 unresolved "Dense BEV grids waste computation" →
  Phase 2 problem "How to represent the scene sparsely?"
- The story must flow: "They solved X, but that created problem Y..."

STEP 4 — For each internal tension, specify:
- tension: Short label (e.g. "Implicit depth is simpler but less reliable")
- description: 1-2 sentences describing the contradiction
- introduced_by: [paper_titles] — papers that made this tension visible
- resolved_by: [paper_titles] — papers that advanced one side
- status: "direction_clear" | "direction_forming" | "open"
- dimension: "representation" | "geometry" | "system" | "evaluation"
- domain_scope: What setting/benchmark this applies to

CRITICAL RULES:
1. Group by PROBLEM, not method. Opposite approaches to the same problem \
   belong together.
2. Papers from the same lineage (e.g., Sparse4D and Sparse4D v2) should \
   usually be in the SAME phase — they solve the same problem.
3. time_range MUST cover ALL key_papers in the phase.
4. Each paper appears as a key_paper in EXACTLY ONE phase.
5. unresolved_problem must be a CONCRETE technical problem, not vague. \
   It should directly motivate the next phase's research.
6. Phase order is chronological — phases do not overlap in time.
7. If two phases have very similar problems, MERGE them.
8. COVERAGE CHECK: After grouping, verify that EVERY paper from the input \
   list appears in exactly one phase. There are {n_papers} input papers — \
   count your key_papers and ensure the total equals {n_papers}. If a paper \
   does not fit perfectly into any phase, assign it to the chronologically \
   nearest phase whose problem scope it most overlaps with.
9. MINIMUM PHASE SIZE: Each phase MUST have at least 2 papers. If a paper \
   addresses a unique problem with no close neighbor, it is a BRIDGE paper — \
   assign it to the chronologically adjacent phase whose problem it most \
   overlaps with. Never create a phase with only 1 paper.

Return JSON:
```json
{{
  "phases": [
    {{
      "name": "How to Build a 3D View from 2D Images?",
      "dominant_question": "Can a reliable bird's-eye-view be constructed from multi-camera 2D images without relying on expensive depth labels?",
      "time_range": "2020-08—2022-11",
      "key_papers": [
        "Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D",
        "BEVDet: High-performance Multi-camera 3D Object Detection in Bird-Eye-View",
        "BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection",
        "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers",
        "BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision"
      ],
      "core_contradiction": "Camera-to-BEV needs depth but dense projection is expensive and depth labels constrain backbones",
      "core_debate": "Is explicit depth supervision needed for accurate BEV perception?",
      "status": "direction_clear",
      "tensions": [
        {{
          "tension": "Implicit depth is simpler but less reliable",
          "description": "LSS and BEVDet showed implicit depth can work, but BEVDepth proved explicit supervision dramatically improves accuracy, creating a dilemma between simplicity and performance.",
          "introduced_by": ["Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D"],
          "resolved_by": ["BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection", "BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision"],
          "status": "direction_clear",
          "dimension": "geometry",
          "domain_scope": "camera-only BEV perception on nuScenes"
        }}
      ],
      "unresolved_problem": "Dense BEV grids waste computation on empty space and require complex view transformations"
    }}
  ]
}}
```"""


# ── Public API ────────────────────────────────────────────────────────

def detect_problem_based_phases(
    claims: list[Claim],
    *,
    field_name: str = "",
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> list[Phase]:
    """Detect phases by clustering papers by problem_addressed semantic similarity.

    Single LLM call. Replaces detect_all_tensions() + merge_tensions_into_phases().

    Args:
        claims: All claims (chronologically ordered, with problem_addressed)
        field_name: Field name for prompt context
        client: LLM client
        model: Model override

    Returns:
        List of Phase objects (4-6, chronologically ordered). Empty list on failure.
    """
    if not claims:
        return []

    if client is None:
        from llm_analyzer import build_analyzer_client
        client = build_analyzer_client()
    if not client:
        return []

    model = _resolve_model(model)
    if not model:
        return []

    # Build paper-centric input: group claims by paper, present chronologically
    papers_map: dict[str, tuple[int, int, list[str]]] = {}  # title → (year, month, [problems])
    for c in claims:
        if c.paper_title not in papers_map:
            papers_map[c.paper_title] = (c.year, getattr(c, 'month', 0), [])
        problems = papers_map[c.paper_title][2]
        problem = c.problem_addressed.strip()
        if problem and problem not in problems:
            problems.append(problem)

    # Sort by year+month
    papers_sorted = sorted(papers_map.items(), key=lambda x: x[1][:2])

    # Build lookup tables for post-processing validation
    all_titles_set = {title for title, _ in papers_sorted}
    title_to_date = {title: (year, month) for title, (year, month, _) in papers_sorted}

    papers_lines = []
    for title, (year, month, problems) in papers_sorted:
        date_str = f"{year}-{month:02d}" if month > 0 else str(year)
        papers_lines.append(f"[{date_str}] {title}")
        for p in problems:
            papers_lines.append(f"  Problem: {p}")

    papers_text = "\n".join(papers_lines)

    prompt = _PROBLEM_PHASE_PROMPT.format(
        field_name=field_name or "this field",
        papers_text=papers_text,
        n_papers=len(papers_sorted),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _PROBLEM_PHASE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        phases = _parse_phases_response(raw)
        # Post-processing: catch unassigned papers and merge singleton phases
        phases = _repair_unassigned_papers(phases, all_titles_set, title_to_date)
        phases = _merge_singleton_phases(phases, all_titles_set, title_to_date)
        return phases
    except Exception as exc:
        print(f"Problem-based phase detection failed: {exc}")
        return []


def _parse_phases_response(raw: str) -> list[Phase]:
    """Parse LLM response into Phase list with embedded Tensions."""
    if not raw or not raw.strip():
        return []

    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            print("Problem phase detection: no JSON found in response")
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            print("Problem phase detection: invalid JSON")
            return []

    raw_phases = data.get("phases", [])
    if isinstance(data, list):
        raw_phases = data

    phases = []
    for p in raw_phases:
        if not isinstance(p, dict):
            continue
        name = p.get("name", "").strip()
        if not name:
            continue

        # Parse internal tensions
        tensions = []
        for t in p.get("tensions", []):
            if isinstance(t, dict) and t.get("tension"):
                tensions.append(Tension(
                    tension=t.get("tension", ""),
                    description=t.get("description", ""),
                    introduced_by=t.get("introduced_by", []),
                    resolved_by=t.get("resolved_by", []),
                    status=t.get("status", "open"),
                    dimension=t.get("dimension", "system"),
                    domain_scope=t.get("domain_scope", ""),
                ))

        phases.append(Phase(
            name=name,
            dominant_question=p.get("dominant_question", ""),
            time_range=p.get("time_range", ""),
            key_papers=p.get("key_papers", []),
            core_contradiction=p.get("core_contradiction", ""),
            core_debate=p.get("core_debate", ""),
            unresolved_problem=p.get("unresolved_problem", ""),
            status=p.get("status", ""),
            tensions=tensions,
        ))

    return phases


def _repair_unassigned_papers(
    phases: list[Phase],
    all_titles: set[str],
    title_to_date: dict[str, tuple[int, int]],
) -> list[Phase]:
    """Assign any input papers not in any phase to the best-fitting phase.

    Heuristic: first try time range containment (paper date falls within phase's
    time_range), then fall back to chronological distance.
    """
    assigned: set[str] = set()
    for p in phases:
        for kp in p.key_papers:
            assigned.add(kp)

    missing = all_titles - assigned
    if not missing:
        return phases

    print(f"  ⚠ {len(missing)} paper(s) unassigned by LLM, auto-assigning: {[m[:60] for m in missing]}")

    # Parse phase time ranges into (min_ym, max_ym)
    def _parse_time_range(tr: str) -> tuple[tuple[int, int], tuple[int, int]]:
        """Parse '2020-08—2022-11' into ((2020,8), (2022,11))."""
        parts = tr.replace("—", "-").split("-")
        if len(parts) >= 4:
            return (int(parts[0]), int(parts[1])), (int(parts[2]), int(parts[3]))
        if len(parts) == 2:
            try:
                return (int(parts[0].strip()), 1), (int(parts[1].strip()), 12)
            except ValueError:
                pass
        return (0, 0), (9999, 99)

    def _ym_in_range(ym: tuple[int, int], lo: tuple[int, int], hi: tuple[int, int]) -> bool:
        return lo <= ym <= hi

    for title in missing:
        paper_ym = title_to_date.get(title, (0, 0))

        # First: try time range containment
        best_idx = -1
        for i, p in enumerate(phases):
            lo, hi = _parse_time_range(p.time_range)
            if _ym_in_range(paper_ym, lo, hi):
                best_idx = i
                break

        # Second: fall back to chronological distance to phase midpoint
        if best_idx < 0:
            best_dist = 999999
            for i, p in enumerate(phases):
                lo, hi = _parse_time_range(p.time_range)
                mid = (lo[0] * 12 + lo[1] + hi[0] * 12 + hi[1]) // 2
                paper_m = paper_ym[0] * 12 + paper_ym[1]
                dist = abs(paper_m - mid)
                if dist < best_dist:
                    best_dist = dist
                    best_idx = i

        phases[best_idx].key_papers.append(title)
        phases[best_idx] = _update_phase_time_range(phases[best_idx], title_to_date)

    return phases


def _merge_singleton_phases(
    phases: list[Phase],
    all_titles: set[str],
    title_to_date: dict[str, tuple[int, int]],
) -> list[Phase]:
    """Merge phases with only 1 paper into the chronologically nearest adjacent phase."""
    if len(phases) <= 1:
        return phases

    merged: list[Phase] = []
    for i, p in enumerate(phases):
        if len(p.key_papers) >= 2:
            merged.append(p)
            continue

        # Singleton phase — merge into nearest adjacent phase
        singleton_paper = p.key_papers[0] if p.key_papers else ""
        if not singleton_paper:
            continue

        # Find chronologically nearest adjacent phase
        if not merged and i + 1 < len(phases):
            # No merged phases yet, merge into next phase
            target = phases[i + 1]
            target.key_papers.append(singleton_paper)
            target.key_papers = list(dict.fromkeys(target.key_papers))  # dedup
            # Merge tensions from singleton phase
            if p.tensions:
                target.tensions.extend(p.tensions)
            print(f"  ⚠ Merged singleton phase '{p.name}' into '{target.name}'")
        else:
            # Merge into previous phase (last in merged list)
            target = merged[-1]
            target.key_papers.append(singleton_paper)
            target.key_papers = list(dict.fromkeys(target.key_papers))
            if p.tensions:
                target.tensions.extend(p.tensions)
            print(f"  ⚠ Merged singleton phase '{p.name}' into '{target.name}'")

    # Recalculate time ranges for affected phases
    for p in merged:
        p = _update_phase_time_range(p, title_to_date)

    return merged


def _update_phase_time_range(phase: Phase, title_to_date: dict[str, tuple[int, int]]) -> Phase:
    """Recalculate phase time_range from key_papers dates."""
    dates = []
    for kp in phase.key_papers:
        ym = title_to_date.get(kp)
        if ym:
            dates.append(ym)
    if not dates:
        return phase
    min_date = min(dates, key=lambda x: (x[0], x[1]))
    max_date = max(dates, key=lambda x: (x[0], x[1]))
    min_str = f"{min_date[0]}-{min_date[1]:02d}" if min_date[1] > 0 else str(min_date[0])
    max_str = f"{max_date[0]}-{max_date[1]:02d}" if max_date[1] > 0 else str(max_date[0])
    phase.time_range = f"{min_str}—{max_str}"
    return phase


# ── Formatting helpers (for prompt injection and markdown) ────────────

def phases_to_text(phases: list[Phase]) -> str:
    """Format phases for narrative prompt injection — rich version with tensions."""
    if not phases:
        return ""

    lines = ["NARRATIVE PHASES (problem-driven, with causal chain):"]
    for i, p in enumerate(phases, 1):
        lines.append(f"\n  Phase {i}: {p.name} ({p.time_range})")
        lines.append(f"    Dominant question: {p.dominant_question}")
        lines.append(f"    Core contradiction: {p.core_contradiction}")
        lines.append(f"    Core debate: {p.core_debate}")
        lines.append(f"    Status: {p.status}")
        lines.append(f"    Key papers: {', '.join(p.key_papers)}")
        if p.tensions:
            lines.append(f"    Internal tensions:")
            for t in p.tensions:
                lines.append(f"      - {t.tension}: {t.description[:120]}")
                lines.append(f"        Introduced by: {', '.join(t.introduced_by[:3])}")
                lines.append(f"        Resolved by: {', '.join(t.resolved_by[:3])}")
        next_label = f"→ Phase {i + 1} seed" if i < len(phases) else "→ open question"
        lines.append(f"    Unresolved problem ({next_label}): {p.unresolved_problem}")
    return "\n".join(lines)


def phases_to_markdown(phases: list[Phase]) -> str:
    """Render phases as markdown overview table."""
    if not phases:
        return ""

    lines = [
        "### 技术发展阶段",
        "",
        "| 阶段 | 时间 | 关键论文 | 遗留问题 |",
        "|------|------|----------|----------|",
    ]
    for p in phases:
        papers = ', '.join(p.key_papers[:3])
        if len(p.key_papers) > 3:
            papers += f" (+{len(p.key_papers) - 3})"
        lines.append(
            f"| **{p.name}** | {p.time_range} | {papers} | "
            f"{p.unresolved_problem} |"
        )
    return "\n".join(lines) + "\n"


def tensions_to_markdown(tensions: list[Tension]) -> str:
    """Render tensions as a markdown table (same interface as tension_detector version)."""
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
