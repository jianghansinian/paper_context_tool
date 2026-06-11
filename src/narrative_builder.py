"""Narrative builder — the product core of V4.

Generates a field's technical evolution story from extracted Claims,
using a drama structure: Conflict → Attempt → Failure → New Attempt → Breakthrough.

For MVP, branches are manually specified. Branch discovery will be automated
in a later phase.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _resolve_model
from paper import Claim, ResearchNarrative, Branch, EvolutionEdge

# ── System prompt for narrative generation ────────────────────────────
_NARRATIVE_SYSTEM = """\
You are a senior research historian specializing in autonomous driving and \
computer vision. You write technical evolution narratives that capture how a \
field's collective understanding shifts over time.

Your writing style:
- Field-centric: the protagonist is the RESEARCH COMMUNITY'S UNDERSTANDING, \
not individual papers
- Phase-driven: organize around paradigm shifts and changing consensus, not \
paper chronology
- Causal: explain WHY understanding shifted, not just WHAT methods appeared
- Specific: use concrete method names, metrics, and mechanisms as evidence
- Concise: every sentence advances the story

A good field-centric narrative:
1. Opens with the CONSENSUS BELIEF at the start of a phase
2. Shows what DISCOVERY or FAILURE cracked that consensus
3. Traces how the NEW UNDERSTANDING stabilized
4. Ends with what the field learned — and what TENSION carried into the next phase

CRITICAL: Write like an expert lecturer. Papers are the characters who drive the \
story — name them naturally as you explain what each contributed and why it mattered. \
The narrative arc is the field's changing understanding; papers are the agents of \
that change. Never write "Paper A does X, then Paper B does Y" as a dry catalog — \
but DO name the papers as you tell the story of how understanding evolved.

CALIBRATION RULE: This analysis is based on a LIMITED SET of papers within this \
specific narrative scope. The field is larger and more complex. You MUST use \
hedged, evidence-bound language. Avoid absolute claims that imply historical \
finality. Specifically:
- NEVER write: "the paradigm shift was complete", "the era of X was over", \
  "X was definitively superior", "the field had fully embraced X", "answered \
  with a resounding no", "the consensus was now clear"
- INSTEAD write: "within this trajectory, X emerged as a strong alternative", \
  "the evidence from these papers suggested that...", "this pointed toward X \
  as a promising direction", "X gained significant traction"
- When describing paradigm shifts, qualify: "Within the scope of these papers, \
  a shift occurred from X to Y" — not "The field abandoned X for Y"
- Anchor every strong claim to a specific paper's evidence: "SparseBEV (2023) \
  demonstrated that..." not "Sparse representations are superior to dense ones"

Return ONLY a JSON object. No other text."""


# ── Per-branch narrative prompt ───────────────────────────────────────
_PHASE_NARRATIVE_PROMPT = """\
Write a technical evolution story for this research phase — the story of how \
the field's collective understanding evolved during this period.

PHASE: {phase_name} ({time_range})
DRIVING PROBLEM: {problem_statement}
CORE PARADIGM: {core_paradigm}

CLAIM EVOLUTION CHAIN (chronological):
{claims_text}

{claim_relations}

{tensions}

{paradigm_shifts}

STRUCTURE YOUR NARRATIVE AS A PROFESSOR'S LECTURE ON HOW UNDERSTANDING EVOLVED:

1. INITIAL CONSENSUS — What did the field believe at the start? What was \
the shared assumption? Name the key paper(s) that established this consensus.

2. DISCOVERIES THAT RESHAPED UNDERSTANDING — Walk through each major turning \
point. For each: name the paper, state what it discovered, explain WHY it \
changed people's minds, and use concrete metrics as evidence. Show how each \
discovery led to or forced the next one.

3. CULMINATION & UNRESOLVED TENSION — What new understanding emerged? What \
became the new baseline? Name the paper(s) that crystallized this. What \
problem remained unsolved that motivated the NEXT phase?

CRITICAL RULES:
- Write like a professor explaining a field's history — papers are CHARACTERS, \
their discoveries are plot points, the field's evolving understanding is the arc
- Introduce paper names naturally with year: "LSS (2020) showed that..." not \
"Then LSS was proposed..." The name anchors the idea, not the sentence structure
- Don't write disembodied statements like "The field realized..." when a specific \
paper drove the realization. Instead: "BEVDepth (2022) revealed that depth \
supervision could push NDS past 60% — the first camera-only method to do so."
- Connect causally: "this finding forced researchers to reconsider..."
- End each paragraph with a sentence that bridges to the next discovery
- Use concrete metrics as evidence of WHY the belief changed
- When a claim ATTACKS or REPLACES a prior one, make this the dramatic center \
of the story — these are paradigm shifts, not incremental improvements
- PARADIGM SHIFTS (provided above) are the key turning points. Build the narrative \
ARC around them — each shift is a moment when the field's beliefs fundamentally \
changed. The story should make the reader feel the ground shifting under their feet.
- CALIBRATION: This narrative is based on a LIMITED paper set. Use hedged language. \
Never write "the paradigm shift was complete" or "the era was over" or "X was \
superior." Instead: "within this trajectory, X gained traction," "the evidence \
from these papers pointed toward X," "X emerged as a promising direction." \
Anchor every strong claim to a specific paper's evidence.

Return JSON:
```json
{{
  "narrative": "..."
}}
```"""


# ── Field overview prompt ─────────────────────────────────────────────
_FIELD_OVERVIEW_PROMPT = """\
Write a 2-3 paragraph overview of this research field's evolution across phases.

FIELD: {field_name}

PHASE DATA (chronological):
{phase_data}

INSTRUCTIONS:
- What drove the formation of this field? What was the original breakthrough?
- How and WHY did the field shift from one phase to the next? What specific \
discovery or failure triggered each paradigm shift? The paradigm relation below \
tells you how the phases relate — use this to frame the transition.
- How did insights from earlier phases influence later ones?
- What is the current trajectory?

CRITICAL: You MUST only reference papers from the PAPER LIST provided above. \
These are the only papers in this analysis. Do NOT mention any paper not listed. \
If the paper list doesn't include a well-known work, do not name it — describe \
the idea without naming a specific paper not in the list.

Write this as a single coherent story. Do NOT list phases one by one.

Return JSON:
```json
{{
  "overview": "..."
}}
```"""


# ── Synthesis prompt ──────────────────────────────────────────────────
_SYNTHESIS_PROMPT = """\
Write a synthesis that identifies the deep patterns across this field's research phases.

FIELD: {field_name}

PHASE NARRATIVES:
{phase_narratives}

OVERVIEW:
{overview}

Your synthesis must have TWO clearly separated parts:

PART 1 — EVIDENCE-BACKED CONCLUSIONS:
Identify 2-3 definitive patterns that are directly supported by the papers analyzed.
These must be conclusions that the paper evidence compels. For each:
- State the pattern
- Name the specific papers and results that support it
- Explain the causal mechanism: "Paper X showed Y, which forced Z"

PART 2 — SPECULATIVE DIRECTIONS (EXPLICITLY LABELED AS SPECULATION):
Identify 1-2 plausible future directions, explicitly acknowledging uncertainty.
These are inferences BEYOND what the papers directly prove. For each:
- State the direction with a confidence qualifier (e.g., "If the trend holds, ...")
- Name the unresolved tension from Part 1 that motivates this speculation
- Explicitly flag: "This is speculation, not a conclusion from the analyzed papers."

CRITICAL: Do NOT present speculation as if it follows inevitably from the evidence.
The reader must be able to tell which statements are paper-grounded and which are
your extrapolation.

Return JSON:
```json
{{
  "evidence_backed": "string (2-3 evidence-backed conclusions)",
  "speculative": "string (1-2 speculative future directions, clearly flagged)"
}}
```"""


# ── Phase structure proposal prompt ───────────────────────────────────

_PHASE_STRUCTURE_PROMPT = """\
You are analyzing a set of research claims to identify the natural paradigm \
structure of a field. Your goal is to find where the FUNDAMENTAL ASSUMPTIONS \
changed — not where methods incrementally improved.

FIELD: {field_name}

Below are {n_claims} claims extracted from {n_papers} papers, sorted chronologically.

CLAIMS:
{claims_text}

A "phase" (or paradigm era) is defined by a shared CORE ASSUMPTION about what \
approach is fundamentally correct. When the assumption changes, a new phase begins.

Contrast:
- INCREMENTAL: "better depth estimation improves detection" → same phase
- PARADIGM SHIFT: "dense BEV grids are unnecessary, sparse queries suffice" → new phase

Examples of paradigm-level assumptions:
- "Depth must be explicitly supervised" vs "Depth can be learned implicitly via attention"
- "Dense BEV grid is required" vs "Sparse object queries can replace the grid"
- "Modular pipeline is necessary" vs "End-to-end learning is viable"

Return JSON:
```json
{{
  "phases": [
    {{
      "name": "string (descriptive label, e.g. 'Dense BEV Era')",
      "time_range": "string (e.g. '2020-2022')",
      "problem_statement": "string (one sentence: the core problem driving this phase)",
      "core_paradigm": "string (one sentence: the shared fundamental assumption)",
      "paper_arxiv_ids": ["id1", "id2", ...],
      "key_evolution": "string (1-2 sentences: how understanding evolved WITHIN this phase)"
    }}
  ],
  "rationale": "string (1-2 sentences explaining why you divided the phases at these boundaries)"
}}
```

IMPORTANT:
- Aim for 2-3 phases. 4+ phases means you are splitting on methods, not paradigms.
- Every paper must appear in exactly one phase.
- Phases must be in chronological order.
- The boundary between phases must represent a GENUINE PARADIGM SHIFT.
"""


def propose_structure(
    claims: list,
    field_name: str = "BEV Perception",
    *,
    client=None,
    model: Optional[str] = None,
) -> Optional[list[dict]]:
    """Ask LLM to propose phase structure from flat claim list.

    Returns list of phase dicts with keys: name, time_range, problem_statement,
    core_paradigm, paper_arxiv_ids, key_evolution.
    Returns None on failure (caller should fall back to single phase).
    """
    if not claims:
        return None

    if client is None:
        client = _build_client()

    model = _resolve_model(model)
    if not model:
        return None

    # Deduplicate papers for counting
    paper_ids = list(set(c.paper_id for c in claims))
    n_papers = len(paper_ids)
    claims_text = _claims_to_text(claims)

    prompt = _PHASE_STRUCTURE_PROMPT.format(
        field_name=field_name,
        n_claims=len(claims),
        n_papers=n_papers,
        claims_text=claims_text,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a research meta-analyst. Identify paradigm shifts in research claim chains. Return ONLY JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=2048,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        phases = data.get("phases", [])
        if phases and isinstance(phases, list):
            return phases
    except Exception as exc:
        print(f"Structure proposal failed: {exc}")

    return None


# ── Public API ────────────────────────────────────────────────────────


def build_narrative(
    branches: list[dict],
    field_name: str = "BEV Perception",
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    seed_paper_id: Optional[str] = None,
) -> Optional[ResearchNarrative]:
    """Build a complete research narrative for a field.

    Args:
        branches: List of dicts with keys:
            - name: branch name
            - problem_statement: what problem this branch addresses
            - claims: list of Claim objects (ordered by year)
        field_name: name of the research field
        client: OpenAI-compatible LLM client
        model: model override
        seed_paper_id: if provided, the seed paper that triggered this analysis

    Returns:
        ResearchNarrative object, or None on failure.
    """
    if client is None:
        client = _build_client()

    model = _resolve_model(model)
    if not model:
        print("Narrative builder: no model available")
        return None

    # Step 1: Generate per-phase narratives
    branch_objects = []
    branch_narratives: dict[str, str] = {}  # name → narrative text
    for b in branches:
        narrative_text = _generate_phase_narrative(
            client, model,
            phase_name=b["name"],
            problem_statement=b.get("problem_statement", ""),
            claims=b.get("claims", []),
            time_range=b.get("time_range", ""),
            core_paradigm=b.get("core_paradigm", ""),
            claim_relations=b.get("claim_relations"),
            tensions=b.get("tensions"),
            paradigm_shifts=b.get("paradigm_shifts"),
        )
        if narrative_text:
            branch_narratives[b["name"]] = narrative_text
            bobj = Branch(
                name=b["name"],
                problem_statement=b.get("problem_statement", ""),
                paper_ids=[c.paper_id for c in b.get("claims", [])],
                claims=b.get("claims", []),
                narrative=narrative_text,
                is_mainstream=b.get("is_mainstream", False),
                time_range=b.get("time_range", ""),
                core_paradigm=b.get("core_paradigm", ""),
                claim_relations=b.get("claim_relations", []),
                paradigm_shifts=b.get("paradigm_shifts", []),
            )
            branch_objects.append(bobj)

    if not branch_objects:
        print("Narrative builder: no branch narratives generated")
        return None

    # Step 2: Generate field overview
    overview = _generate_field_overview(
        client, model, field_name, branch_objects, branch_narratives)

    # Step 3: Generate synthesis
    synthesis = _generate_synthesis(
        client, model, field_name, branch_objects, branch_narratives, overview)

    return ResearchNarrative(
        field_name=field_name,
        seed_paper_id=seed_paper_id,
        overview=overview,
        branches=branch_objects,
        synthesis=synthesis,
    )


def _build_client() -> Optional[OpenAI]:
    try:
        from llm_analyzer import build_analyzer_client
        return build_analyzer_client()
    except Exception:
        return None


def _claims_to_text(claims: list[Claim]) -> str:
    """Format claims as a readable evolution chain."""
    lines = []
    for i, c in enumerate(claims):
        lines.append(
            f"[{c.year}] {c.paper_title}\n"
            f"  arXiv: {c.paper_id.replace('arxiv:', '')}\n"
            f"  Claim: {c.statement}\n"
            f"  Evidence: {c.evidence}\n"
            f"  Problem addressed: {c.problem_addressed}\n"
            f"  Relation to prior: {c.claim_type}"
        )
        if i < len(claims) - 1:
            lines.append("  ↓")
    return "\n".join(lines)


def _format_relations_for_prompt(relations: list[dict]) -> str:
    """Format claim relations for injection into the narrative prompt."""
    if not relations:
        return ""
    lines = [
        "PAPER-TO-PAPER CLAIM RELATIONSHIPS (classified by a meta-analyst):",
        "Use these to identify turning points. ATTACK/REPLACE = paradigm shift moment.",
        "CRITICAL: Papers marked PARALLEL belong to DIFFERENT research lineages.",
        "They happen to appear consecutively in time but are NOT causally related.",
        "Do NOT narrate parallel papers as if one led to the other — they are",
        "independent developments addressing different problems.",
    ]
    for r in relations:
        rel = r["relation"].upper()
        if r["relation"] == "parallel":
            rel = "PARALLEL (different lineage — narrate as independent development)"
        lines.append(
            f"  {r['source_paper']} → {r['target_paper']}: {rel}"
        )
        lines.append(f"    {r['explanation']}")
    return "\n".join(lines)


def _generate_phase_narrative(
    client: OpenAI,
    model: str,
    phase_name: str,
    problem_statement: str,
    claims: list[Claim],
    *,
    time_range: str = "",
    core_paradigm: str = "",
    claim_relations: Optional[list[dict]] = None,
    tensions: Optional[list[dict]] = None,
    paradigm_shifts: Optional[list[dict]] = None,
) -> str:
    """Generate field-centric narrative for a single phase."""
    if not claims:
        return ""

    claims_text = _claims_to_text(claims)

    # Format claim relations for prompt injection
    relations_text = ""
    if claim_relations:
        relations_text = _format_relations_for_prompt(claim_relations)

    # Format tensions for prompt injection
    tensions_text = ""
    if tensions:
        from tension_detector import format_tensions_for_narrative
        tensions_text = format_tensions_for_narrative(tensions, phase_name=phase_name)

    # Format paradigm shifts for prompt injection
    shifts_text = ""
    if paradigm_shifts:
        from paradigm_shift_detector import format_shifts_for_narrative
        shifts_text = format_shifts_for_narrative(paradigm_shifts, phase_name=phase_name)

    prompt = _PHASE_NARRATIVE_PROMPT.format(
        phase_name=phase_name,
        time_range=time_range or "N/A",
        problem_statement=problem_statement or f"How to advance {phase_name.lower()}",
        core_paradigm=core_paradigm or "Not specified",
        claims_text=claims_text,
        claim_relations=relations_text,
        tensions=tensions_text,
        paradigm_shifts=shifts_text,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _NARRATIVE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=4096,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_json_field(raw, "narrative")
        return parsed or ""
    except Exception as exc:
        print(f"Phase narrative failed for {phase_name}: {exc}")
        return ""


def _generate_field_overview(
    client: OpenAI,
    model: str,
    field_name: str,
    branch_objects: list[Branch],
    branch_narratives: dict[str, str],
) -> str:
    """Generate field-level overview narrative across phases."""
    blocks = []
    for b in branch_objects:
        # Deduplicate by paper_id, collect title+year
        seen: dict[str, tuple[str, int]] = {}
        for c in b.claims:
            if c.paper_id not in seen:
                seen[c.paper_id] = (c.paper_title, c.year)
        paper_list = ", ".join(f"{title} ({year})" for title, year in
                               sorted(seen.values(), key=lambda x: x[1]))
        narrative = branch_narratives.get(b.name, "")
        paradigm_relation = ""
        for r in b.claim_relations:
            if r["relation"] in ("replace", "attack") and "PARADIGM" in r.get("explanation", ""):
                paradigm_relation = f"Paradigm relation to previous phase: {r['relation'].upper()} — {r['explanation']}"
                break

        blocks.append(
            f"PHASE: {b.name} ({b.time_range})\n"
            f"Core paradigm: {b.core_paradigm}\n"
            f"Problem: {b.problem_statement}\n"
            f"Papers: {paper_list}\n"
            f"{paradigm_relation}\n"
            f"Narrative summary: {narrative[:400]}"
        )

    prompt = _FIELD_OVERVIEW_PROMPT.format(
        field_name=field_name,
        phase_data="\n\n---\n\n".join(blocks),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _NARRATIVE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=2048,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_json_field(raw, "overview")
        return parsed or ""
    except Exception as exc:
        print(f"Field overview failed: {exc}")
        return ""


def _generate_synthesis(
    client: OpenAI,
    model: str,
    field_name: str,
    branch_objects: list[Branch],
    branch_narratives: dict[str, str],
    overview: str,
) -> str:
    """Generate future synthesis across phases.

    Returns a combined string with evidence-backed and speculative sections
    clearly separated.
    """
    narrative_texts = "\n\n".join(
        f"--- {b.name} ---\n{branch_narratives.get(b.name, '')}"
        for b in branch_objects
    )

    prompt = _SYNTHESIS_PROMPT.format(
        field_name=field_name,
        phase_narratives=narrative_texts,
        overview=overview,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _NARRATIVE_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=1536,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        evidence = _parse_json_field(raw, "evidence_backed") or ""
        speculative = _parse_json_field(raw, "speculative") or ""

        if evidence and speculative:
            return (
                f"### Evidence-Backed Conclusions\n\n{evidence}\n\n"
                f"### Speculative Future Directions\n\n"
                f"> _The following is extrapolation beyond the analyzed papers. "
                f"It reflects plausible trajectories, not established fact._\n\n"
                f"{speculative}"
            )
        elif evidence:
            return f"### Evidence-Backed Conclusions\n\n{evidence}"
        # Fallback: try old "synthesis" field
        parsed = _parse_json_field(raw, "synthesis")
        return parsed or evidence or ""
    except Exception as exc:
        print(f"Synthesis generation failed: {exc}")
        return ""


def _parse_json_field(raw: str, field: str) -> Optional[str]:
    """Extract a string field from a JSON response."""
    if not raw or not raw.strip():
        return None
    # Remove markdown fence if present
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data.get(field, "").strip() or None
    except json.JSONDecodeError:
        # Try regex extraction
        pattern = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            return m.group(1).replace("\\n", "\n").replace('\\"', '"')
    return None


# ── Convenience function for MVP testing ──────────────────────────────


def build_narrative_from_claims(
    claims_by_branch: dict[str, list[Claim]],
    problem_statements: dict[str, str],
    field_name: str = "BEV Perception",
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    claim_relations: Optional[dict[str, list[dict]]] = None,
    tensions: Optional[list[dict]] = None,
    paradigm_shifts: Optional[list[dict]] = None,
) -> Optional[ResearchNarrative]:
    """Build narrative from claims organized by branch.

    Args:
        claims_by_branch: {branch_name: [Claim, ...]}
        problem_statements: {branch_name: problem_statement}
        field_name: field name
        client: LLM client
        model: model override
        claim_relations: {branch_name: [relation_dict, ...]} — optional per-branch claim relations
        tensions: optional list of research tension dicts
        paradigm_shifts: optional list of paradigm shift dicts

    Returns:
        ResearchNarrative or None
    """
    # Assign paradigm shifts to their respective phases
    shifts_by_phase: dict[str, list[dict]] = {}
    if paradigm_shifts:
        for s in paradigm_shifts:
            phase = s.get("phase", "")
            if phase:
                shifts_by_phase.setdefault(phase, []).append(s)

    branches = []
    for name, claims in claims_by_branch.items():
        # Sort by year
        claims = sorted(claims, key=lambda c: c.year)
        branches.append({
            "name": name,
            "problem_statement": problem_statements.get(name, ""),
            "claims": claims,
            "is_mainstream": len(claims) >= 3,
            "time_range": problem_statements.get(f"{name}__time_range", ""),
            "core_paradigm": problem_statements.get(f"{name}__core_paradigm", ""),
            "claim_relations": (claim_relations or {}).get(name),
            "tensions": tensions or [],
            "paradigm_shifts": shifts_by_phase.get(name, []),
        })

    # Sort branches: mainstream first, then by paper count
    branches.sort(key=lambda b: (not b["is_mainstream"], -len(b["claims"])))

    return build_narrative(
        branches=branches,
        field_name=field_name,
        client=client,
        model=model,
    )
