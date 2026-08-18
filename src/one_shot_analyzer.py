"""True One-Shot Field Analysis — Scheme B.

Feeds ALL papers' raw text (title, year, month, abstract, introduction,
results textual conclusions, conclusion) to the LLM in ONE call.

The LLM infers stages, shifts, claims, and tensions directly from raw text —
no pre-extracted claims/relations/tensions needed.

Pipeline:
  1. One-shot analysis (1 LLM call) → stages + shifts + claims + tensions

Structured narrative generation from the analysis output lives in
one_shot_narrative.py (design: design_stage_boundary.md §6.6).
"""

from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _resolve_model, _extract_json_object
from paper import Claim, Tension, Phase, ParadigmShift
from text_extractor import assemble_paper_text_for_one_shot


# ═══════════════════════════════════════════════════════════════════════
# Step 1 — One-Shot Analysis (1 LLM call)
# ═══════════════════════════════════════════════════════════════════════

_ONE_SHOT_SYSTEM = """\
You are a research historian who analyzes how scientific fields evolve.
Your task: read a collection of papers and identify the STAGES of its development —
each stage is a logically distinct step in the field's evolution, not necessarily
a long time period. A stage may contain only a few papers or even a single
foundational paper that establishes a new direction.

══════════════════════════════════════════════════════════════════════════
CORE FRAMEWORK
══════════════════════════════════════════════════════════════════════════

A STAGE is a set of papers that share the same approach to the CORE TECHNICAL
PROBLEM. Stages are separated by STAGE BOUNDARIES — points where EITHER:

  A) HOW the core problem is solved changes fundamentally.
     The community adopts a different kind of mechanism, representation,
     or framework for addressing the central challenge.

  B) WHAT the work is trying to achieve changes fundamentally.
     The community shifts from BUILDING the core capability to APPLYING it
     for a fundamentally different downstream purpose.

To find boundaries, compare papers along three axes:

  AXIS 1 (PRIMARY — determines stage membership):
    HOW is the core technical artifact constructed or represented?
    Papers using the same kind of construction belong to the same stage.
    Papers using a fundamentally different kind of construction belong to
    different stages — even if they share vocabulary, application domain,
    or publication time.

  AXIS 2 (SECONDARY — determines ordering WITHIN a stage):
    How does each paper refine or extend the shared construction approach?
    This axis orders papers within a stage (e.g., improving a component's
    accuracy, incorporating additional data sources, scaling up). It does
    NOT define stage boundaries.

  AXIS 3 (TERTIARY — determines task-scope boundaries):
    Is the work about BUILDING the core capability itself, or about USING it
    as a component within a fundamentally different, larger system or goal?
    "Build" and "apply for a different purpose" are different stages.

KEY DISTINCTION — "How" change vs "Improvement":
  If a paper uses a FUNDAMENTALLY DIFFERENT MECHANISM to construct the core
  artifact (e.g., a switch from one computational paradigm to a qualitatively
  different one), that is a new stage.
  If a paper improves the SAME mechanism (e.g., refines a component, adds a
  new data source, improves optimization), that is progress within the same stage.

══════════════════════════════════════════════════════════════════════════
HOW TO IDENTIFY STAGE BOUNDARIES
══════════════════════════════════════════════════════════════════════════

Read ALL papers. For each paper, identify:
  - What is the CORE TECHNICAL ARTIFACT it constructs or studies?
  - HOW does it construct or study that artifact? (The mechanism)
  - WHAT is the ultimate purpose? (Building the artifact vs applying it)

LITMUS TEST for a candidate boundary between two groups of papers:

  CONSTRUCTION TEST:
  Do the two groups use FUNDAMENTALLY DIFFERENT mechanisms to construct
  or study the core artifact? Would a researcher from group A look at
  group B's mechanism and say "this is a completely different way of
  doing it — it's not just an improvement of our approach"?

  PURPOSE TEST:
  Is group A BUILDING the core artifact, while group B is USING it as a
  component within a fundamentally different, larger system? Would group
  A's authors say "we built the tool; they're using our tool to solve a
  different problem we weren't trying to solve"?

  If YES to either test → stage boundary.

WHAT IS NOT A STAGE BOUNDARY:
- An improvement or refinement of the same construction mechanism
- Adding a component or data modality to the same mechanism
- Better performance on the same benchmarks
- Better engineering of the same approach

STAGE SIZE:
- A stage TYPICALLY contains at least 2 papers, but a genuinely foundational
  paper that introduces a novel construction mechanism and stands alone
  conceptually MAY form its own stage.
- Do NOT force singleton papers into adjacent stages if they are
  conceptually distinct. Let the construction mechanism decide.
- Conversely, do NOT split a coherent group just to create more stages.

Time overlap between stages is normal and expected. Chronology is for
ordering within a stage, not for defining stage membership.

══════════════════════════════════════════════════════════════════════════
WHAT TO OUTPUT
══════════════════════════════════════════════════════════════════════════

1. STAGES (called "phases" in the output JSON for compatibility):
   - Its DOMINANT QUESTION (what the community was trying to figure out)
   - Its CORE TENSION (the central bottleneck driving work)
   - Its assigned papers (every paper belongs to exactly one stage)

2. STAGE TRANSITIONS (shifts):
   - What changed between adjacent stages (which dimension and how)
   - What evidence (papers, results) triggered the transition

3. TENSIONS — debates within a stage:
   - Different refinements of the SAME construction mechanism → tension
   - Different construction mechanisms → stage boundary, not a tension

4. CLAIMS — each paper's core assertion. A claim is a JUDGMENT, not a
   method description.

CRITICAL: Do NOT prescribe how many stages there are. The papers determine
that — if construction mechanisms shifted 4 times, output 4 stages. Let the
evidence speak, not your expectations."""


_ONE_SHOT_PROMPT = """\
Analyze the evolution of {field_name} from the following papers. Read ALL papers \
carefully, then identify the STAGES of this field's development.

══════════════════════════════════════════════════════════════════════════
PAPERS (chronological order)
══════════════════════════════════════════════════════════════════════════

{papers_text}

══════════════════════════════════════════════════════════════════════════
YOUR TASK
══════════════════════════════════════════════════════════════════════════

STEP 1 — Identify STAGES by grouping papers that share the same construction
  mechanism for the core technical artifact.

  For each paper, first identify:
  - What is the CORE TECHNICAL ARTIFACT this paper constructs or studies?
  - HOW does it construct or study it? Describe the mechanism in your own words.
    (Do NOT use predefined categories — let each paper tell you what it does.)
  - WHAT is the ultimate PURPOSE? (Building the artifact itself, or using it
    as a component in a fundamentally larger system?)

  Group papers by HOW they construct the core artifact. Papers using
  FUNDAMENTALLY DIFFERENT construction mechanisms belong to different stages.

  Then, within each construction-based group, check the PURPOSE:
  - Are some papers building the artifact, while others are applying it as
    a component in a fundamentally different kind of system?
  - If yes, split into separate stages — "build" and "apply for different
    purpose" are different stages.

  LITMUS TEST for each candidate boundary:

  CONSTRUCTION TEST:
  Does group B use a FUNDAMENTALLY DIFFERENT mechanism from group A to
  construct or study the core artifact? Would a group-A researcher see
  group B's mechanism and say "this is not an improvement of our approach —
  it's a completely different way of doing it"?

  PURPOSE TEST:
  Is group A BUILDING the core artifact, while group B is USING it within
  a fundamentally different, larger system? Would group-A authors say "we
  built the tool; they're using our tool to solve a different problem"?

  If YES to either test → different stages.

  STAGE SIZE:
  - Typically at least 2 papers per stage, but a genuinely foundational
    paper that introduces a novel construction mechanism MAY stand alone.
  - Do NOT force a conceptually distinct singleton into an adjacent stage.
  - Do NOT split a coherent group just to create more stages.

  Every paper must be assigned to exactly one stage.

STEP 2 — Describe STAGE TRANSITIONS (shifts):

  For each stage boundary, describe:
  - What was the construction mechanism / purpose BEFORE this point?
  - What shifted? (Construction mechanism? Purpose? Both?)
  - What triggered this transition? Name the specific papers that introduced
    the new mechanism or repurposed the artifact.

  The founding of the field is NOT a shift — there is no "before."

STEP 3 — Extract CLAIMS from each paper:

  For each paper, extract its most important claims — assertions about what
  is true or what works. A claim is a JUDGMENT backed by evidence, not a
  method description.

STEP 4 — Identify TENSIONS within each stage:

  A tension is a debate between competing refinements WITHIN the same stage
  — same construction mechanism, different specific choices. If papers use
  DIFFERENT construction mechanisms, that's a stage boundary, not a tension.

OUTPUT FORMAT — Return a single JSON object:

```json
{{
  "phases": [
    {{
      "index": 0,
      "name": "Stage name (descriptive label)",
      "dominant_question": "The core question the community was asking",
      "core_tension": "The central bottleneck driving work in this stage",
      "papers": ["Exact Paper Title 1", "Exact Paper Title 2"],
      "year_range": "YYYY-MM—YYYY-MM"
    }}
  ],
  "shifts": [
    {{
      "shift_name": "Short label for this transition",
      "from_phase": 0,
      "to_phase": 1,
      "old_question": "What the community was asking before",
      "new_question": "What the community started asking instead",
      "catalyst_papers": ["Paper That Introduced the New Direction"],
      "trigger": "What caused this transition"
    }}
  ],
  "claims": [
    {{
      "paper": "Exact Paper Title",
      "statement": "A falsifiable judgment about what is true or what works",
      "evidence": "Supporting evidence from the paper",
      "claim_level": "paradigm | methodological | engineering"
    }}
  ],
  "tensions": [
    {{
      "phase": 0,
      "name": "Short label for this debate",
      "description": "What is the disagreement about?",
      "positions": [
        {{"paper": "Paper Title A", "position": "Position description", "evidence": "..."}},
        {{"paper": "Paper Title B", "position": "Opposing position description", "evidence": "..."}}
      ]
    }}
  ]
}}
```

QUALITY CHECKS:
- [ ] EVERY paper in exactly one stage: {n_papers} input = {n_papers} assigned
- [ ] Stages reflect construction-mechanism or purpose changes (not just improvements)
- [ ] Stage count = shift count + 1
- [ ] Tensions are intra-stage (same construction mechanism, different refinements)
- [ ] Claims are falsifiable judgments with evidence, not method descriptions
- [ ] Paper titles match EXACTLY as given (copy-paste from input)

Return ONLY the JSON object. No other text."""


def analyze_field_one_shot(
    papers: list,
    field_name: str,
    client: OpenAI,
    *,
    model: Optional[str] = None,
) -> dict:
    """True One-Shot analysis: feed all papers' raw text, get stages + shifts + claims + tensions.

    Args:
        papers: list of Paper objects with title, year, month, abstract, full_text.
        field_name: e.g. "BEV Perception"
        client: OpenAI-compatible LLM client.
        model: model override.

    Returns:
        dict with keys: stages, shifts, claims, tensions, raw_response.
        Empty dict on failure.
    """
    if not papers:
        print("One-shot analysis: no papers provided")
        return {}

    model = _resolve_model(model)
    if not model:
        print("One-shot analysis: no model available")
        return {}

    if len(papers) > 50:
        print(f"WARNING: {len(papers)} papers may exceed token limits for one-shot analysis.")
        print("  Consider using the multi-step pipeline (Scheme A) for large paper sets.")

    # Assemble per-paper text
    paper_texts = []
    for p in sorted(papers, key=lambda x: (getattr(x, "year", 9999), getattr(x, "month", 0))):
        text = assemble_paper_text_for_one_shot(p)
        paper_texts.append(text)

    papers_text = "\n\n---\n\n".join(paper_texts)

    # Rough token estimate for logging
    est_tokens = len(papers_text) // 4
    print(f"One-shot input: {len(papers)} papers, {len(papers_text):,} chars (~{est_tokens} tokens)")

    prompt = _ONE_SHOT_PROMPT.format(
        field_name=field_name,
        papers_text=papers_text,
        n_papers=len(papers),
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _ONE_SHOT_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=config.LLM_ANALYZER_MAX_TOKENS,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC * 3,
        )
        raw = response.choices[0].message.content or ""
        return _parse_one_shot_response(raw, papers)
    except Exception as exc:
        print(f"One-shot analysis failed: {exc}")
        return {}


def _parse_one_shot_response(raw: str, papers: list) -> dict:
    """Parse the one-shot LLM response into structured data."""
    if not raw or not raw.strip():
        return {}

    data = _extract_json_object(raw)
    if not data:
        # Fallback: find JSON object directly
        m = re.search(r"\{[\s\S]*\"phases\"[\s\S]*\}", raw)
        if not m:
            m = re.search(r"\{[\s\S]*\"stages\"[\s\S]*\}", raw)  # backward compat
        if m:
            try:
                data = json.loads(m.group(0))
            except json.JSONDecodeError:
                print("One-shot: failed to parse JSON from response")
                return {}
        else:
            print("One-shot: no valid JSON found in response")
            return {}

    # Accept both "phases" (new) and "stages" (backward compat)
    phases = data.get("phases", data.get("stages", []))
    shifts = data.get("shifts", [])
    claims = data.get("claims", [])
    tensions = data.get("tensions", [])

    # Check paper coverage
    assigned_papers: set[str] = set()
    for p in phases:
        for t in p.get("papers", []):
            assigned_papers.add(t)

    input_papers = {p.title for p in papers}
    missing = input_papers - assigned_papers
    extra = assigned_papers - input_papers

    if missing:
        print(f"  One-shot coverage gap: {len(missing)} papers not assigned to any phase")
    if extra:
        print(f"  One-shot extra papers: {len(extra)} papers not in input")

    result = {
        "phases": phases,
        "shifts": shifts,
        "claims": claims,
        "tensions": tensions,
        "raw_response": raw,
    }

    print(f"  One-shot output: {len(phases)} phases, {len(shifts)} shifts, "
          f"{len(claims)} claims, {len(tensions)} tensions")
    if missing:
        print(f"  WARNING: {len(missing)} papers not assigned: {', '.join(sorted(missing)[:5])}")

    return result




# ═══════════════════════════════════════════════════════════════════════
# Conversion helpers — one-shot output → existing data models
# ═══════════════════════════════════════════════════════════════════════

def one_shot_result_to_phases(result: dict, field_name: str) -> list[Phase]:
    """Convert one-shot output phases to Phase objects."""
    phases_out = []
    src_phases = result.get("phases", result.get("stages", []))
    for p in src_phases:
        # Find tensions for this phase
        phase_tensions = []
        for t in result.get("tensions", []):
            t_phase = t.get("phase", t.get("stage", -1))
            if t_phase == p.get("index"):
                positions = t.get("positions", [])
                introduced_by = [pos.get("paper", "") for pos in positions]
                phase_tensions.append(Tension(
                    tension=t.get("name", ""),
                    description=t.get("description", ""),
                    introduced_by=introduced_by,
                    resolved_by=[],
                    status="direction_forming",
                    dimension="",
                    domain_scope="",
                ))

        # Find unresolved problem: look at the shift that follows this phase
        unresolved = ""
        for shift in result.get("shifts", []):
            from_p = shift.get("from_phase", shift.get("from_stage", -1))
            if from_p == p.get("index"):
                old_q = shift.get("old_question", shift.get("old_belief", "?"))
                new_q = shift.get("new_question", shift.get("new_belief", "?"))
                unresolved = f"Community moved from asking '{old_q}' to '{new_q}'"
                break

        phases_out.append(Phase(
            name=p.get("name", f"Phase {p.get('index', '?')}"),
            time_range=p.get("year_range", ""),
            core_contradiction=p.get("core_tension", ""),
            key_papers=p.get("papers", []),
            core_debate=p.get("core_tension", ""),
            unresolved_problem=unresolved,
            dominant_question=p.get("dominant_question", p.get("core_belief", "")),
            tensions=phase_tensions,
            status="direction_forming",
        ))

    return phases_out


def one_shot_result_to_shifts(result: dict) -> list[ParadigmShift]:
    """Convert one-shot output shifts to ParadigmShift objects."""
    shifts = []
    for s in result.get("shifts", []):
        shifts.append(ParadigmShift(
            shift_name=s.get("shift_name", ""),
            description=s.get("trigger", s.get("description", "")),
            old_paradigm=s.get("old_question", s.get("old_belief", "")),
            new_paradigm=s.get("new_question", s.get("new_belief", "")),
            catalyst_papers=s.get("catalyst_papers", []),
            magnitude="paradigm_shift",
            level="research_question",
            dimension="",
            year_range="",
        ))
    return shifts


def one_shot_result_to_claims(result: dict) -> list[Claim]:
    """Convert one-shot output claims to Claim objects."""
    claims = []
    for c in result.get("claims", []):
        claims.append(Claim(
            paper_id="",
            paper_title=c.get("paper", ""),
            year=0,
            month=0,
            statement=c.get("statement", ""),
            evidence=c.get("evidence", ""),
            problem_addressed="",
            claim_type="introduces",
            claim_level=c.get("claim_level", "methodological"),
        ))
    return claims
