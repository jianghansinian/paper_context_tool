"""Narrative builder — the product core of V8.

Generates a field's technical evolution story organized by PHASE (time-based
tension clusters), with CAUSAL CHAINS linking phases together.

V8 architecture: Phase-Centric with causal chain.
- Phase is the chapter unit (2-4 phases per field)
- Each phase is a time period with a core contradiction
- Phase N's unresolved_problem → Phase N+1's core_contradiction
- Tensions and RQs are Phase content, not chapter titles
- V5 narrative arc preserved within each phase
"""
from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _resolve_model
from paper import (
    Claim, ClaimRelation, Tension, ParadigmShift, Direction,
    NarrativeSection, ResearchNarrative, ResearchQuestion, Phase,
)


# ── System prompt: professor teaching a lecture ────────────────────────
_NARRATIVE_SYSTEM = """\
You are a historian of science writing for an audience that wants to understand \
HOW ideas evolved — not a chronological catalog of papers.

WRITING MODEL: Idea-Centric History
- Each phase is defined by its CORE IDEA (what the field DISCOVERED), not by \
  which papers happened to be published in that period.
- Papers are FOOTNOTES to ideas, not the story itself. Mention them in passing: \
  "X showed Y (Paper, YEAR)." Do NOT give each paper its own subsection with \
  contribution/limitation bullet points.
- Lead with the INSIGHT, then name the paper that crystallized it. NOT the reverse.
- Example of idea-centric writing:
  "The key discovery of this phase was that sparsity itself was not the \
  bottleneck — adaptability was. Sparse4D (2022) proved sparse detection \
  was viable. SparseBEV (2023) then showed it could surpass dense methods. \
  Sparse4Dv2 solved the remaining temporal efficiency problem."
  This is one paragraph. Three papers, one idea. No bullet points.

TIME DISCIPLINE — CRITICAL:
- ONLY reference papers that exist WITHIN or BEFORE this phase's time range.
- If a phase ends in 2022, do NOT mention results from 2023 or 2024 papers.
- You may mention that a problem REMAINS UNSOLVED, but do not name future solvers.

FORMAT:
- **Bold** core concepts and paper names on first mention. NOT every metric value.
- Paragraphs: 2-4 sentences MAX.
- Each phase: CORE QUESTION (bold) → SETUP (1-2 sentences) → CORE DISCOVERY \
  (idea, with paper examples inline) → TURNING POINT (1 sentence) → \
  **TAKEAWAY** (one bold sentence) → UNSOLVED (1-2 sentences).
- Do NOT repeat the same evidence multiple times in the same phase.
- Avoid listing paper contributions/limitations as bullet points.

Return ONLY a JSON object. No other text."""


# ── Per-phase narrative prompt (V8: DeepSeek-style lecture) ────────────
_PHASE_NARRATIVE_PROMPT = """\
Write a concise historical account of this phase in {field_name}.

THIS PHASE: **{phase_name}** ({time_range})

TIME DISCIPLINE: ONLY papers from {time_range} or earlier exist in this phase.

PREVIOUS UNSOLVED PROBLEM: "{prev_unresolved}"

CORE CONTRADICTION: {core_contradiction}
CORE DEBATE: {core_debate}

KEY PAPERS (these are the facts — use them as evidence, not as the story):
{key_papers}

KEY TENSIONS:
{tensions_text}

EVOLUTIONARY RELATIONSHIPS (within this phase):
{claim_relations}

---
WRITE THE NARRATIVE as a concise historical synthesis, following this structure:

1. **CORE QUESTION** — Bold standalone sentence. The question this phase wrestled with.

2. SETUP — {setup_instruction}

3. CORE DISCOVERY — This is the heart of the narrative. Lead with the IDEA, not \
   the papers:
   - First state what the phase DISCOVERED or RESOLVED (the conceptual advance).
   - Then give examples: mention 2-4 key papers inline, each in ONE sentence: \
     "X demonstrated Y (**Paper**, YEAR)."
   - Group papers that share the same conclusion into a single idea statement.
   - Do NOT write "Paper A... then Paper B... then Paper C" — start with the \
     common thread and use papers as supporting examples.
   - Mention a metric ONLY if it's the turning-point number. Don't list NDS for \
     every paper.

4. TURNING POINT — 1 sentence. What specific result shifted the debate?

5. **TAKEAWAY** — One bold memorable sentence. Like "Sparsity was never the \
   problem — adaptability was."

6. UNSOLVED — 1 sentence ending: "But this created a new problem: {unresolved_problem}"

AVOID:
- Paper-by-paper summaries with contribution/limitation bullet points
- Repeating the same evidence that appears in the Direction block
- Listing every metric for every paper

FORMAT: Use paragraph breaks (blank line) between each section (CORE QUESTION, SETUP,
CORE DISCOVERY, TURNING POINT, TAKEAWAY, UNSOLVED). The CORE DISCOVERY should be
2-3 paragraphs with natural breaks between idea groups.

Return JSON:
```json
{{"narrative": "..."}}
```"""


# ── Field overview prompt (V8: Phase预告) ─────────────────────────────
_FIELD_OVERVIEW_PROMPT = """\
Write a ONE-PARAGRAPH field overview for {field_name}. Be concise and structured.

PHASES:
{phases_text}

PARADIGM SHIFTS:
{shifts_text}

Write exactly ONE paragraph (4-6 sentences) that covers:
- The field's starting point and first breakthrough
- The N phases (name each, 1 sentence each)
- The overall trajectory (dense→sparse, modular→end-to-end, etc.)

Use **bold** for phase names and key concepts. No bullet points, no section breaks.

Return JSON:
```json
{{"overview": "..."}}
```"""


# ── Synthesis prompt ──────────────────────────────────────────────────
_SYNTHESIS_PROMPT = """\
Write the concluding section for {field_name}. Three outputs needed.

PHASE SUMMARIES:
{phase_summaries}

ALL CLAIMS (for reading list):
{claims_text}

1. SYNTHESIS — ONE paragraph (3-4 sentences) summarizing the overall trajectory \
   across all phases. What changed and why? Use **bold** for key concepts.

2. OPEN QUESTIONS — 0-3 questions the field still hasn't resolved, based on \
   the unsolved problems at the end of each phase. Return as a list of strings. \
   If no clear open questions, return an empty list.

3. READING LIST — Group key papers by phase. For each paper, provide: \
   title, year, one-sentence contribution. Return as a list of objects: \
   [{{"phase": "Phase Name", "title": "...", "year": 2022, "contribution": "..."}}]
   Include at most 12 papers total across all phases — pick the most important.

Return JSON:
```json
{{
  "synthesis": "...",
  "open_questions": ["question 1?", "question 2?"],
  "reading_list": [{{"phase": "...", "title": "...", "year": 2022, "contribution": "..."}}]
}}
```"""


# ── Public API ────────────────────────────────────────────────────────


def build_narrative(
    claims: list[Claim],
    claim_relations: list,
    research_questions: list[ResearchQuestion],
    tensions: list,
    paradigm_shifts: list,
    phases: list[Phase] | None = None,
    field_name: str = "BEV Perception",
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
    seed_paper_id: Optional[str] = None,
) -> Optional[ResearchNarrative]:
    """Build a V8 research narrative organized by Phase.

    V8 architecture: each Phase becomes one NarrativeSection, linked by
    causal chain (Phase N's unresolved_problem → Phase N+1).
    RQs and Tensions are Phase content, not chapter titles.
    """
    if client is None:
        client = _build_client()

    model = _resolve_model(model)
    if not model:
        print("Narrative builder: no model available")
        return None

    # Use phases if provided; otherwise fall back to RQ-based (backward compat)
    if not phases:
        print("Narrative builder: no phases, falling back to RQ-based sections")
        return _build_rq_based(
            claims, claim_relations, research_questions, tensions,
            paradigm_shifts, field_name, client, model, seed_paper_id)

    # Index claims by paper title
    paper_titles_to_claims: dict[str, list[Claim]] = {}
    for c in claims:
        paper_titles_to_claims.setdefault(c.paper_title, []).append(c)

    # Collect all phase paper titles for matching
    all_phase_papers: set[str] = set()
    for p in phases:
        for kp in p.key_papers:
            all_phase_papers.add(kp)

    # Build one section per phase
    sections: list[NarrativeSection] = []
    section_narratives: dict[str, str] = {}

    for i, phase in enumerate(phases):
        prev_unresolved = phases[i - 1].unresolved_problem if i > 0 else (
            "NONE — this is the FIRST phase. Set up the field's original motivation "
            "directly instead of referencing any previous work."
        )

        # Gather papers involved in this phase
        involved_papers: set[str] = set(phase.key_papers)
        # Also add papers from phase's tensions
        for t in phase.tensions:
            for name in t.introduced_by + t.resolved_by:
                involved_papers.add(name)

        # Parse phase end year for time filtering
        phase_max_year = 9999
        try:
            parts = phase.time_range.split("-")
            if len(parts) == 2:
                phase_max_year = int(parts[1].strip())
        except (ValueError, AttributeError):
            pass

        # Collect claims for these papers, filtered by phase time range
        section_claims: list[Claim] = []
        seen_ids: set[str] = set()
        paper_to_year: dict[str, int] = {}
        for paper_title in involved_papers:
            matched = _fuzzy_match_paper(paper_title, paper_titles_to_claims)
            if matched:
                for c in paper_titles_to_claims[matched]:
                    paper_to_year[matched] = c.year
                    if c.paper_id not in seen_ids and c.year <= phase_max_year:
                        section_claims.append(c)
                        seen_ids.add(c.paper_id)
        section_claims.sort(key=lambda c: c.year)

        # Filter involved_papers to only those within time range
        involved_papers = {p for p in involved_papers
                          if paper_to_year.get(_fuzzy_match_paper(p, paper_titles_to_claims) or "", 9999) <= phase_max_year}
        # Exclude papers that are key_papers of strictly later phases (boundary guard)
        later_key_papers: set[str] = set()
        for j in range(i + 1, len(phases)):
            later_key_papers.update(t.lower() for t in phases[j].key_papers)
        involved_papers = {p for p in involved_papers
                          if p.lower() not in later_key_papers}

        # Find relevant relations — BOTH papers must be in-phase (TIME DISCIPLINE)
        section_relations = []
        for r in claim_relations:
            src = r.source_paper if hasattr(r, 'source_paper') else r['source_paper']
            tgt = r.target_paper if hasattr(r, 'target_paper') else r['target_paper']
            if src in involved_papers and tgt in involved_papers:
                section_relations.append(r)

        # Find relevant RQs (simple keyword overlap with phase debate/contradiction)
        section_rqs = _match_rqs_to_phase(research_questions, phase)

        # Aggregate direction from RQs (phase-filtered)
        section_direction = _aggregate_direction(section_rqs, phase, section_claims)

        # Find relevant paradigm shifts
        section_shifts = _match_shifts_to_phase(paradigm_shifts, phase)

        narrative_text = _generate_phase_narrative(
            client, model,
            field_name=field_name,
            phase=phase,
            prev_unresolved=prev_unresolved,
            is_first=(i == 0),
            is_last=(i == len(phases) - 1),
            tensions=phase.tensions,
            research_questions=section_rqs,
            direction=section_direction,
            claims=section_claims,
            claim_relations=section_relations,
        )

        section = NarrativeSection(
            title=f"Phase {i + 1}: {phase.name} ({phase.time_range})",
            phase=phase,
            claims=section_claims,
            claim_relations=section_relations,
            paradigm_shifts=section_shifts,
            direction=section_direction,
            narrative=narrative_text,
        )
        sections.append(section)
        section_narratives[phase.name] = narrative_text

    if not sections:
        print("Narrative builder: no sections generated")
        return None

    # Generate field overview (Phase-based)
    overview = _generate_field_overview_v8(
        client, model, field_name, phases, research_questions, paradigm_shifts)

    # Generate synthesis + open questions + reading list
    synthesis, open_questions, reading_list = _generate_synthesis_v8(
        client, model, field_name, sections, section_narratives, overview, claims)

    return ResearchNarrative(
        field_name=field_name,
        seed_paper_id=seed_paper_id,
        overview=overview,
        sections=sections,
        phases=phases,
        paradigm_shifts=paradigm_shifts if isinstance(paradigm_shifts, list) else [],
        research_questions=research_questions,
        tensions=tensions if isinstance(tensions, list) else [],
        claims=claims,
        claim_relations=claim_relations if isinstance(claim_relations, list) else [],
        open_questions=open_questions,
        reading_list=reading_list,
        synthesis=synthesis,
    )


def _build_rq_based(
    claims, claim_relations, research_questions, tensions,
    paradigm_shifts, field_name, client, model, seed_paper_id,
) -> Optional[ResearchNarrative]:
    """Fallback: RQ-based sections (V6/V7 backward compat)."""
    from research_question_detector import filter_lecture_rqs
    lecture_rqs = filter_lecture_rqs(research_questions)
    if not lecture_rqs:
        return None

    paper_titles_to_claims = {}
    for c in claims:
        paper_titles_to_claims.setdefault(c.paper_title, []).append(c)

    sections = []
    for rq in lecture_rqs:
        involved_papers = set(rq.introduced_by)
        for pos in rq.positions:
            if isinstance(pos, dict):
                involved_papers.add(pos.get("paper", ""))
        involved_papers.discard("")

        section_claims = []
        seen_ids = set()
        for paper_title in involved_papers:
            matched = _fuzzy_match_paper(paper_title, paper_titles_to_claims)
            if matched:
                for c in paper_titles_to_claims[matched]:
                    if c.paper_id not in seen_ids:
                        section_claims.append(c)
                        seen_ids.add(c.paper_id)
        section_claims.sort(key=lambda c: c.year)

        section_relations = []
        for r in claim_relations:
            src = r.source_paper if hasattr(r, 'source_paper') else r['source_paper']
            tgt = r.target_paper if hasattr(r, 'target_paper') else r['target_paper']
            if src in involved_papers or tgt in involved_papers:
                section_relations.append(r)

        # Create a synthetic Phase for the section
        phase = Phase(
            name=rq.short_name or rq.question[:50],
            time_range="2020-2024",
            core_contradiction=rq.description,
            key_papers=list(involved_papers)[:6],
            core_debate=rq.question,
            unresolved_problem="(see synthesis)",
            tensions=rq.tensions if rq.tensions else [],
        )

        narrative_text = _generate_phase_narrative(
            client, model, field_name=field_name, phase=phase,
            prev_unresolved="(see field overview)", is_first=False, is_last=False,
            tensions=rq.tensions, research_questions=[rq],
            direction=rq.direction, claims=section_claims,
            claim_relations=section_relations,
        )

        section = NarrativeSection(
            title=rq.question,
            phase=phase,
            claims=section_claims,
            claim_relations=section_relations,
            paradigm_shifts=[],
            direction=rq.direction,
            narrative=narrative_text,
        )
        sections.append(section)

    overview = _generate_field_overview_v8(
        client, model, field_name, [], research_questions, paradigm_shifts)

    return ResearchNarrative(
        field_name=field_name, seed_paper_id=seed_paper_id,
        overview=overview, sections=sections,
        paradigm_shifts=paradigm_shifts if isinstance(paradigm_shifts, list) else [],
        research_questions=research_questions,
        tensions=tensions if isinstance(tensions, list) else [],
        claims=claims,
        claim_relations=claim_relations if isinstance(claim_relations, list) else [],
        synthesis="",
    )


# ── Internal helpers ──────────────────────────────────────────────────

def _build_client() -> Optional[OpenAI]:
    try:
        from llm_analyzer import build_analyzer_client
        return build_analyzer_client()
    except Exception:
        return None


def _fuzzy_match_paper(name: str, paper_index: dict[str, list]) -> Optional[str]:
    """Match a paper name to keys in the index, with fuzzy matching."""
    if name in paper_index:
        return name
    name_lower = name.lower().strip()
    for key in paper_index:
        key_lower = key.lower().strip()
        if name_lower == key_lower:
            return key
        if name_lower in key_lower or key_lower in name_lower:
            return key
        # Extract short name (before colon)
        name_short = name_lower.split(":")[0].strip()
        key_short = key_lower.split(":")[0].strip()
        if name_short == key_short:
            return key
    return None


def _match_rqs_to_phase(rqs: list[ResearchQuestion], phase: Phase) -> list[ResearchQuestion]:
    """Find RQs relevant to a phase by keyword overlap."""
    if not rqs:
        return []
    phase_text = (phase.core_contradiction + " " + phase.core_debate).lower()
    relevant = []
    for rq in rqs:
        rq_text = (rq.question + " " + rq.description).lower()
        rq_words = set(rq_text.split()) - {"the", "a", "an", "is", "of", "in", "to", "for", "do", "we", "can", "how", "what", "does", "did", "?"}
        p_words = set(phase_text.split()) - {"the", "a", "an", "is", "of", "in", "to", "for"}
        if len(rq_words & p_words) >= 2:
            relevant.append(rq)
    return relevant


def _match_shifts_to_phase(shifts: list, phase: Phase) -> list:
    """Find paradigm shifts relevant to a phase."""
    if not shifts:
        return []
    phase_text = (phase.core_contradiction + " " + phase.core_debate).lower()
    relevant = []
    for s in shifts:
        if hasattr(s, 'shift_name'):
            s_text = (s.shift_name + " " + s.description).lower()
        else:
            s_text = (s.get('shift_name', '') + " " + s.get('description', '')).lower()
        p_words = set(phase_text.split()) - {"the", "a", "an", "is", "of", "in", "to", "for"}
        s_words = set(s_text.split()) - {"the", "a", "an", "is", "of", "in", "to", "for"}
        if len(p_words & s_words) >= 3:
            relevant.append(s)
    return relevant


def _aggregate_direction(rqs: list[ResearchQuestion], phase: Phase, claims: list[Claim]) -> Optional[Direction]:
    """Aggregate direction from the best-matching RQ, filtered to this phase's time range.

    Picks the RQ with the most keyword overlap with the phase's core_contradiction +
    core_debate. Filters support/opposing papers AND evidence_summary text to only
    reference papers within the phase's time range.
    """
    if not rqs:
        return None

    # Parse phase end year from time_range
    max_year = 9999
    try:
        parts = phase.time_range.split("-")
        if len(parts) == 2:
            max_year = int(parts[1].strip())
    except (ValueError, AttributeError):
        pass

    # Collect all paper titles that exist before/at max_year
    valid_titles: set[str] = set()
    for c in claims:
        if c.year <= max_year:
            valid_titles.add(c.paper_title.lower())

    # Score RQs by keyword overlap with phase's core contradiction + debate
    phase_text = (phase.core_contradiction + " " + phase.core_debate).lower()
    _stop = {"the", "a", "an", "is", "of", "in", "to", "for", "do", "we",
             "can", "how", "what", "does", "did", "and", "or", "that", "this",
             "its", "has", "been", "was", "are", "be", "not", "no", "but"}
    p_words = set(phase_text.split()) - _stop

    best_rq: Optional[ResearchQuestion] = None
    best_score = 0
    for rq in rqs:
        if not rq.direction:
            continue
        rq_text = (rq.question + " " + rq.description + " " + rq.direction.statement).lower()
        rq_words = set(rq_text.split()) - _stop
        score = len(p_words & rq_words)
        if score > best_score:
            # Verify this RQ has at least one paper in this phase
            d = rq.direction
            support = [p for p in d.support_papers if p.lower() in valid_titles]
            opposing = [p for p in d.opposing_papers if p.lower() in valid_titles]
            if support or opposing:
                best_score = score
                best_rq = rq

    if best_rq and best_rq.direction:
        d = best_rq.direction
        support = [p for p in d.support_papers if p.lower() in valid_titles]
        opposing = [p for p in d.opposing_papers if p.lower() in valid_titles]

        # Adjust statement when phase evidence contradicts field-wide direction.
        # If the phase's own key papers are predominantly on the "opposing" side,
        # the direction statement should reflect the phase's actual position.
        statement = d.statement
        confidence = d.confidence
        phase_key_titles = {t.lower() for t in phase.key_papers}
        n_support_in_phase = len([p for p in support if p.lower() in phase_key_titles])
        n_oppose_in_phase = len([p for p in opposing if p.lower() in phase_key_titles])

        if opposing and not support:
            statement = (
                f"Within this phase, the evidence did NOT yet point toward "
                f"the eventual field consensus. "
                f"The papers here predominantly explored the opposing position."
            )
            confidence = "low"
        elif n_oppose_in_phase > n_support_in_phase and len(opposing) >= len(support):
            statement = (
                f"Within this phase, the dominant approach was the OPPOSITE of "
                f"the eventual field direction. The current phase still "
                f"operated under the older paradigm."
            )
            confidence = "low"

        # Filter evidence_summary to phase-relevant papers
        filtered_evidence = _filter_evidence_text(d.evidence_summary, valid_titles, support, opposing)

        return Direction(
            statement=statement,
            support_papers=support,
            opposing_papers=opposing,
            confidence=confidence,
            evidence_summary=filtered_evidence,
        )
    return None


def _filter_evidence_text(evidence: str, valid_titles: set[str], support: list[str], opposing: list[str]) -> str:
    """Filter evidence to only reference papers within this phase's time range.

    If the evidence mentions papers not in valid_titles (future papers), those
    references are stripped. If the evidence is entirely about future papers,
    return a note that evidence in this phase is limited.
    """
    if not evidence:
        return ""
    evidence_lower = evidence.lower()
    # Check if evidence mentions paper names that are known and in-phase
    in_phase_mentioned = any(t in evidence_lower for t in valid_titles if len(t) > 5)
    # If no in-phase papers are mentioned, evidence is likely from outside this phase
    if not in_phase_mentioned and (support or opposing):
        return f"Evidence drawn from papers within this phase: {', '.join((support + opposing)[:4])}"
    return evidence


def _format_tensions_for_phase(tensions: list) -> str:
    """Format tensions for phase narrative prompt."""
    if not tensions:
        return "(No specific tensions — use the core debate to structure the narrative)"

    lines = ["TENSIONS within this phase (your narrative arc):"]
    for i, t in enumerate(tensions, 1):
        if hasattr(t, 'tension'):
            name, desc, intro, resolv = t.tension, t.description, t.introduced_by, t.resolved_by
        else:
            name = t.get('tension', '')
            desc = t.get('description', '')
            intro = t.get('introduced_by', [])
            resolv = t.get('resolved_by', [])

        lines.append(f"\n  Tension {i}: {name}")
        lines.append(f"    Description: {desc}")
        lines.append(f"    Introduced by: {', '.join(intro[:3])}")
        lines.append(f"    Advanced by: {', '.join(resolv[:3])}")
    return "\n".join(lines)


def _format_rqs_for_phase(rqs: list[ResearchQuestion]) -> str:
    """Format research questions for phase narrative prompt."""
    if not rqs:
        return "(No specific research questions identified for this phase)"

    lines = ["RESEARCH QUESTIONS debated in this phase:"]
    for rq in rqs:
        lines.append(f"  Q: {rq.question}")
        if rq.direction:
            lines.append(f"     Direction: {rq.direction.statement}")
    return "\n".join(lines)


def _format_direction_for_phase(direction, phase_time_range: str = "") -> str:
    """Format direction for phase — compact 3-line output."""
    if not direction:
        return "(No structured direction)"

    if hasattr(direction, 'statement'):
        stmt, sup, opp, conf = (
            direction.statement, direction.support_papers,
            direction.opposing_papers, direction.confidence)
    else:
        stmt = direction.get('statement', '')
        sup = direction.get('support_papers', [])
        opp = direction.get('opposing_papers', [])
        conf = direction.get('confidence', 'medium')

    all_papers = sup[:3] + opp[:3]
    why = ", ".join(all_papers) if all_papers else "(none)"

    return "\n".join([
        f"Direction: {stmt}",
        f"Confidence: {conf}",
        f"Why: {why}",
    ])


def _format_relations_for_prompt(relations: list) -> str:
    """Format claim relations for prompt injection."""
    if not relations:
        return "(No relationships classified)"

    lines = [
        "Paper-to-paper claim relationships:",
        "ATTACK/REPLACE = paradigm clash. PARALLEL = independent developments.",
    ]
    for r in relations:
        if hasattr(r, 'source_paper'):
            src, tgt, rel, expl = r.source_paper, r.target_paper, r.relation, r.explanation
        else:
            src, tgt, rel, expl = r['source_paper'], r['target_paper'], r['relation'], r['explanation']
        rel_upper = rel.upper()
        if rel == "parallel":
            rel_upper = "PARALLEL (independent — do not chain causally)"
        lines.append(f"  {src} -> {tgt}: {rel_upper}")
        lines.append(f"    {expl}")
    return "\n".join(lines)


def _generate_phase_narrative(
    client: OpenAI,
    model: str,
    *,
    field_name: str,
    phase: Phase,
    prev_unresolved: str,
    is_first: bool = False,
    is_last: bool,
    tensions: list,
    research_questions: list[ResearchQuestion],
    direction,
    claims: list[Claim],
    claim_relations: list,
) -> str:
    """Generate professor-style lecture narrative for one phase."""
    tensions_text = _format_tensions_for_phase(tensions)
    questions_text = _format_rqs_for_phase(research_questions)
    direction_text = _format_direction_for_phase(direction, phase.time_range)
    relations_text = _format_relations_for_prompt(claim_relations)
    key_papers_text = ", ".join(phase.key_papers)

    prompt = _PHASE_NARRATIVE_PROMPT.format(
        field_name=field_name,
        phase_name=phase.name,
        time_range=phase.time_range,
        prev_unresolved=prev_unresolved,
        setup_instruction=(
            "1-2 sentences. Briefly reference the previous phase's unsolved problem "
            "(see PREVIOUS UNSOLVED PROBLEM above)."
            if not is_first
            else "SKIP this step. Start the narrative directly from the field's "
                 "ORIGINAL MOTIVATION — no 'previous phase' exists."
        ),
        core_contradiction=phase.core_contradiction,
        core_debate=phase.core_debate,
        key_papers=key_papers_text,
        tensions_text=tensions_text,
        questions_text=questions_text,
        direction_text=direction_text,
        claim_relations=relations_text,
        unresolved_problem=phase.unresolved_problem,
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
        print(f"Phase narrative failed for '{phase.name}': {exc}")
        return ""


def _generate_field_overview_v8(
    client: OpenAI,
    model: str,
    field_name: str,
    phases: list[Phase],
    research_questions: list[ResearchQuestion],
    paradigm_shifts: list,
) -> str:
    """Generate field-level overview: Phase预告."""
    if phases:
        from tension_detector import phases_to_text
        phases_text = phases_to_text(phases)
    else:
        phases_text = "(Phases not yet detected — organized by research questions)"
        phases_text += "\n"
        for i, rq in enumerate(research_questions, 1):
            phases_text += f"  Q{i}: {rq.question}\n"

    questions_text = ""
    for i, rq in enumerate(research_questions, 1):
        questions_text += (
            f"Q{i} [{rq.level.upper()}] {rq.question}\n"
            f"   Context: {rq.description}\n"
            f"   Status: {rq.status}\n"
        )

    shifts_text = ""
    if paradigm_shifts:
        s_lines = []
        for s in paradigm_shifts:
            if hasattr(s, 'shift_name'):
                s_lines.append(f"  {s.shift_name}: {s.description[:150]}")
            else:
                s_lines.append(f"  {s.get('shift_name', '?')}: {s.get('description', '')[:150]}")
        shifts_text = "\n".join(s_lines)

    prompt = _FIELD_OVERVIEW_PROMPT.format(
        field_name=field_name,
        phases_text=phases_text,
        questions_text=questions_text,
        shifts_text=shifts_text,
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


def _generate_synthesis_v8(
    client: OpenAI,
    model: str,
    field_name: str,
    sections: list[NarrativeSection],
    section_narratives: dict[str, str],
    overview: str,
    claims: list[Claim],
):
    """Generate synthesis, open questions, and reading list."""
    phase_summaries = "\n\n".join(
        f"--- {s.title} ---\n{section_narratives.get(s.phase.name if hasattr(s, 'phase') and s.phase else s.title, '')}"
        for s in sections
    )

    # Build claims text for reading list generation
    claims_lines = []
    for c in claims:
        claims_lines.append(f"[{c.year}] {c.paper_title}: {c.statement[:120]}")
    claims_text = "\n".join(claims_lines[:30])  # limit to avoid prompt bloat

    prompt = _SYNTHESIS_PROMPT.format(
        field_name=field_name,
        phase_summaries=phase_summaries,
        claims_text=claims_text,
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
        synthesis = _parse_json_field(raw, "synthesis") or ""
        open_questions = _parse_json_list(raw, "open_questions") or []
        reading_list = _parse_json_list_of_dicts(raw, "reading_list") or []
        return synthesis, open_questions, reading_list
    except Exception as exc:
        print(f"Synthesis generation failed: {exc}")
        return "", [], []


def _parse_json_field(raw: str, field: str) -> Optional[str]:
    """Extract a string field from a JSON response."""
    if not raw or not raw.strip():
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return data.get(field, "").strip() or None
    except json.JSONDecodeError:
        pattern = rf'"{field}"\s*:\s*"((?:[^"\\]|\\.)*)"'
        m = re.search(pattern, raw, re.DOTALL)
        if m:
            return m.group(1).replace("\\n", "\n").replace('\\"', '"')
    return None


def _parse_json_list(raw: str, field: str) -> Optional[list]:
    """Extract a list field from a JSON response."""
    if not raw or not raw.strip():
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            val = data.get(field, [])
            return val if isinstance(val, list) else []
    except json.JSONDecodeError:
        pass
    return []


def _parse_json_list_of_dicts(raw: str, field: str) -> Optional[list[dict]]:
    """Extract a list of dicts field from a JSON response."""
    if not raw or not raw.strip():
        return None
    raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    raw = re.sub(r"\s*```$", "", raw)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            val = data.get(field, [])
            if isinstance(val, list):
                return [item for item in val if isinstance(item, dict)]
    except json.JSONDecodeError:
        pass
    return []
