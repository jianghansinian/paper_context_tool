"""Claim relation builder — classifies pairwise claim relationships.

Core V4 module that identifies how claims relate to each other:
  ATTACK   — B contradicts A's core assumption
  REPLACE  — B renders A's entire paradigm obsolete
  IMPROVE  — B fixes a specific limitation of A
  SUPPORT  — B provides independent evidence for A
  EXTEND   — B applies A to a new domain
  PARALLEL — B addresses a fundamentally different problem from A (independent)

Single-call design: relation classifier directly judges relationship type.
No separate lineage gatekeeper — "parallel" (independent) is just another
relation type the LLM can choose when papers address different problems.

These relations turn a flat claim list into a Claim Evolution Chain,
enabling narrative generation to show "Claim Wars" rather than just
paper-by-paper summaries.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _resolve_model
from paper import Claim, ClaimRelation


# ── System prompt ─────────────────────────────────────────────────────
_CLAIM_RELATION_SYSTEM = """\
You are a research meta-analyst specializing in how scientific claims \
relate to each other across time. You classify the relationship between \
an EARLIER claim and a LATER claim.

Relation types (choose exactly one):
- attack: B directly contradicts or refutes A's core assumption. \
B provides evidence that A's premise was wrong or insufficient.
- replace: B argues that A's entire approach paradigm is unnecessary or \
obsolete. B doesn't just fix A — B says the whole problem framing should change.
- improve: B accepts A's core approach but addresses a specific limitation \
or weakness. B builds on A rather than challenging it.
- support: B provides independent evidence that strengthens A's claim from \
a different angle or setting.
- extend: B applies A's insight to a new domain, task, or problem setting.
- parallel: B addresses a fundamentally different problem from A. The claims \
are independent developments in different research directions.

Key distinctions:
- "attack": "A's assumption X is incorrect" (denies a premise)
- "replace": "Even if X works, the whole approach is wrong" (denies the paradigm)
- "improve": "X is right, but has problem Y" (accepts the premise, fixes a flaw)
- "parallel": "A and B solve different problems; neither builds on nor challenges the other"

CRITICAL: "parallel" means the two papers serve DIFFERENT downstream tasks
(detection vs planning vs prediction). Different approaches to the SAME downstream
task are NOT parallel — they compete in the same arena.

EXAMPLES of NON-parallel (all serve DETECTION):
- Dense BEV detector (A) → Sparse detector (B): "replace" — B says you don't need BEV
- Single-frame detector (A) → Temporal detector (B): "improve" — B adds a dimension
- Depth-supervised detector (A) → Attention-based detector (B): "replace" — B says
  explicit depth labels are unnecessary

EXAMPLES of parallel (serve DIFFERENT tasks):
- Detection paper (A) → Planning paper (B): "parallel" — different end goals
- Perception paper (A) → Prediction paper (B): "parallel" — different end goals

Return ONLY a JSON object. No other text."""

# ── Per-pair prompt ───────────────────────────────────────────────────
_CLAIM_RELATION_PROMPT = """\
Analyze the relationship between Claim A (earlier) and Claim B (later).

CLAIM A [{year_a}] {paper_a}:
Problem: {problem_a}
Claim: "{claim_a}"

CLAIM B [{year_b}] {paper_b}:
Problem: {problem_b}
Claim: "{claim_b}"

STEP 1 — Identify the DOWNSTREAM TASK each paper ultimately serves:
- What is the end goal? (3D object detection? motion prediction? planning? map segmentation?)
- If both papers serve the SAME downstream task, they are in the SAME arena — even if \
  their methods are radically different. Dense BEV detection and sparse detection both \
  serve 3D detection → same arena. Modular planning and end-to-end planning both serve \
  planning → same arena.

STEP 2 — Classify the relationship:
- "parallel": papers serve DIFFERENT downstream tasks (e.g., detection vs planning). \
  Neither paper's contribution affects the other's core problem.
- "replace": B serves the SAME downstream task as A, but argues that A's entire \
  approach (not just a detail) is unnecessary. B says "you don't need to do it that \
  way at all." E.g., sparse detection replacing dense BEV detection — same task, \
  fundamentally different belief about what approach is needed.
- "attack": B directly contradicts a specific assumption A made. B says "A's premise \
  X is wrong." More targeted than replace.
- "improve": B serves the same task, accepts A's overall approach, but fixes a \
  specific limitation.
- "extend": B applies A's approach to a NEW task or domain.
- "support": B provides independent evidence that strengthens A's claim.

CRITICAL: If both papers ultimately serve the SAME downstream task but with \
fundamentally different philosophies (e.g., dense vs sparse, modular vs end-to-end), \
the relationship is likely "replace" or "attack", NOT "parallel".

How does Claim B relate to Claim A? Choose one: attack, replace, improve, \
support, extend, parallel.

Return JSON:
```json
{{
  "relation": "attack|replace|improve|support|extend|parallel",
  "explanation": "One sentence explaining WHY this relation exists."
}}
```"""


# ── Public API ────────────────────────────────────────────────────────

def classify_claim_relation(
    claim_a: Claim,
    claim_b: Claim,
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Optional[ClaimRelation]:
    """Classify the relationship between two claims.

    Returns ClaimRelation or None on failure.
    """
    if client is None:
        from llm_analyzer import build_analyzer_client
        client = build_analyzer_client()
    if not client:
        return None

    model = _resolve_model(model)
    if not model:
        return None

    def _fmt_date(c: Claim) -> str:
        m = getattr(c, 'month', 0)
        return f"{c.year}-{m:02d}" if m > 0 else str(c.year)

    prompt = _CLAIM_RELATION_PROMPT.format(
        year_a=_fmt_date(claim_a),
        paper_a=claim_a.paper_title,
        problem_a=claim_a.problem_addressed,
        claim_a=claim_a.statement,
        year_b=_fmt_date(claim_b),
        paper_b=claim_b.paper_title,
        problem_b=claim_b.problem_addressed,
        claim_b=claim_b.statement,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CLAIM_RELATION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=512,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        if isinstance(data, dict) and "relation" in data:
            return ClaimRelation(
                source_paper=claim_a.paper_title,
                target_paper=claim_b.paper_title,
                source_claim=claim_a.statement,
                target_claim=claim_b.statement,
                relation=data["relation"],
                explanation=data.get("explanation", ""),
                source_year=claim_a.year,
                target_year=claim_b.year,
            )
    except Exception as exc:
        print(f"Claim relation classification failed: {exc}")

    return None


def _guess_downstream_task(paper_title: str, claims: list[Claim]) -> str:
    """Guess paper's downstream task from title and claim problem statements.

    Used to group papers into task cohorts before building relations,
    preventing false causal chains between different tasks (detection vs planning).
    """
    text = paper_title.lower()
    for c in claims:
        text += " " + c.problem_addressed.lower()

    # Planning/prediction keywords
    if any(kw in text for kw in ["planning", "trajectory", "motion prediction",
                                   "end-to-end autonomous", "self-driving"]):
        return "planning"
    # Detection keywords
    if any(kw in text for kw in ["detection", "detector", "3d object", "bevdet",
                                   "sparsebev", "bevformer", "multi-camera 3d",
                                   "sparse4d"]):
        return "detection"
    # Tracking
    if any(kw in text for kw in ["tracking", "multi-object track"]):
        return "tracking"
    # Prediction (use compound terms to avoid false match on "velocity prediction" in detection)
    if any(kw in text for kw in ["trajectory forecast", "motion forecast",
                                   "behavior prediction", "intent prediction"]):
        return "prediction"
    # Mapping/segmentation
    if any(kw in text for kw in ["mapping", "segmentation", "hd map", "lane"]):
        return "mapping"

    return "other"


def build_paper_chain_relations(
    claims_by_paper: dict[str, list[Claim]],
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> list[ClaimRelation]:
    """Build claim relations grouped by downstream task, then chronological within each task.

    V5.2: Papers are first grouped by downstream task (detection/planning/etc.),
    then ordered chronologically within each task. Consecutive papers within the
    same task are compared via LLM. Cross-task pairs are marked parallel without
    an LLM call — this prevents false causal chains like "detection paper → planning
    paper" being narrated as if one led to the other.

    Returns list of ClaimRelation objects.
    """
    # Group papers by downstream task
    task_groups: dict[str, list[tuple[str, list[Claim]]]] = {}
    for pid, claims in claims_by_paper.items():
        task = _guess_downstream_task(pid, claims)
        task_groups.setdefault(task, []).append((pid, claims))

    relations: list[ClaimRelation] = []

    # Within each task group: sort by year and build chains
    for task, papers in task_groups.items():
        papers.sort(key=lambda x: (x[1][0].year, getattr(x[1][0], 'month', 0)))

        for i in range(len(papers) - 1):
            pid_a, claims_a = papers[i]
            pid_b, claims_b = papers[i + 1]
            claim_a = claims_a[0]
            claim_b = claims_b[0]

            result = classify_claim_relation(claim_a, claim_b, client=client, model=model)
            if result:
                relations.append(result)
            else:
                relations.append(ClaimRelation(
                    source_paper=claim_a.paper_title,
                    target_paper=claim_b.paper_title,
                    source_claim=claim_a.statement,
                    target_claim=claim_b.statement,
                    source_year=claim_a.year,
                    target_year=claim_b.year,
                    relation="unknown",
                    explanation="Classification failed",
                ))

    # Cross-task pairs: add parallel edges for consecutive papers across task boundaries
    # Sort all papers by year for cross-task linking
    all_sorted = sorted(claims_by_paper.items(), key=lambda x: (x[1][0].year, getattr(x[1][0], 'month', 0)))
    for i in range(len(all_sorted) - 1):
        pid_a, claims_a = all_sorted[i]
        pid_b, claims_b = all_sorted[i + 1]
        task_a = _guess_downstream_task(pid_a, claims_a)
        task_b = _guess_downstream_task(pid_b, claims_b)

        if task_a != task_b:
            claim_a = claims_a[0]
            claim_b = claims_b[0]
            # Check if this cross-task pair already has a relation
            already_related = any(
                (r.source_paper == claim_a.paper_title and r.target_paper == claim_b.paper_title)
                for r in relations
            )
            if not already_related:
                relations.append(ClaimRelation(
                    source_paper=claim_a.paper_title,
                    target_paper=claim_b.paper_title,
                    source_claim=claim_a.statement,
                    target_claim=claim_b.statement,
                    source_year=claim_a.year,
                    target_year=claim_b.year,
                    relation="parallel",
                    explanation=f"Different downstream tasks — {task_a} vs {task_b}",
                ))

    return relations


def relations_to_text(relations: list[ClaimRelation]) -> str:
    """Format claim relations as readable text for narrative prompt injection."""
    if not relations:
        return ""

    lines = ["CLAIM EVOLUTION CHAIN (how each paper's primary claim relates to the prior one):"]
    for r in relations:
        rel_upper = r.relation.upper()
        label = rel_upper
        if r.relation == "parallel":
            label = "PARALLEL (independent development — do NOT narrate as causal)"
        lines.append(
            f"  {r.source_paper} → {r.target_paper}: {label}"
        )
        lines.append(f"    Explanation: {r.explanation}")
    return "\n".join(lines)


