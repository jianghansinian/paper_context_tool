"""Claim relation builder — classifies pairwise claim relationships.

Core V4 module that identifies how claims relate to each other:
  ATTACK   — B contradicts A's core assumption
  REPLACE  — B renders A's entire paradigm obsolete
  IMPROVE  — B fixes a specific limitation of A
  SUPPORT  — B provides independent evidence for A
  EXTEND   — B applies A to a new domain
  INDEPENDENT — unrelated claims

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
from paper import Claim

# ── Research lineage classifier ───────────────────────────────────────
# Added P0 fix: before classifying HOW two claims relate, first ask IF they
# belong to the same research lineage. This prevents false causal chains
# between parallel developments (e.g., Sparse4D → VAD is NOT causal).

_LINEAGE_SYSTEM = """\
You are a research historian who determines whether two papers belong to the \
same research lineage. A "research lineage" means both papers attempt to solve \
ESSENTIALLY THE SAME PROBLEM at a high level.

For example:
- "How to build better BEV features from multi-camera images?" → BEVDet, \
  BEVDepth, BEVFormer, BEVDet4D all share this lineage
- "How to do perception via sparse queries?" → Sparse4D, SparseBEV share this
- "How to unify perception, prediction, and planning?" → UniAD, VAD, \
  SparseDrive share this

SENSITIVITY RULE — When to answer YES, NO, or UNCERTAIN:

Use NO ONLY when papers address FUNDAMENTALLY DIFFERENT DOMAINS where one \
paper's contribution has no bearing on the other's core problem:
- BEVFormer (perception representation) vs VAD (planning representation) → NO
- Sparse4D (sparse detection) vs UniAD (end-to-end planning) → NO
- BEVDepth (depth estimation) vs SparseDrive (sparse planning) → NO

Use UNCERTAIN when papers share the SAME HIGHER-LEVEL GOAL but address it from \
different angles. The bar for NO is HIGH — most papers in the same field share \
some common ground. If the papers could reasonably cite each other for technical \
reasons (not just as background), lean toward YES or UNCERTAIN:
- BEVDet4D (temporal cues for BEV detection) vs BEVFormer (spatiotemporal \
  transformers for BEV) → UNCERTAIN or YES (both address BEV perception, just \
  different mechanisms: explicit feature fusion vs cross-attention)
- BEVDepth (depth supervision) vs BEVDet4D (temporal fusion) → UNCERTAIN \
  (both aim to improve BEV detection, different sub-problems)
- SparseBEV (sparse detection) vs Sparse4Dv2 (sparse temporal fusion) → YES \
  (both advance the sparse detection paradigm, different aspects)

Key heuristic: if BOTH papers ultimately aim to improve the SAME downstream \
task (detection, planning, prediction) using related representational choices, \
they share a research lineage at the field level. Only answer NO when the \
papers belong to different TASK domains entirely (detection vs planning).

Return ONLY a JSON object. No other text."""

_LINEAGE_PROMPT = """\
Do these two papers attempt to solve essentially the same research problem?

PAPER A [{year_a}] {paper_a}:
Problem addressed: {problem_a}
Claim: "{claim_a}"

PAPER B [{year_b}] {paper_b}:
Problem addressed: {problem_b}
Claim: "{claim_b}"

First, identify the CORE RESEARCH PROBLEM each paper addresses in 1 sentence.
Then answer: do they share the same research lineage?

Return JSON:
```json
{{
  "problem_a": "string (1 sentence: the core problem Paper A addresses)",
  "problem_b": "string (1 sentence: the core problem Paper B addresses)",
  "same_lineage": "YES|NO|UNCERTAIN",
  "reason": "One sentence explanation."
}}
```"""


def classify_same_lineage(
    claim_a: Claim,
    claim_b: Claim,
    *,
    client=None,
    model=None,
):
    """Determine if two papers belong to the same research lineage.

    Returns {"same_lineage": "YES|NO|UNCERTAIN", "reason": str, "problem_a": str, "problem_b": str}
    or None on failure.
    """
    if client is None:
        from llm_analyzer import build_analyzer_client
        client = build_analyzer_client()
    if not client:
        return None

    model = _resolve_model(model)
    if not model:
        return None

    prompt = _LINEAGE_PROMPT.format(
        year_a=claim_a.year,
        paper_a=claim_a.paper_title,
        problem_a=claim_a.problem_addressed,
        claim_a=claim_a.statement,
        year_b=claim_b.year,
        paper_b=claim_b.paper_title,
        problem_b=claim_b.problem_addressed,
        claim_b=claim_b.statement,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _LINEAGE_SYSTEM},
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
        if isinstance(data, dict) and "same_lineage" in data:
            return data
    except Exception as exc:
        print(f"Lineage classification failed: {exc}")

    return None


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
- independent: B addresses a fundamentally different problem; the claims \
do not interact.

Key distinction:
- "attack": "A's assumption X is incorrect" (denies a premise)
- "replace": "Even if X works, the whole approach is wrong" (denies the paradigm)
- "improve": "X is right, but has problem Y" (accepts the premise, fixes a flaw)

Return ONLY a JSON object. No other text."""

# ── Per-pair prompt ───────────────────────────────────────────────────
_CLAIM_RELATION_PROMPT = """\
Analyze the relationship between Claim A (earlier) and Claim B (later).

CLAIM A [{year_a}] {paper_a}:
"{claim_a}"

CLAIM B [{year_b}] {paper_b}:
"{claim_b}"

How does Claim B relate to Claim A? Choose one: attack, replace, improve, \
support, extend, independent.

Return JSON:
```json
{{
  "relation": "attack|replace|improve|support|extend|independent",
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
) -> Optional[dict]:
    """Classify the relationship between two claims.

    Returns {"relation": str, "explanation": str} or None on failure.
    """
    if client is None:
        from llm_analyzer import build_analyzer_client
        client = build_analyzer_client()
    if not client:
        return None

    model = _resolve_model(model)
    if not model:
        return None

    prompt = _CLAIM_RELATION_PROMPT.format(
        year_a=claim_a.year,
        paper_a=claim_a.paper_title,
        claim_a=claim_a.statement,
        year_b=claim_b.year,
        paper_b=claim_b.paper_title,
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
            return data
    except Exception as exc:
        print(f"Claim relation classification failed: {exc}")

    return None


def build_paper_chain_relations(
    claims_by_paper: dict[str, list[Claim]],
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> list[dict]:
    """Build claim relations between chronologically consecutive papers.

    For each pair of consecutive papers (sorted by year of earliest claim),
    compares the primary claim of the earlier paper with the primary claim
    of the later paper.

    Args:
        claims_by_paper: {paper_id: [Claim, ...]} — claims grouped by paper

    Returns:
        List of relation dicts with keys:
            source_paper, target_paper, source_claim, target_claim,
            relation, explanation
    """
    # Sort papers by year of their first claim
    sorted_papers = sorted(claims_by_paper.items(), key=lambda x: x[1][0].year)

    relations = []
    for i in range(len(sorted_papers) - 1):
        pid_a, claims_a = sorted_papers[i]
        pid_b, claims_b = sorted_papers[i + 1]

        # Use the first claim from each paper as the primary claim
        claim_a = claims_a[0]
        claim_b = claims_b[0]

        # P0 fix: check same research lineage BEFORE classifying relation type.
        # Papers from different lineages (e.g., perception vs planning) should
        # NOT be narrated as causal chains.
        lineage = classify_same_lineage(claim_a, claim_b, client=client, model=model)
        if lineage and lineage.get("same_lineage") == "NO":
            relations.append({
                "source_paper": claim_a.paper_title,
                "target_paper": claim_b.paper_title,
                "source_claim": claim_a.statement,
                "target_claim": claim_b.statement,
                "source_year": claim_a.year,
                "target_year": claim_b.year,
                "relation": "parallel",
                "explanation": f"Different research lineages — {lineage.get('reason', 'not causally related')}",
            })
            continue

        result = classify_claim_relation(claim_a, claim_b, client=client, model=model)
        if result:
            relations.append({
                "source_paper": claim_a.paper_title,
                "target_paper": claim_b.paper_title,
                "source_claim": claim_a.statement,
                "target_claim": claim_b.statement,
                "source_year": claim_a.year,
                "target_year": claim_b.year,
                "relation": result["relation"],
                "explanation": result["explanation"],
            })
        else:
            # Fallback: mark as unclassified
            relations.append({
                "source_paper": claim_a.paper_title,
                "target_paper": claim_b.paper_title,
                "source_claim": claim_a.statement,
                "target_claim": claim_b.statement,
                "source_year": claim_a.year,
                "target_year": claim_b.year,
                "relation": "unknown",
                "explanation": "Classification failed",
            })

    return relations


def relations_to_text(relations: list[dict]) -> str:
    """Format claim relations as readable text for narrative prompt injection."""
    if not relations:
        return ""

    lines = ["CLAIM EVOLUTION CHAIN (how each paper's primary claim relates to the prior one):"]
    for r in relations:
        rel_upper = r["relation"].upper()
        label = rel_upper
        if r["relation"] == "parallel":
            label = "PARALLEL (different lineage — do NOT narrate as causal)"
        lines.append(
            f"  {r['source_paper']} → {r['target_paper']}: {label}"
        )
        lines.append(f"    Explanation: {r['explanation']}")
    return "\n".join(lines)


def classify_paradigm_relation(
    earlier_paradigm: str,
    later_paradigm: str,
    earlier_phase_name: str,
    later_phase_name: str,
    *,
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> Optional[dict]:
    """Classify how a later paradigm relates to an earlier one.

    This is different from claim-level relations — it compares the FUNDAMENTAL
    ASSUMPTIONS of two eras, not individual paper claims. A paradigm relation
    of "replace" means the later era rendered the earlier era's core assumption
    obsolete.
    """
    if client is None:
        from llm_analyzer import build_analyzer_client
        client = build_analyzer_client()
    if not client:
        return None

    model = _resolve_model(model)
    if not model:
        return None

    prompt = _PARADIGM_RELATION_PROMPT.format(
        earlier_paradigm=earlier_paradigm,
        later_paradigm=later_paradigm,
        earlier_phase=earlier_phase_name,
        later_phase=later_phase_name,
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
            return data
    except Exception as exc:
        print(f"Paradigm relation classification failed: {exc}")

    return None


_PARADIGM_RELATION_PROMPT = """\
Compare the CORE PARADIGM of an earlier research era with a later one.

EARLIER ERA ({earlier_phase}):
Paradigm: "{earlier_paradigm}"

LATER ERA ({later_phase}):
Paradigm: "{later_paradigm}"

How does the later paradigm relate to the earlier one?

- attack: The later paradigm directly contradicts the earlier one. It says "the earlier belief was wrong."
- replace: The later paradigm renders the earlier one obsolete. It says "the entire approach is unnecessary."
- improve: The later paradigm keeps the earlier one's core approach but addresses limitations.
- extend: The later paradigm applies the earlier one to a new domain.

Return JSON:
```json
{{
  "relation": "attack|replace|improve|extend",
  "explanation": "One sentence explaining the paradigm relationship."
}}
```"""
