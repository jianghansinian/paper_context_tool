"""Claim extraction — the core V4 abstraction.

``extract_claims(paper, llm_client) -> list[Claim]`` extracts falsifiable
research claims from a paper, carefully distinguishing Claim (a judgment
about what works and why) from Solution (a description of what was done).

This distinction is the foundation of the V4 Research Narrative Engine.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _extract_json_object, _resolve_model
from paper import Paper, Claim

# ── System prompt: teach LLM to distinguish Claim from Solution ──────
_CLAIM_SYSTEM_PROMPT = """\
You are an expert research analyst specializing in autonomous driving and \
computer vision. Your task is to extract falsifiable research CLAIMS from \
academic papers.

A CLAIM is NOT a description of what the paper did. A CLAIM is a judgment \
the paper makes about what is true, what works, what is better, and why.

CRITICAL DISTINCTION — Claim vs Solution:

    Solution (DO NOT extract):
    - "We use V-JEPA for video pre-training"
    - "The model has a three-module framework"
    - "We train on nuScenes with Adam optimizer"
    - "We propose Lift-Splat-Shoot for view transformation"
    These describe WHAT was done. They are not claims.

    Claim (EXTRACT these):
    - "Predictive video pre-training can outperform perception-heavy \
pipelines by 3 PDMS even with a simple decoder"
    - "Dense BEV representation is unnecessary; sparse queries preserve \
accuracy at significantly lower computational cost"
    - "Temporal self-attention can implicitly learn depth from multi-frame \
data, removing the need for explicit depth supervision"
    - "Depth probability distributions can lift 2D features to 3D BEV space \
without requiring explicit depth estimation"
    These assert WHAT IS TRUE about the approach. They are falsifiable.

A good Claim:
1. Is falsifiable — an experiment could in principle prove it wrong
2. Contains a comparative or absolute judgment (better/faster/sufficient/\
unnecessary/capable)
3. Explains WHY something works (mechanism), not just THAT it works
4. Is specific enough to be meaningful

A bad Claim (do NOT extract):
1. Describes the method ("we propose X")
2. States obvious facts ("dataset Y has Z samples")
3. Is too vague to be falsifiable ("our method is effective")
4. Reports a result without the claim BEHIND the result

For each paper, extract 2-4 claims. Each claim must pass the falsifiability \
test: "Could someone design an experiment that would prove this wrong?"

Return ONLY a JSON array. No other text."""


# ── Few-shot examples ─────────────────────────────────────────────────
_FEW_SHOT_POSITIVE = """\
--- EXAMPLE: DETR (Carion et al., ECCV 2020) ---

Paper abstract: We present a new method that views object detection as a \
direct set prediction problem. Our approach streamlines the detection pipeline, \
effectively removing the need for many hand-designed components like a \
non-maximum suppression procedure or anchor generation that explicitly encode \
our prior knowledge about the task. The main ingredients of the new framework, \
called DEtection TRansformer or DETR, are a set-based global loss that forces \
unique predictions via bipartite matching, and a transformer encoder-decoder \
architecture. Given a fixed small set of learned object queries, DETR reasons \
about the relations of the objects and the global image context to directly \
output the final set of predictions in parallel. The new model is conceptually \
simple and does not require a specialized library, unlike many other modern \
detectors. DETR demonstrates accuracy and run-time performance on par with the \
well-established and highly-optimized Faster R-CNN baseline on the challenging \
COCO object detection dataset. Moreover, DETR can be easily generalized to \
produce panoptic segmentation in a unified manner. We show that it \
significantly outperforms competitive baselines.

CORRECT claims extracted:
```json
[
  {
    "statement": "Object detection can be formulated as a direct set \
prediction problem, eliminating the need for hand-designed components like \
NMS and anchor generation",
    "evidence": "DETR matches Faster R-CNN accuracy on COCO without NMS or \
anchors; panoptic segmentation extension shows generality",
    "problem_addressed": "Object detection pipelines rely on hand-crafted \
components (NMS, anchors) that encode task-specific priors",
    "claim_type": "introduces"
  },
  {
    "statement": "A transformer encoder-decoder with learned object queries \
can reason about object relations and global context to directly output \
predictions in parallel",
    "evidence": "DETR achieves on-par accuracy with highly-optimized Faster \
R-CNN; attention maps show global reasoning behavior",
    "problem_addressed": "Traditional detectors process objects independently \
without modeling global relationships",
    "claim_type": "introduces"
  }
]
```"""


_FEW_SHOT_NEGATIVE = """\
--- WHAT NOT TO EXTRACT (from the same DETR abstract) ---

The following are SOLUTIONS, not CLAIMS. Do NOT include anything like these:

```json
[
  {
    "statement": "We propose DETR, a new method for object detection",
    "evidence": "",
    "problem_addressed": "",
    "claim_type": "introduces"
  },
  {
    "statement": "DETR uses a bipartite matching loss and transformer architecture",
    "evidence": "",
    "problem_addressed": "",
    "claim_type": ""
  }
]
```

The first is just announcing a method. The second is a method description.
Neither makes a falsifiable judgment about what is true or better.
"""


# ── Public API ────────────────────────────────────────────────────────


def extract_claims(
    paper: Paper,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
) -> list[Claim]:
    """Extract falsifiable research claims from a paper.

    Args:
        paper: Paper object with at minimum title and abstract.
        client: OpenAI-compatible LLM client.
        model: Model override.

    Returns:
        List of Claim objects. Empty list on failure.
    """
    if client is None:
        client = _build_client()

    model = _resolve_model(model)
    if not model:
        print(f"Claim extraction: no model available for {paper.title[:60]}")
        return []

    text = paper.abstract or ""
    if paper.full_text:
        # Include introduction/conclusion if available (first + last 4000 chars
        # of full text)
        intro = paper.full_text[:4000] if paper.full_text else ""
        conclusion = _extract_conclusion(paper.full_text) if paper.full_text else ""
        if conclusion:
            text = f"{text}\n\nINTRODUCTION:\n{intro}\n\nCONCLUSION:\n{conclusion}"
        elif intro:
            text = f"{text}\n\nPAPER TEXT:\n{intro}"

    if not text.strip():
        print(f"Claim extraction: no text available for {paper.title[:60]}")
        return []

    prompt = _build_prompt(paper, text)
    return _call_llm(client, model, prompt, paper)


def _build_client() -> Optional[OpenAI]:
    """Build LLM client from config."""
    try:
        from llm_analyzer import build_analyzer_client
        return build_analyzer_client()
    except Exception:
        return None


def _extract_conclusion(full_text: str) -> str:
    """Extract the conclusion/discussion section from full text."""
    if not full_text:
        return ""
    patterns = [
        r"(?i)\b(conclusion|concluding\s+remarks|discussion)\b",
        r"(?i)\b(limitations?\s+and\s+(future|conclusion))\b",
    ]
    for pattern in patterns:
        m = re.search(pattern, full_text)
        if m:
            start = m.start()
            # Take ~3000 chars from conclusion start
            return full_text[start:start + 3000]
    # Fallback: last 2000 chars
    return full_text[-2000:]


def _build_prompt(paper: Paper, text: str) -> str:
    """Build the claim extraction prompt with few-shot examples."""

    # Truncate text to ~8K chars to leave room for few-shot examples
    max_text = 8000
    if len(text) > max_text:
        text = text[:max_text]

    return f"""\
{_FEW_SHOT_POSITIVE}

{_FEW_SHOT_NEGATIVE}

--- NOW EXTRACT CLAIMS FROM THIS PAPER ---

TITLE: {paper.title}
YEAR: {paper.year}
AUTHORS: {', '.join(paper.authors[:5])}{'...' if len(paper.authors) > 5 else ''}

{text}

---

Extract 2-4 falsifiable claims from this paper. Each claim must:
1. Be a judgment about what is true/better, not a method description
2. Be falsifiable
3. Include specific evidence

Return JSON array only:
```json
[
  {{
    "statement": "...",
    "evidence": "...",
    "problem_addressed": "...",
    "claim_type": "introduces|improves|replaces|extends"
  }}
]
```"""


def _call_llm(
    client: OpenAI,
    model: str,
    prompt: str,
    paper: Paper,
) -> list[Claim]:
    """Call LLM and parse response into Claim list."""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _CLAIM_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        return _parse_response(raw, paper)
    except Exception as exc:
        print(f"Claim extraction failed for {paper.title[:60]}: {exc}")
        return []


def _parse_response(raw: str, paper: Paper) -> list[Claim]:
    """Parse LLM response into Claim objects."""
    if not raw or not raw.strip():
        return []

    # Try standard JSON extraction first
    parsed = _extract_json_object(raw)
    if parsed:
        items = parsed if isinstance(parsed, list) else parsed.get("claims", [])
    else:
        # Fallback: try to find JSON array directly
        m = re.search(r"\[\s*\{.*?\}\s*\]", raw, re.DOTALL)
        if not m:
            print(f"Claim extraction: no JSON found in response for {paper.title[:60]}")
            return []
        try:
            items = json.loads(m.group(0))
        except json.JSONDecodeError:
            print(f"Claim extraction: invalid JSON for {paper.title[:60]}")
            return []

    claims = []
    for item in items:
        if not isinstance(item, dict):
            continue
        statement = item.get("statement", "").strip()
        if not statement:
            continue
        # Filter out obvious Solution descriptions
        if _is_solution(statement):
            continue
        claims.append(Claim(
            paper_id=paper.id,
            paper_title=paper.title,
            year=paper.year,
            statement=statement,
            evidence=item.get("evidence", "").strip(),
            problem_addressed=item.get("problem_addressed", "").strip(),
            claim_type=item.get("claim_type", "introduces"),
        ))

    return claims


# ── Solution detection heuristics ────────────────────────────────────

_SOLUTION_PATTERNS = [
    r"^(we|this\s+paper|our\s+work)\s+(propose|present|introduce|describe)",
    r"^(the|our)\s+(model|method|framework|system|architecture|approach)\s+(uses?|employs?|consists?|is\s+based)",
    r"^(we|this\s+paper)\s+(use|apply|employ|train|evaluate|test)",
    r"^the\s+(proposed|presented)\s+(model|method|framework)",
]


def _is_solution(statement: str) -> bool:
    """Check if a statement looks like a Solution rather than a Claim."""
    lower = statement.lower().strip()
    for pattern in _SOLUTION_PATTERNS:
        if re.match(pattern, lower):
            return True
    # Too short to be a meaningful claim
    if len(lower.split()) < 8:
        return True
    return False
