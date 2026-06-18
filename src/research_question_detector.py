"""Research question detector — identifies the QUESTIONS that define a field.

V7 core module. Unified detection: given claims, outputs ResearchQuestions with
nested Tensions and Direction in a single LLM call. This prevents the fragmentation
of having 4 independent detectors that produce overlapping abstractions.

Architecture:
    claims → [one LLM call] → ResearchQuestion[]
                                ├── question, description, level, status
                                ├── positions[] (paper→stance→evidence)
                                ├── tensions[] (debates within this RQ, 1-3)
                                └── direction (where evidence points)

Tension and Direction are NOT parallel to ResearchQuestion — they are nested
WITHIN it. This enforces the hierarchy: RQ is the organizing entity, Tensions
are the debates that arose while answering it, Direction is where the evidence
currently points.
"""
from __future__ import annotations

import json
import re
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _resolve_model
from paper import Claim, ResearchQuestion, Tension, Direction


# ── System prompt ─────────────────────────────────────────────────────
_RQ_SYSTEM = """\
You are a professor designing a graduate lecture on the evolution of a research \
field. Your task is to produce a structured analysis of the field's core research \
questions, the debates (tensions) within each question, and where the evidence \
currently points (direction).

THE HIERARCHY YOU MUST PRESERVE:

    ResearchQuestion (top-level: the question the field debated)
      ├── Positions (different answers papers gave to this question)
      ├── Tensions (specific contradictions/debates within this question)
      └── Direction (where the evidence favors now, with support/oppose/confidence)

CRITICAL: Do NOT collapse this hierarchy. A ResearchQuestion is a QUESTION \
ending with "?". A Tension is a CONTRADICTION between competing answers. \
A Direction is a CONCLUSION about which answer the evidence favors.

ResearchQuestion — must be:
  - A specific, answerable question ending with "?"
  - Something the field genuinely debated (not obvious or trivial)
  - Scoped enough that multiple papers can take different positions
  - Good: "Do we need dense BEV representation for 3D perception?"
  - Bad:  "What is the future of autonomous driving?"

Tension (within an RQ) — a specific debate or contradiction:
  - Emerges from different papers taking OPPOSING positions on the RQ
  - Has introducers (papers that made this tension visible) and resolvers \
    (papers that advanced one side)
  - Good: "Early methods achieved SOTA with dense grids (60.9 NDS), but \
    sparse methods later showed matched accuracy at lower cost"
  - Bad: "Dense vs sparse" (just a label, not a tension description)

Direction (conclusion of an RQ) — where the evidence points:
  - States which answer the evidence favors
  - Names specific supporting and opposing papers
  - Gives a confidence level: high/medium/low
  - Summarizes key evidence in 1-2 sentences
  - Must be HEDGED: "within this set of papers, the evidence favors..."

LEVELS for ResearchQuestion:
- "field": Defines the entire research area. Every paper must have an opinion.
- "paradigm": A major methodological debate within the field.
- "engineering": Practical constraint or implementation trade-off.

Only "field" and "paradigm" level RQs become lecture sections. \
"engineering" questions are footnotes.

STATUS for ResearchQuestion:
- "direction_clear": The field has largely converged on an answer
- "direction_forming": A dominant answer is emerging but not settled
- "open": Actively debated, no clear winner

Return ONLY a JSON object. No other text."""

# ── Per-field prompt ──────────────────────────────────────────────────
_RQ_PROMPT = """\
You are designing a lecture titled "The Evolution of {field_name}".

Below are the key claims extracted from {n_papers} papers in this field, \
sorted chronologically with their claim levels (paradigm/methodological/engineering).

CLAIMS:
{claims_text}

---

Think like a professor: you have 90 minutes to teach this field's evolution. \
Identify 3-4 CORE RESEARCH QUESTIONS that structure the field's intellectual \
history. Then, for each question, identify the tensions (debates) within it \
and where the evidence currently points.

INSTRUCTIONS — For each ResearchQuestion:

1. THE QUESTION (required):
   - A clear, specific question ending with "?"
   - A short name (3-5 words)
   - 1-2 sentences of context: why this question mattered
   - Level: "field" or "paradigm" (avoid "engineering" unless it shaped the field)
   - Status: "direction_clear" | "direction_forming" | "open"

2. POSITIONS (required):
   - Map each relevant paper to its position on this question
   - Include the specific evidence it provides
   - A paper may appear in multiple RQs, but prefer assigning it to the ONE \
     question it most directly addresses

3. TENSIONS (1-3 per RQ, required):
   - Each tension is a SPECIFIC debate within this RQ
   - Named as a contradiction, not a question (e.g. "Depth labels improve \
     accuracy but constrain backbone choice")
   - Has introducers (papers that first exposed the tension) and resolvers \
     (papers that advanced one side)
   - Has a status: "direction_clear" | "direction_forming" | "open"
   - Has a dimension: "representation" | "geometry" | "system" | "evaluation"
   - Has a domain_scope: what setting/benchmark this applies to
   - Tensions must involve at least 2 different papers

4. DIRECTION (required):
   - A statement of where the evidence currently points
   - Support papers: which papers' evidence backs this direction
   - Opposing papers: which papers complicate or challenge this direction
   - Confidence: "high" | "medium" | "low"
   - Evidence summary: 1-2 sentences of the key supporting evidence
   - Must use HEDGED language: "within this set of papers...", "the evidence \
     analyzed here suggests...", not "the field has settled on..."

CRITICAL RULES:
- 3-4 questions total. Merge overlapping questions; 3 good questions >> 5 redundant ones.
- Each paper's claims should PRIMARILY appear in ONE question. Avoid the same \
  paper dominating multiple questions.
- Tensions are WITHIN an RQ — they are the specific debates that papers fought over \
  while trying to answer this question.
- Direction is the CONCLUSION for this RQ — where the evidence points right now.
- Paradigm-level claims (>paradigm<) often define the questions themselves.
- Methodological claims (>methodological<) are evidence for positions.
- Engineering claims (>engineering<) may not appear in any RQ — that's fine.

Return JSON:
```json
{{
  "research_questions": [
    {{
      "question": "Is explicit depth supervision necessary for BEV perception?",
      "short_name": "Depth Necessity",
      "description": "Early BEV methods estimated depth implicitly from multi-view geometry, but the results were noisy. The field debated whether explicit LiDAR-supervised depth was essential for high-quality BEV detection.",
      "level": "paradigm",
      "status": "direction_forming",
      "introduced_by": ["BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection"],
      "positions": [
        {{
          "paper": "BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection",
          "position": "Yes — explicit depth supervision is essential for high-quality BEV detection",
          "evidence": "BEVDepth achieves 60.9% NDS, first camera-only method to surpass 60% NDS on nuScenes"
        }},
        {{
          "paper": "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers",
          "position": "No — spatiotemporal attention can learn geometry implicitly without any depth labels",
          "evidence": "BEVFormer achieves 56.9% NDS without depth supervision, matching LiDAR-based methods"
        }}
      ],
      "tensions": [
        {{
          "tension": "Explicit depth improves accuracy but constrains architecture",
          "description": "BEVDepth showed explicit depth supervision dramatically improved accuracy (60.9 NDS), but this required depth-pre-trained backbones, creating a dependency that limited architectural flexibility.",
          "introduced_by": ["BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection"],
          "resolved_by": ["BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers"],
          "status": "direction_forming",
          "dimension": "geometry",
          "domain_scope": "in multi-camera 3D object detection on nuScenes"
        }}
      ],
      "direction": {{
        "statement": "Within this trajectory, explicit depth supervision is not essential for state-of-the-art BEV perception; attention-based implicit geometry learning can match or exceed depth-supervised methods",
        "support_papers": ["BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers", "SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos"],
        "opposing_papers": ["BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection"],
        "confidence": "medium",
        "evidence_summary": "BEVFormer achieved 56.9 NDS without depth supervision, and SparseBEV later reached 67.5 NDS without any depth labels, surpassing BEVDepth's 60.9 NDS."
      }}
    }}
  ]
}}
```"""


# ── Public API ────────────────────────────────────────────────────────

def detect_research_questions(
    claims: list[Claim],
    *,
    field_name: str = "",
    client: Optional[OpenAI] = None,
    model: Optional[str] = None,
) -> list[ResearchQuestion]:
    """Detect 3-5 core research questions from claims.

    Returns list of ResearchQuestion objects. Empty list on failure.
    """
    if client is None:
        from llm_analyzer import build_analyzer_client
        client = build_analyzer_client()
    if not client:
        return []

    model = _resolve_model(model)
    if not model:
        return []

    if not claims:
        return []

    # Build claims text: group by paper, show chronologically
    claims_by_paper: dict[str, list[Claim]] = {}
    for c in claims:
        claims_by_paper.setdefault(c.paper_title, []).append(c)

    claims_parts = []
    for paper_title, paper_claims in sorted(claims_by_paper.items(),
                                             key=lambda x: (x[1][0].year, getattr(x[1][0], 'month', 0))):
        c0 = paper_claims[0]
        m = getattr(c0, 'month', 0)
        date_str = f"{c0.year}-{m:02d}" if m > 0 else str(c0.year)
        claims_parts.append(f"\n[{date_str}] {paper_title}")
        for j, c in enumerate(paper_claims, 1):
            level_tag = f"<{c.claim_level}>" if c.claim_level else ""
            claims_parts.append(f"  Claim {j} {level_tag}: {c.statement}")
            if c.evidence:
                claims_parts.append(f"    Evidence: {c.evidence}")

    claims_text = "\n".join(claims_parts)

    prompt = _RQ_PROMPT.format(
        field_name=field_name or "this field",
        n_papers=len(claims_by_paper),
        claims_text=claims_text,
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _RQ_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=4096,
            timeout=config.LLM_ANALYZER_TIMEOUT_SEC,
        )
        raw = response.choices[0].message.content or ""
        return _parse_response(raw)
    except Exception as exc:
        print(f"Research question detection failed: {exc}")
        return []


def _parse_response(raw: str) -> list[ResearchQuestion]:
    """Parse LLM response into ResearchQuestion list with nested Tensions and Direction."""
    if not raw or not raw.strip():
        return []

    # Strip markdown fences
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if not m:
            print("RQ detection: no JSON found in response")
            return []
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            print("RQ detection: invalid JSON")
            return []

    items = data.get("research_questions", [])
    if isinstance(data, list):
        items = data

    results = []
    for item in items:
        if not isinstance(item, dict):
            continue
        question_text = item.get("question", "").strip()
        if not question_text:
            continue

        # Parse nested tensions
        tensions = []
        for t in item.get("tensions", []):
            if isinstance(t, dict):
                tensions.append(Tension(
                    tension=t.get("tension", ""),
                    description=t.get("description", ""),
                    introduced_by=t.get("introduced_by", []),
                    resolved_by=t.get("resolved_by", []),
                    status=t.get("status", "open"),
                    dimension=t.get("dimension", "system"),
                    domain_scope=t.get("domain_scope", ""),
                ))

        # Parse nested direction
        direction = None
        d = item.get("direction")
        if isinstance(d, dict) and d.get("statement"):
            direction = Direction(
                statement=d.get("statement", ""),
                support_papers=d.get("support_papers", []),
                opposing_papers=d.get("opposing_papers", []),
                confidence=d.get("confidence", "medium"),
                evidence_summary=d.get("evidence_summary", ""),
            )

        results.append(ResearchQuestion(
            question=question_text,
            short_name=item.get("short_name", ""),
            description=item.get("description", ""),
            level=item.get("level", "paradigm"),
            status=item.get("status", "open"),
            positions=item.get("positions", []),
            introduced_by=item.get("introduced_by", []),
            tensions=tensions,
            direction=direction,
        ))

    return results


def rqs_to_text(rqs: list[ResearchQuestion]) -> str:
    """Format research questions for narrative prompt injection."""
    if not rqs:
        return ""

    lines = ["RESEARCH QUESTIONS (the skeleton of this field's intellectual history):"]
    for i, rq in enumerate(rqs, 1):
        lines.append(f"\n  Q{i} [{rq.level.upper()}] {rq.question}")
        lines.append(f"     Status: {rq.status}")
        lines.append(f"     Context: {rq.description}")
        if rq.positions:
            lines.append("     Positions:")
            for pos in rq.positions:
                paper = pos.get("paper", "?")
                position = pos.get("position", "?")
                lines.append(f"       - {paper}: {position}")
        if rq.tensions:
            lines.append("     Tensions within this question:")
            for t in rq.tensions:
                lines.append(f"       - {t.tension}: {t.description[:120]}")
        if rq.direction:
            d = rq.direction
            lines.append(f"     Direction: {d.statement}")
            lines.append(f"       Confidence: {d.confidence}")
            lines.append(f"       Supporting: {', '.join(d.support_papers[:3])}")
            if d.opposing_papers:
                lines.append(f"       Opposing: {', '.join(d.opposing_papers[:3])}")
    return "\n".join(lines)


def filter_lecture_rqs(rqs: list[ResearchQuestion]) -> list[ResearchQuestion]:
    """Filter to only field/paradigm level questions for lecture sections."""
    return [rq for rq in rqs if rq.level in ("field", "paradigm")]
