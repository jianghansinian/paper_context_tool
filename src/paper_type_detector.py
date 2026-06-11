"""Paper type detector for V3 schema-driven pipeline.

Determines which ``PaperTypeProfile`` best describes a given paper (e.g.,
experimental, theoretical, survey, ...) using a lightweight LLM call on
the paper's title, abstract, and section headings.

Falls back to the domain's ``default_paper_type`` when confidence is low
or LLM is unavailable.
"""

from __future__ import annotations

import json
import re
import textwrap
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _extract_json_object
from domains.base import DomainProfile
from paper import Paper

# Confidence thresholds
_HIGH_CONFIDENCE = 0.7
_LOW_CONFIDENCE = 0.5
_METADATA_ONLY_CONFIDENCE = 0.4


def _build_detection_input(paper: Paper) -> str:
    """Assemble title + abstract + intro + headings + conclusion for type detection."""
    text = paper.full_text or ""
    abstract = paper.abstract or ""

    # Introduction: first ~2000 chars after abstract
    intro = ""
    if abstract and abstract in text:
        start = text.find(abstract) + len(abstract)
        intro = text[start:start + 2000].strip()
    elif text:
        intro = text[:2000]

    # Section headings
    headings = re.findall(
        r'(?m)^\s*(?:\d+[\.\)]\s*)?([A-Z][A-Za-z\s]{2,50})$', text
    )
    headings_str = ", ".join(headings[:20])

    # Conclusion: section after "Conclusion" or "Discussion"
    conclusion = ""
    for tag in ["Conclusion", "Concluding Remarks", "Discussion"]:
        pattern = rf'(?im)^\s*(?:\d+[\.\)]\s*)?{tag}\s*$'
        m = re.search(pattern, text)
        if m:
            conclusion = text[m.end():m.end() + 800].strip()
            break

    parts = [f"TITLE: {paper.title}", f"ABSTRACT: {abstract}"]
    if intro:
        parts.append(f"INTRODUCTION: {intro}")
    if headings_str:
        parts.append(f"SECTIONS: {headings_str}")
    if conclusion:
        parts.append(f"CONCLUSION: {conclusion}")

    return "\n\n".join(parts)


def _build_detection_prompt(paper: Paper, domain: DomainProfile) -> str:
    """Build the LLM prompt for paper type detection."""
    content = _build_detection_input(paper)

    type_descs = []
    for pt in domain.paper_types:
        type_descs.append(f"- {pt.type_name}: {pt.description}")

    return textwrap.dedent(f"""\
        You are classifying an academic paper from the {domain.domain_name} domain.

        Candidate paper types:
        {chr(10).join(type_descs)}

        Paper content:
        {content}

        Determine which paper type best describes this paper. Return a JSON object:
        {{"paper_type": "type_name", "confidence": 0.0-1.0, "reasoning": "one sentence"}}

        Return ONLY the JSON object.
    """)


def detect_paper_type(
    paper: Paper,
    domain: DomainProfile,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
) -> str:
    """Detect the paper type and return the PaperTypeProfile type_name.

    Falls back to ``domain.default_paper_type`` when detection fails or
    confidence is too low.
    """
    default = domain.default_paper_type

    if client is None:
        return default

    from llm_analyzer import _resolve_model
    model_name = _resolve_model(model)
    if not model_name:
        return default

    prompt = _build_detection_prompt(paper, domain)

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=200,
        )
        raw = resp.choices[0].message.content or ""
        result = _extract_json_object(raw)
    except Exception:
        return default

    if result is None:
        return default

    paper_type = result.get("paper_type", default)
    confidence = result.get("confidence", 0.0)

    has_full_text = bool(paper.full_text)
    threshold = _LOW_CONFIDENCE if has_full_text else _METADATA_ONLY_CONFIDENCE

    if confidence < threshold:
        return default

    # Verify the returned type exists in this domain
    if domain.get_paper_type(paper_type) is None:
        return default

    return paper_type
