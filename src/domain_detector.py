"""Domain detector for V3 schema-driven pipeline.

Determines which academic domain a paper belongs to (ai_ml, biology,
materials_science, ...) using a lightweight LLM call on title + abstract.

When ``DOMAIN`` env var is set, detection is skipped and the explicit value
is used directly.
"""

from __future__ import annotations

import textwrap
from typing import Optional

from openai import OpenAI

from domains import get_domain, list_domains, DomainProfile
from llm_analyzer import _extract_json_object, _resolve_model
from paper import Paper


def _build_domain_detection_prompt(paper: Paper) -> str:
    """Build a prompt to detect which academic domain a paper belongs to."""
    domains = list_domains()
    domain_list = "\n".join(
        f"- {d}: {get_domain(d).domain_description}" for d in domains
    )

    abstract = paper.abstract or ""

    return textwrap.dedent(f"""\
        Based on the paper's title and abstract, determine which academic domain
        it belongs to.

        Candidate domains:
        {domain_list}

        TITLE: {paper.title}
        ABSTRACT: {abstract[:1500]}

        Return a JSON object:
        {{"domain": "domain_name", "confidence": 0.0-1.0, "reasoning": "one sentence"}}

        Return ONLY the JSON object.
    """)


def detect_domain(
    paper: Paper,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
) -> DomainProfile:
    """Detect the domain and return the corresponding DomainProfile.

    Uses explicit DOMAIN env var if set; otherwise queries the LLM with
    title + abstract.  Falls back to ai_ml on any failure.
    """
    import config
    default = get_domain("ai_ml")

    # Explicit override
    if config.DOMAIN:
        explicit = config.DOMAIN.strip()
        try:
            return get_domain(explicit)
        except KeyError:
            print(f"Unknown DOMAIN '{explicit}', falling back to ai_ml.")
            return default

    if client is None:
        return default

    model_name = _resolve_model(model)
    if not model_name:
        return default

    prompt = _build_domain_detection_prompt(paper)

    try:
        resp = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=150,
        )
        raw = resp.choices[0].message.content or ""
        result = _extract_json_object(raw)
    except Exception:
        return default

    if result is None:
        return default

    domain_name = result.get("domain", "ai_ml")
    confidence = result.get("confidence", 0.0)

    if confidence < 0.5:
        return default

    try:
        return get_domain(domain_name)
    except KeyError:
        return default
