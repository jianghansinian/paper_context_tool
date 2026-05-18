"""Technical route analysis and comparison for V3 pipeline.

Groups papers by technical approach (not topic), identifies mainstream routes,
and compares the seed paper against mainstream approaches.
"""
from __future__ import annotations

import json
import textwrap
from collections import defaultdict
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _extract_json_object, _resolve_model
from paper import Paper, StructuredUnderstanding


def analyze_routes(
    papers: list[Paper],
    seed_paper: Paper,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
) -> Optional[dict]:
    """Group papers by technical approach and identify mainstream routes.

    Returns: {
        "branches": [
            {
                "name": "Branch name",
                "description": "Technical approach description",
                "paper_ids": [...],
                "is_mainstream": True,
                "common_tags": ["tag1", ...],
            }, ...
        ],
        "overview": "Field overview text",
    }
    Returns None on failure (caller should handle gracefully).
    """
    if client is None or len(papers) < 2:
        return _fallback_routes(papers)

    model = _resolve_model(model)
    if not model:
        return _fallback_routes(papers)

    # Build compact paper summaries
    paper_summaries = []
    for i, p in enumerate(papers):
        s = p.structured
        summary = f"[{i}] {p.title} ({p.year}, {p.citation_count} citations)"
        if s and s.architecture_overview:
            arch = textwrap.shorten(s.architecture_overview, width=200, placeholder="...")
            summary += f"\n    Architecture: {arch}"
        if s and s.key_insight:
            insight = textwrap.shorten(s.key_insight, width=150, placeholder="...")
            summary += f"\n    Key insight: {insight}"
        paper_summaries.append(summary)

    papers_text = "\n\n".join(paper_summaries)

    user_lens = ""
    if seed_paper.user_description:
        user_lens = (
            f"USER FOCUS: {seed_paper.user_description}\n"
            "When grouping, pay attention to how this focus relates to each branch.\n\n"
        )

    prompt = textwrap.dedent(f"""\
        Analyze the following papers and group them by TECHNICAL APPROACH.

        Each paper has been structurally analyzed. Group papers that share the SAME
        technical methodology (e.g., "Transformer-based detection", "LSS-based BEV",
        "Diffusion-based generation"). Do NOT group by topic/application.

        {user_lens}
        Papers:
        {papers_text}

        Return a JSON object:
        {{
          "overview": "1-2 paragraph overview of the field's technical landscape",
          "branches": [
            {{
              "name": "concise technical branch name (3-8 words)",
              "description": "what defines this approach technically",
              "paper_indices": [0, 3, 5],
              "is_mainstream": true,
              "common_technical_tags": ["shared technique 1", "shared technique 2"]
            }}
          ]
        }}

        Rules:
        - Every paper must be assigned to exactly one branch.
        - A branch is mainstream if it has many papers or high citation counts.
        - Name branches by technical approach, NOT by topic.
        - Return ONLY the JSON object.
    """)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=3000,
        )
        raw = resp.choices[0].message.content or ""
        result = _extract_json_object(raw)
        if result:
            # Map paper indices to IDs
            for branch in result.get("branches", []):
                indices = branch.get("paper_indices", [])
                branch["paper_ids"] = [
                    papers[i].id for i in indices if 0 <= i < len(papers)
                ]
                del branch["paper_indices"]
            return result
    except Exception as exc:
        print(f"Route analysis failed: {exc}")

    return _fallback_routes(papers)


def _fallback_routes(papers: list[Paper]) -> dict:
    """Simple fallback: group by year range."""
    groups = defaultdict(list)
    for p in papers:
        decade = (p.year // 5) * 5 if p.year else 0
        groups[decade].append(p)

    branches = []
    for decade, group_papers in sorted(groups.items()):
        branches.append({
            "name": f"Papers circa {decade}-{decade + 4}" if decade else "Undated",
            "description": f"{len(group_papers)} papers from this period",
            "paper_ids": [p.id for p in group_papers],
            "is_mainstream": len(group_papers) >= 3,
            "common_technical_tags": [],
        })

    return {
        "branches": branches,
        "overview": f"Auto-grouped by year (LLM unavailable). {len(papers)} papers in {len(branches)} groups.",
    }


def compare_with_mainstream(
    seed_paper: Paper,
    routes: dict,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
) -> Optional[dict]:
    """Compare seed paper against identified mainstream routes.

    Returns: {
        "comparison_matrix": [...],
        "narrative": "Comparison narrative text",
        "unique_positioning": "How the seed paper is uniquely positioned",
    }
    """
    branches = routes.get("branches", [])
    if not branches or client is None:
        return _fallback_comparison(seed_paper, routes)

    model = _resolve_model(model)
    if not model:
        return _fallback_comparison(seed_paper, routes)

    mainstream = [b for b in branches if b.get("is_mainstream")]
    if not mainstream:
        mainstream = branches[:3]  # take top 3 if none marked mainstream

    seed_s = seed_paper.structured
    seed_desc = f"Title: {seed_paper.title}\n"
    if seed_s:
        seed_desc += f"Architecture: {seed_s.architecture_overview}\n"
        seed_desc += f"Key insight: {seed_s.key_insight}\n"
        if seed_s.components:
            comps = ", ".join(c.name for c in seed_s.components[:5])
            seed_desc += f"Components: {comps}\n"

    branch_descs = []
    for b in mainstream:
        branch_descs.append(f"- {b['name']}: {b.get('description', '')}")

    user_lens = ""
    if seed_paper.user_description:
        user_lens = (
            f"USER FOCUS: {seed_paper.user_description}\n"
            "Emphasize this aspect in the comparison.\n\n"
        )

    prompt = textwrap.dedent(f"""\
        Compare the seed paper against mainstream technical approaches.

        SEED PAPER:
        {seed_desc}

        {user_lens}
        MAINSTREAM APPROACHES:
        {chr(10).join(branch_descs)}

        Return a JSON object:
        {{
          "comparison_matrix": [
            {{
              "dimension": "e.g., Core Architecture",
              "seed_paper": "how the seed paper handles this",
              "mainstream_approach": "how mainstream approaches handle this",
              "advantage": "seed paper's advantage or difference"
            }}
          ],
          "narrative": "1-2 paragraph narrative comparing the seed paper to mainstream work",
          "unique_positioning": "How the seed paper is uniquely positioned in the field"
        }}

        Return ONLY the JSON object.
    """)

    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2000,
        )
        raw = resp.choices[0].message.content or ""
        return _extract_json_object(raw)
    except Exception as exc:
        print(f"Comparative analysis failed: {exc}")

    return _fallback_comparison(seed_paper, routes)


def _fallback_comparison(seed_paper: Paper, routes: dict) -> dict:
    return {
        "comparison_matrix": [],
        "narrative": "",
        "unique_positioning": "",
    }
