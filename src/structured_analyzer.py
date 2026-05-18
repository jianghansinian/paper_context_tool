"""Structured understanding engine — the core V3 abstraction.

``analyze_paper_structure(paper, llm_client) -> StructuredUnderstanding`` applies
the same structured analysis to any paper, extracting architecture, formulas,
training/inference pipelines, and results.

Falls back gracefully when full text is unavailable or LLM fails.
"""
from __future__ import annotations

import json
import textwrap
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _extract_json_object, build_analyzer_client, _resolve_model
from paper import Paper, StructuredUnderstanding

_MAX_TEXT_CHARS = 120000  # ~30K tokens for English, covers most papers


def _build_analysis_prompt(paper: Paper) -> str:
    """Build the LLM prompt for structured paper analysis."""
    text = paper.full_text or paper.abstract
    if not text:
        return ""

    if len(text) > _MAX_TEXT_CHARS:
        # Keep abstract + truncate body
        body_start = text.find(paper.abstract) if paper.abstract else 0
        body_start = body_start + len(paper.abstract) if body_start >= 0 else 0
        available = _MAX_TEXT_CHARS - len(paper.abstract) - 2000
        if available > 0:
            text = paper.abstract + "\n\n" + text[body_start:body_start + available]
        else:
            text = text[:_MAX_TEXT_CHARS]
        text += "\n\n[Text truncated due to length — focus on available sections.]"

    authors_str = ", ".join(paper.authors[:8])
    if len(paper.authors) > 8:
        authors_str += f" et al."

    user_lens = ""
    if paper.user_description:
        user_lens = textwrap.dedent(f"""\
            USER FOCUS (use this as an analytical lens):
            {paper.user_description}

            When analyzing, pay special attention to the aspects mentioned above.
            Highlight how they relate to the paper's design choices.
        """)

    prompt = textwrap.dedent(f"""\
        You are an expert AI researcher analyzing a paper in depth.

        TITLE: {paper.title}
        AUTHORS: {authors_str}
        YEAR: {paper.year}

        {user_lens}
        INSTRUCTIONS:
        1. First, understand the core problem and the paper's motivation.
        2. Identify the architectural design — especially the architecture diagram
           (usually Figure 1). Read its caption and surrounding text carefully.
           Describe the data flow and what each component does.
        3. Extract ALL key formulas with names and explanations. Do not skip any
           that are central to the method.
        4. Describe the training pipeline in detail: data used, loss functions,
           optimizer, training procedure, any tricks or staged training.
        5. Describe the inference pipeline: forward pass, post-processing steps.
        6. Extract main experimental results with dataset names, metrics, and values.
        7. Identify key contributions and acknowledged limitations.

        Return a JSON object with this EXACT structure (use null for missing, [] for empty):

        {{
          "problem": "What problem does this paper solve?",
          "motivation": "Why is this problem important? Limitations of existing work?",
          "key_insight": "The core idea or key insight of this paper",
          "architecture_overview": "Overall description of the method/architecture",
          "architecture_figure": "Detailed explanation of the architecture diagram (Figure 1), describing components and data flow. Use null if no such figure.",
          "components": [
            {{
              "name": "Component name",
              "purpose": "What this component does",
              "details": "Implementation details — dimensions, layer configs, etc. null if unknown",
              "referenced_figure": "e.g. 'Figure 2(a)' — null if none"
            }}
          ],
          "formulas": [
            {{
              "name": "Formula name (e.g. 'Cross-Attention', 'Focal Loss')",
              "latex": "LaTeX expression if available, null otherwise",
              "explanation": "What this formula computes",
              "significance": "Why this formula matters to the method"
            }}
          ],
          "training_data": "What datasets were used for training",
          "loss_functions": ["loss function 1", "loss function 2"],
          "optimizer": "Optimizer name and learning rate/schedule",
          "training_procedure": "Detailed training procedure: data augmentation, staged training, tricks",
          "inference_procedure": "How inference works: forward pass flow",
          "post_processing": "Any post-processing steps. null if none.",
          "main_results": [
            {{
              "dataset": "Dataset name",
              "metric": "Metric name",
              "value": "Achieved value",
              "comparison": "How this compares to baselines. null if not stated."
            }}
          ],
          "ablation_results": ["Key ablation finding 1", "Key ablation finding 2"],
          "qualitative_results": "Description of qualitative/visualization results. null if none.",
          "contributions": ["Contribution 1", "Contribution 2"],
          "limitations": ["Limitation 1", "Limitation 2"]
        }}

        CRITICAL:
        - For architecture_figure: carefully read the caption and text around the architecture
          diagram (usually Figure 1). Describe EACH component and how data flows through them.
        - For components: extract EVERY major module of the architecture with implementation details.
        - For formulas: include ALL key mathematical expressions with their names.
        - For training_procedure: be as detailed as possible.
        - Return ONLY the JSON object. No other text.

        PAPER TEXT:
        {text}
    """)
    return prompt


def _parse_analysis_response(raw: str) -> Optional[dict]:
    """Parse the LLM JSON response, healing common issues."""
    if not raw:
        return None
    parsed = _extract_json_object(raw)
    if parsed is None:
        return None
    # Ensure list fields are actually lists
    for key in ("components", "formulas", "main_results", "loss_functions",
                "ablation_results", "contributions", "limitations"):
        if not isinstance(parsed.get(key), list):
            parsed[key] = []
    return parsed


def analyze_paper_structure(
    paper: Paper,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
) -> Optional[StructuredUnderstanding]:
    """Analyze a paper's structure using LLM — the core V3 abstraction.

    Applies to any paper (seed or key paper). Returns None on failure so the
    caller can fall back to heuristic analysis.
    """
    if client is None:
        print("No LLM client available for structured analysis.")
        if paper.abstract:
            return StructuredUnderstanding(
                problem=paper.abstract,
                key_insight=paper.abstract,
                architecture_overview=paper.abstract,
            )
        return None

    model = _resolve_model(model)
    if not model:
        return None

    prompt = _build_analysis_prompt(paper)
    if not prompt:
        print(f"No text available for {paper.title}")
        return None

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": "You are an expert AI researcher. You output JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=6000,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_analysis_response(raw)

        if parsed is None:
            print(f"Failed to parse structured analysis response for: {paper.title}")
            return None

        return StructuredUnderstanding.from_dict(parsed)

    except Exception as exc:
        print(f"Structured analysis failed for {paper.title}: {exc}")
        if paper.abstract:
            return StructuredUnderstanding(
                problem=paper.abstract,
                key_insight=paper.abstract,
            )
        return None
