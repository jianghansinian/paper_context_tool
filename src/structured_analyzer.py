"""Structured understanding engine — the core V3 abstraction.

``analyze_paper_structure(paper, llm_client) -> StructuredUnderstanding`` applies
the same structured analysis to any paper, extracting architecture, formulas,
training/inference pipelines, and results.

Falls back gracefully when full text is unavailable or LLM fails.

Prompt generation is driven by a ``PaperTypeProfile`` schema (see
``src/domains/``).  When no profile is given, the AI/ML experimental profile
is used as default.
"""
from __future__ import annotations

import json
import re
import textwrap
from typing import Optional

from openai import OpenAI

import config
from llm_analyzer import _extract_json_object, build_analyzer_client, _resolve_model
from paper import Paper, StructuredUnderstanding
from domains.base import PaperTypeProfile, FieldDef

_MAX_TEXT_CHARS = 120000  # ~30K tokens; DeepSeek-V3 has 128K context

# ── Smart truncation: section priority scoring ────────────────────────
# Higher score = keep first when budget is tight.
_SECTION_SCORE: dict[str, int] = {
    "abstract": 100,
    "introduction": 90,
    "method": 100, "approach": 100, "architecture": 100, "model": 100,
    "design": 100, "algorithm": 100, "framework": 100, "pipeline": 100,
    "training": 100, "inference": 100, "implementation": 95,
    "problem": 90, "formulation": 90, "motivation": 90, "overview": 90,
    "preliminar": 80, "background": 80,
    "experiment": 95, "result": 95, "evaluation": 95, "ablation": 95,
    "analysis": 85, "discussion": 80, "conclusion": 85,
    "dataset": 65, "data": 65, "setup": 65,
    "related": 40, "literature": 40,
    "appendix": 10, "acknowledgment": 5, "reference": 5,
    "bibliography": 5, "supplementary": 10, "citation": 5,
}

# Regexes for section boundary detection (ordered by specificity).
_SECTION_RE = re.compile(
    r'\n('
    r'\d+(?:\.\d+)*\.?\s+[A-Z][A-Za-z\s\-]{2,60}'  # "3.1. Method Overview"
    r'|'
    r'#{1,3}\s+[A-Z][A-Za-z\s\-]{2,60}'  # "## Method Overview"
    r'|'
    r'[A-Z][A-Z\s\-]{4,60}'  # "METHOD" or "METHOD OVERVIEW"
    r'|'
    r'[A-Z][A-Za-z\s\-]{3,60}\n[=\-]{3,}'  # underlined headers
    r')'
    r'(?:\n|:\s*\n)',
    re.MULTILINE,
)


def _score_section_header(header: str) -> int:
    """Return a priority score for a section header (0 = unrecognized)."""
    lower = header.lower().strip("#").strip()
    # Normalise numbering: "3.1. method" → "method"
    lower = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', lower)
    for keyword, score in _SECTION_SCORE.items():
        if keyword in lower:
            return score
    return 50  # unknown sections get middling priority


def _smart_truncate(text: str, abstract: str, max_chars: int) -> str:
    """Truncate *text* to fit *max_chars*, preserving high-value sections."""
    if len(text) <= max_chars:
        return text

    # 1. Split into sections at detected headers
    section_spans: list[tuple[int, int, str]] = []  # (start, end, header_text)
    for m in _SECTION_RE.finditer(text):
        header = m.group(1).strip()
        start = m.start()
        section_spans.append((start, -1, header))

    if not section_spans:
        # No sections detected — take prefix strategy (abstract + body start)
        return _prefix_truncate(text, abstract, max_chars)

    # Fill in end positions
    for i, (start, _, header) in enumerate(section_spans):
        if i + 1 < len(section_spans):
            end = section_spans[i + 1][0]
        else:
            end = len(text)
        section_spans[i] = (start, end, header)

    # 2. Score each section
    scored: list[tuple[int, int, str, int]] = []
    for start, end, header in section_spans:
        score = _score_section_header(header)
        scored.append((start, end, header, score))

    # 3. Build the output: pre-section text (abstract area) + ranked sections
    if scored:
        pre_body = text[:scored[0][0]]
    else:
        pre_body = text[:max_chars]

    # Prefer to preserve the beginning (contains abstract)
    budget = max_chars - len(pre_body)
    if budget <= 0:
        return pre_body[:max_chars]

    # Sort sections by score (descending), then by position (ascending)
    ranked = sorted(enumerate(scored), key=lambda x: (-x[1][3], x[0]))

    # Greedily include sections in original order up to budget
    selected = [False] * len(scored)
    remaining = budget

    for idx, (start, end, header, score) in ranked:
        sec_text = text[start:end]
        sec_len = len(sec_text)
        if remaining <= 0:
            break
        if sec_len <= remaining:
            selected[idx] = True
            remaining -= sec_len
        else:
            # Partial: take first remaining chars + note
            selected[idx] = True
            # We'll trim this section in the reconstruction step
            remaining = 0

    # 4. Reconstruct in original order
    parts = [pre_body]
    for i, (start, end, header, score) in enumerate(scored):
        if not selected[i]:
            continue
        sec_text = text[start:end]
        if len(sec_text) > budget:
            # This section exceeds remaining budget — take a prefix
            sec_text = sec_text[:budget] + "\n\n[... section truncated]"
            budget = 0
        else:
            budget -= len(sec_text)
        parts.append(sec_text)

    result = "".join(parts)
    if len(result) < len(text):
        result += "\n\n[Text selectively truncated — method/experiment sections prioritized over related work/appendix.]"
    return result


def _prefix_truncate(text: str, abstract: str, max_chars: int) -> str:
    """Fallback: keep abstract + body prefix when sections can't be detected."""
    if len(text) <= max_chars:
        return text
    body_start = text.find(abstract) if abstract else 0
    body_start = body_start + len(abstract) if body_start >= 0 else 0
    available = max_chars - len(abstract) - 2000
    if available > 0:
        result = abstract + "\n\n" + text[body_start:body_start + available]
    else:
        result = text[:max_chars]
    return result + "\n\n[Text truncated due to length — focus on available sections.]"

# ── JSON type descriptions per field kind ────────────────────────────
_KIND_JSON_TYPE: dict[str, str] = {
    "text": "string",
    "list[str]": '["item1", "item2", ...]',
    "component_table": (
        '[{"name": "Component name", "purpose": "What it does", '
        '"details": "Implementation details — null if unknown", '
        '"referenced_figure": "Figure X — null if none"}]'
    ),
    "formula_table": (
        '[{"name": "Formula name", "latex": "LaTeX or null", '
        '"explanation": "What it computes", '
        '"significance": "Why it matters"}]'
    ),
    "result_table": (
        '[{"dataset": "Dataset name", "metric": "Metric name", '
        '"value": "Achieved value", '
        '"comparison": "vs baseline — null if not stated"}]'
    ),
    "key_value_table": '[{"key": "Key", "value": "Value"}]',
    "structured_list": (
        '[{"title": "Item title", "description": "Description", '
        '"papers": ["paper ref 1"]}]'
    ),
}


def _describe_json_type(field: FieldDef) -> str:
    """Return a human-readable JSON type description for *field*.

    For table kinds, generates the example from the field's actual column
    definitions so the LLM sees the correct field names in the JSON schema.
    """
    kind = field.kind
    if kind in ("component_table", "result_table", "key_value_table"):
        cols = field.columns
        if cols:
            example = {c.name: f"<{c.label_en.lower()}>" for c in cols}
            return json.dumps([example], indent=2)
    return _KIND_JSON_TYPE.get(kind, "string")


def _build_analysis_prompt(
    paper: Paper,
    profile: Optional[PaperTypeProfile] = None,
    domain_name: str = "ai_ml",
) -> str:
    """Build the LLM prompt for structured paper analysis.

    When *profile* is given, generates instructions and JSON schema from it.
    Otherwise falls back to the AI/ML experimental profile.
    """
    if profile is None:
        from domains.ai_ml import EXPERIMENTAL_PROFILE
        profile = EXPERIMENTAL_PROFILE
    return _build_prompt_from_schema(paper, profile, domain_name)


def _build_prompt_from_schema(
    paper: Paper, profile: PaperTypeProfile, domain_name: str
) -> str:
    """Build the LLM analysis prompt entirely from a PaperTypeProfile schema."""
    text = paper.full_text or paper.abstract
    if not text:
        return ""

    text = _smart_truncate(text, paper.abstract or "", _MAX_TEXT_CHARS)

    authors_str = ", ".join(paper.authors[:8])
    if len(paper.authors) > 8:
        authors_str += " et al."

    user_lens = ""
    if paper.user_description:
        user_lens = textwrap.dedent(f"""\
            USER FOCUS (use this as an analytical lens):
            {paper.user_description}

            When analyzing, pay special attention to the aspects mentioned above.
            Highlight how they relate to the paper's design choices.
        """)

    # ── Build numbered instructions from field prompts ──
    # Emphasis markers for fields that benefit from aggressive extraction
    _EMPHASIS: dict[str, str] = {
        "formula_table": " Include ALL formulas central to the method — do not skip any.",
        "component_table": " Include EVERY component with full implementation details.",
        "result_table": " Include ALL rows with complete data. For ablation tables, "
                       "include ALL experiments — component ablations, hyper-parameter "
                       "sweeps, pre-training comparisons, etc.",
        "structured_list": " Provide a complete breakdown — include ALL stages/steps with details.",
        "text": " Structure your response clearly: use **bold** for key concept names, "
               "method names, and technical terms. Use proper paragraph breaks (\\n\\n) "
               "between distinct topics, modules, or sub-themes. "
               "CRITICAL — numbered items (1)(2)(3) or 1. 2. 3. MUST each appear "
               "on a SEPARATE LINE. Do NOT chain multiple numbered points on the "
               "same line. Write concisely — do NOT repeat information that belongs "
               "in other fields.",
    }
    instructions = []
    for i, f in enumerate(profile.fields, 1):
        req_mark = "" if f.required else " (optional — return null if not applicable)"
        emphasis = _EMPHASIS.get(f.kind, "")
        instructions.append(f"{i}. {f.prompt}{emphasis}{req_mark}")

    instructions_text = "\n".join(instructions)

    # ── Build JSON schema from fields ──
    field_schemas = []
    for f in profile.fields:
        json_type = _describe_json_type(f)
        optional_note = "  // Optional — use null if not applicable" if not f.required else ""
        field_schemas.append(f'  "{f.name}": {json_type}{optional_note}')

    json_schema = "{\n" + ",\n".join(field_schemas) + "\n}"

    # ── Identify required vs optional fields ──
    required_names = [f.name for f in profile.required_fields]
    optional_names = [f.name for f in profile.optional_fields]

    prompt = textwrap.dedent(f"""\
        You are an expert researcher analyzing a paper in the {domain_name} domain.
        This is a {profile.type_name} type paper.

        TITLE: {paper.title}
        AUTHORS: {authors_str}
        YEAR: {paper.year}

        {user_lens}
        INSTRUCTIONS:
        {instructions_text}

        Return a JSON object with this EXACT structure:

        {json_schema}

        CRITICAL — AVOID REPETITION ACROSS FIELDS:
        1. Each field has a DISTINCT role. Read the FIELD-SPECIFIC GUIDANCE below
           carefully — it tells you what belongs in each field and what does NOT.
        2. problem = WHAT is the technical challenge (precise definition only).
           motivation = WHY it matters (existing approaches + their failures).
           architecture_overview = WHAT the pipeline does + HOW data flows.
           design_rationale = WHY the architecture was designed this way
           (alternatives rejected, trade-offs, dependency chains).
        3. Do NOT let these fields overlap. If you find yourself writing the
           same content in two fields, one of them is in the wrong field.
        4. Be CONCISE. A shorter, well-structured answer is better than a long
           one that repeats information from other fields.
        5. Use PROPER FORMATTING within text fields:
           - **bold** for key concept names, method names, module names
           - Blank lines between distinct topics or sub-themes
           - Bullet lists or numbered breakdowns for clarity
           - Tables (via table-format fields) for comparisons

        FIELD-SPECIFIC GUIDANCE:
        - field_evolution: EXACTLY 2 sentences. Trace the paradigm arc only —
          what each stage ACHIEVED, not what it failed at. Do NOT enumerate
          sub-problems or name individual papers. The gap at the end should be
          one phrase, not a detailed breakdown.
        - problem: Define each sub-problem precisely. Use blank lines between
          distinct sub-problems. Focus on WHAT the technical challenge IS.
        - motivation: Do NOT re-describe what VA, VLA, or any other approach
          category does (evolution already covered that). Do NOT re-list the
          sub-problems (problem already did that). Focus ONLY on stakes:
          what breaks in the real world, what capability would be unlocked.
          Write 1 paragraph. Do NOT name individual prior methods.
        - core_question: State the RESEARCH QUESTION only — the question the
          paper asks, not the answer. Do NOT describe the solution or insight
          (those go in key_insight). This is the "does X solve Y?" question.
        - key_insight: Summarize the paper in 2-3 sentences: problem → approach
          → result. This is the ANSWER, not the question.
        - related_work_context: Focus ONLY on specific technical inheritance
          and opposition. Do NOT re-describe what each research direction
          does (already covered by field_evolution and motivation). Assume
          the reader knows the landscape — only tell them what this paper
          specifically inherits and what it specifically argues against.
          Write 2-3 sentences per direction, not a full catalog.
        - architecture_overview: Describe WHAT each module does and HOW data
          flows. Use **bold** for module names. Be clear and concise. Do NOT
          explain WHY design choices were made (that goes in design_rationale).
        - architecture_figure: Trace the COMPLETE data flow from the diagram.
          Identify training-only vs inference-only paths.
        - components: For each component, specify its INPUT/OUTPUT and how it
          connects to upstream/downstream components.
        - design_rationale: This is the "WHY" section. Trace dependency chains
          between design decisions — how does choice A enable choice B? What
          alternatives were rejected and why? What trade-offs were accepted?
          Do NOT re-describe WHAT each module does or what problem it solves.
        - formulas: Include ALL formulas. For each, explain WHERE it is used in
          the architecture and WHY this specific formulation was chosen.
        - training_procedure: Capture the full training loop including any
          checkpoint/rollback mechanisms, failure handling, and human interaction.
        - training_stages: For each stage, explain WHY this ORDER — what depends
          on what. What would break if stages were reordered?
        - inference_procedure: Trace from sensor input to final action. Include
          real deployment details (hardware, latency, sensor config) if available.
        - ablation_results: Extract ALL ablation rows as a table. For each row,
          specify the CONFIGURATION, the PERFORMANCE IMPACT (exact numbers), and
          the KEY INSIGHT (what this ablation reveals).
        - evaluation_setup: Explain the evaluation framework in plain language —
          what benchmarks, what each metric means, what the reference baseline is.
          Do NOT list results here.
        - main_results: Include ALL rows with complete metric data and baseline
          comparisons. The comparison column should name the specific baseline.
        - industry_comparison: Compare this paper vs the industry baseline across
          4-6 meaningful dimensions. Focus on CONCRETE, specific differences —
          avoid vague claims. Return null if the comparison would be speculative.
        - synthesis: End with an explicit qualitative classification of the paper
          (MILESTONE / STRONG CONTRIBUTION / INCREMENTAL / EXPLORATORY).
        - Required fields (must always have a value): {', '.join(required_names)}
        - Optional fields (use null when not applicable): {', '.join(optional_names)}
        - For list fields, use [] (empty array) when there are no items.
        - For table fields, use [] (empty array) when there are no entries.
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
                "ablation_results", "contributions", "limitations",
                "training_stages", "industry_comparison"):
        if not isinstance(parsed.get(key), list):
            parsed[key] = []
    return parsed


def analyze_paper_structure(
    paper: Paper,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
    profile: Optional[PaperTypeProfile] = None,
    domain_name: str = "ai_ml",
) -> Optional[StructuredUnderstanding]:
    """Analyze a paper's structure using LLM — the core V3 abstraction.

    Applies to any paper (seed or key paper). Returns None on failure so the
    caller can fall back to heuristic analysis.

    When *profile* is given, prompt generation is tailored to that paper type.
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

    prompt = _build_analysis_prompt(paper, profile=profile, domain_name=domain_name)
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
            max_tokens=24000,
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
