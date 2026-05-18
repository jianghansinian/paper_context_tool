"""LLM-powered analysis functions for the research map pipeline.

Every function in this module follows the same contract:
- Accept `client: Optional[OpenAI]` as the last positional parameter
- Return structured data on success, None on failure
- On failure (API error / parse error / no client), the caller falls back
  to its existing heuristic behaviour.
"""

import json
import re
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

from config import (
    LLM_ANALYZER_MAX_RETRIES,
    LLM_ANALYZER_MODEL,
    LLM_ANALYZER_TIMEOUT_SEC,
    LLM_API_KEY,
    LLM_BASE_URL,
)
import config


# ---------------------------------------------------------------------------
# JSON extraction helpers
# ---------------------------------------------------------------------------

def _extract_json_array(text: str) -> Optional[List]:
    """Extract a JSON array from *text*, stripping optional markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    # find the outermost [ … ]
    m = re.search(r"(\[.*\])", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


def _extract_json_object(text: str) -> Optional[Dict]:
    """Extract a JSON object from *text*, stripping optional markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.MULTILINE)
    m = re.search(r"(\{.*\})", cleaned, re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(1))
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
# Client factory
# ---------------------------------------------------------------------------

def build_analyzer_client() -> Optional[OpenAI]:
    """Create an OpenAI-compatible client for LLM analysis calls."""
    key = LLM_API_KEY
    if not key:
        print("No LLM API key configured; LLM-powered analysis disabled.")
        return None
    return OpenAI(api_key=key, base_url=LLM_BASE_URL,
                  timeout=LLM_ANALYZER_TIMEOUT_SEC,
                  max_retries=LLM_ANALYZER_MAX_RETRIES)


def _resolve_model(model: Optional[str] = None) -> Optional[str]:
    return model or LLM_ANALYZER_MODEL


# ---------------------------------------------------------------------------
# Helper: format a paper dict for inclusion in a prompt
# ---------------------------------------------------------------------------

def _fmt_paper(idx: int, paper: Dict) -> str:
    title = paper.get("title", "")
    abstract = (paper.get("abstract") or "")[:800]  # truncate for cost
    return f"--- Paper {idx} ---\nTitle: {title}\nAbstract: {abstract}"


# ---------------------------------------------------------------------------
# 1. Relevance Filter
# ---------------------------------------------------------------------------

def filter_relevant_papers(
    papers: List[Dict],
    query: str,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
    min_score: str = "borderline",
) -> List[Dict]:
    """Use LLM to judge each paper's relevance to *query*.

    Returns the subset of papers whose relevance is at least *min_score*
    (``"relevant"`` or ``"borderline"``).  On any failure the *papers* list
    is returned unchanged (graceful degradation).
    """
    if client is None or not papers:
        return papers

    model_name = _resolve_model(model)
    if not model_name:
        return papers

    # Batch into groups of at most 30 to avoid token limits and
    # improve response reliability.
    batch_size = 30
    all_kept: List[Dict] = []
    total_removed = 0

    for batch_start in range(0, len(papers), batch_size):
        batch = papers[batch_start:batch_start + batch_size]
        formatted = "\n\n".join(
            _fmt_paper(batch_start + i, p) for i, p in enumerate(batch)
        )

        prompt = textwrap.dedent(f"""\
            You are a research relevance filter. Judge whether each paper belongs
            to the research field described below.

            Research Query: "{query}"

            Consider the title and abstract carefully. Be strict — papers from
            different fields that happen to share a keyword should be marked
            irrelevant.

            Respond with a JSON array in the same order as the papers:
            [
              {{"index": {batch_start}, "judgment": "relevant"|"borderline"|"irrelevant",
                "reason": "brief explanation"}},
              ...
            ]

            Definitions:
            - "relevant"    → clearly belongs to this research field
            - "borderline"  → tangentially related (e.g. uses the technique in a different domain)
            - "irrelevant"  → unrelated field, different meaning of keyword, clearly off-topic

            Papers:
            {formatted}""")

        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=4096,
            )
            raw = response.choices[0].message.content or ""
        except Exception as exc:  # noqa: BLE001
            print(f"Warning: Relevance filter batch API call failed ({exc}). "
                  f"Keeping all {len(batch)} papers in this batch.")
            all_kept.extend(batch)
            continue

        judgments = _extract_json_array(raw)
        if judgments is None or len(judgments) != len(batch):
            print(f"Warning: Relevance filter batch parse failed "
                  f"(got {len(judgments) if judgments else 0} judgments for "
                  f"{len(batch)} papers). Keeping all papers in this batch.")
            all_kept.extend(batch)
            continue

        for paper, j in zip(batch, judgments):
            score = (j or {}).get("judgment", "relevant")
            paper["_relevance"] = score
            if min_score == "relevant":
                if score == "relevant":
                    all_kept.append(paper)
                else:
                    total_removed += 1
            else:  # default: borderline or better
                if score in ("relevant", "borderline"):
                    all_kept.append(paper)
                else:
                    total_removed += 1

    if total_removed > 0:
        print(f"Relevance filter: kept {len(all_kept)}/{len(papers)} papers "
              f"(removed {total_removed} irrelevant).")

    _save_relevant_papers(all_kept)
    return all_kept


def _save_relevant_papers(papers: List[Dict]) -> None:
    """Persist the relevant-paper subset as intermediate output."""
    path = config.OUTPUT_RELEVANT_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "title": p.get("title"),
                    "year": p.get("year"),
                    "citation_count": p.get("citation_count"),
                    "relevance": p.get("_relevance", "unknown"),
                }
                for p in papers
            ],
            f,
            ensure_ascii=False,
            indent=2,
        )


# ---------------------------------------------------------------------------
# 2. Branch Analysis
# ---------------------------------------------------------------------------

def analyze_branch(
    papers: List[Dict],
    branch_info: Dict,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
) -> Optional[Dict]:
    """Analyse a single research branch via LLM.

    Returns a dict with keys:

    - ``branch_name`` — refined name
    - ``narrative`` — 2-3 paragraph evolution story
    - ``key_papers`` — list of dicts with *title*, *year*, *link*,
      *significance*, *importance_rank*
    - ``paradigm_shifts`` — list of *from_approach*, *to_approach*,
      *trigger_paper*
    - ``technical_forks`` — list of *description*, *representative_papers*

    Returns **None** when the LLM is unavailable or parsing fails (caller
    should fall back to heuristic ``rank_key_papers``).
    """
    if client is None or not papers:
        return None

    formatted = "\n\n".join(_fmt_paper(i, p) for i, p in enumerate(papers))

    branch_name = branch_info.get("branch_name", "unknown")
    keywords = branch_info.get("keywords", [])

    prompt = textwrap.dedent(f"""\
        You are analysing a research branch (sub-field) of academic papers.

        Research Branch: "{branch_name}"
        Branch Keywords: {keywords}

        Below are all papers in this branch. Analyse them.

        TASKS:
        1. Generate a concise technical branch name (3-8 words).
        2. Identify the 3-8 KEY papers. For each explain WHY it matters
           (e.g. "first to propose X", "achieved SOTA on Y", "introduced
           paradigm shift Z").
        3. Rank key papers by importance (1 = most important).
        4. Write a 2-3 paragraph narrative of how this branch evolved:
           initial problem, key breakthroughs, approaches tried, current
           state.
        5. If the branch has clear paradigm shifts (fundamental approach
           changes), describe them.
        6. If the branch has technical forks (divergent sub-approaches),
           describe them with representative papers.

        Respond ONLY with valid JSON (no markdown fences):
        {{
          "branch_name": "refined technical branch name",
          "narrative": "evolution narrative...",
          "key_papers": [
            {{"title": "...", "year": 2020, "link": "https://...",
              "significance": "why this paper matters", "importance_rank": 1}}
          ],
          "paradigm_shifts": [
            {{"from_approach": "old approach", "to_approach": "new approach",
              "trigger_paper": "paper that caused the shift"}}
          ],
          "technical_forks": [
            {{"description": "fork description",
              "representative_papers": ["Paper A", "Paper B"]}}
          ]
        }}

        Papers in this branch:
        {formatted}""")

    model_name = _resolve_model(model)
    if not model_name:
        return None

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=4096,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Branch analysis API call failed ({exc}). Falling back to heuristic.")
        return None

    result = _extract_json_object(raw)
    if result is None:
        print("Warning: Branch analysis JSON parse failed. Falling back to heuristic.")
        return None

    return result


# ---------------------------------------------------------------------------
# 3. Cross-Branch Evolution Analysis
# ---------------------------------------------------------------------------

def analyze_evolution(
    all_branches: List[Dict],
    field: str,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
) -> Optional[Dict]:
    """Analyse how branches relate to each other and the field's overall evolution.

    Returns a dict with keys:

    - ``overview`` — 1-2 paragraph field summary
    - ``cross_branch_relationships`` — list of *branches*, *relationship*,
      *description*
    - ``temporal_ordering`` — list of branch names ordered by historical emergence

    Returns **None** when unavailable.
    """
    if client is None or not all_branches:
        return None

    formatted = ""
    for i, branch in enumerate(all_branches, 1):
        kps = branch.get("key_papers", [])
        kp_summary = "; ".join(
            f"{p.get('title', '?')} ({p.get('year', '?')})"
            for p in (kps[:5] if isinstance(kps, list) else [])
        )
        narrative = (branch.get("narrative", "") or "")[:500]
        formatted += (
            f"--- Branch {i}: {branch.get('branch_name', '?')} ---\n"
            f"Key papers: {kp_summary}\n"
            f"Narrative: {narrative}\n\n"
        )

    prompt = textwrap.dedent(f"""\
        You are analysing a research field and its sub-branches.

        Research Field: "{field}"

        Below are all discovered branches with key papers and evolution
        narratives. Analyse how they relate to each other and describe the
        overall evolution of the field.

        Respond ONLY with valid JSON:
        {{
          "overview": "1-2 paragraph summary of how the field evolved overall",
          "cross_branch_relationships": [
            {{
              "branches": ["Branch A", "Branch B"],
              "relationship": "precursor_to|technical_fork|parallel_development|application_area",
              "description": "how these branches relate"
            }}
          ],
          "temporal_ordering": ["Branch 1", "Branch 2", ...]
        }}

        Branches:
        {formatted}""")

    model_name = _resolve_model(model)
    if not model_name:
        return None

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Evolution analysis failed ({exc}). Skipping.")
        return None

    result = _extract_json_object(raw)
    if result is None:
        print("Warning: Evolution analysis JSON parse failed. Skipping.")
        return None

    return result


# ---------------------------------------------------------------------------
# 4. Output Validation
# ---------------------------------------------------------------------------

def validate_output(
    field_map: Dict,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
) -> Optional[Dict]:
    """Non-blocking quality validation of the final field map.

    Returns a dict with *quality_score*, *issues*, *missing_topics*,
    *suggested_improvements*, or **None** when unavailable.
    """
    if client is None:
        return None

    # Build a compact textual representation of the field map
    lines = [f"Field: {field_map.get('field', '?')}\n"]
    for branch in field_map.get("branches", []):
        lines.append(f"Branch: {branch.get('branch_name', '?')}")
        lines.append(f"  Papers: {branch.get('paper_count', 0)}")
        kps = branch.get("key_papers", [])
        if isinstance(kps, list):
            for p in kps:
                title = p.get("title", "?")
                year = p.get("year", "?")
                sig = (p.get("significance", "") or "")[:100]
                lines.append(f"  Key: {title} ({year}) — {sig}")
        lines.append("")

    roadmap_text = "\n".join(lines)

    prompt = textwrap.dedent(f"""\
        You are validating a research roadmap document for quality and accuracy.

        {roadmap_text}

        Review the roadmap above. Consider:
        1. Are the branch names accurate and informative?
        2. Are the key papers genuinely important for their branches?
        3. Are there obvious missing papers or entire branches?
        4. Are any papers clearly misclassified?
        5. Is the evolution narrative coherent?

        Respond ONLY with valid JSON:
        {{
          "quality_score": 7,
          "issues": [
            {{"severity": "warning",
              "description": "specific issue description",
              "location": "Branch: X"}}
          ],
          "missing_topics": ["end-to-end learning"],
          "suggested_improvements": ["Consider adding a branch for ..."]
        }}""")

    model_name = _resolve_model(model)
    if not model_name:
        return None

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Output validation failed ({exc}). Skipping.")
        return None

    result = _extract_json_object(raw)
    if result is None:
        print("Warning: Output validation JSON parse failed. Skipping.")
        return None

    return result


# ---------------------------------------------------------------------------
# 5. Convenience: print validation report
# ---------------------------------------------------------------------------

def print_validation_report(validation: Optional[Dict]) -> None:
    """Print a human-readable quality report to stderr/stdout."""
    if not validation:
        return
    score = validation.get("quality_score", "?")
    print(f"\n{'='*50}")
    print(f"Quality validation report")
    print(f"{'='*50}")
    print(f"  Score: {score}/10")
    for issue in validation.get("issues", []):
        sev = issue.get("severity", "info")
        desc = issue.get("description", "")
        loc = issue.get("location", "")
        print(f"  [{sev.upper()}] {desc}" + (f" ({loc})" if loc else ""))
    missing = validation.get("missing_topics", [])
    if missing:
        print(f"  Missing topics: {', '.join(missing)}")
    improvements = validation.get("suggested_improvements", [])
    if improvements:
        for imp in improvements:
            print(f"  Suggestion: {imp}")
    print(f"{'='*50}\n")


# ---------------------------------------------------------------------------
# 6. Chinese Translation
# ---------------------------------------------------------------------------

def translate_field_map_for_zh(
    field_map: Dict,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
) -> Dict:
    """Translate text fields in *field_map* from English to Chinese.

    Preserves paper titles, technical terms, branch names, and URLs in
    English.  Returns a deep copy with translated text, or the original
    *field_map* unchanged on any failure (graceful degradation).
    """
    if client is None:
        return field_map

    # Collect all translatable text with path labels
    texts: List[tuple] = []

    overview = field_map.get("overview", "")
    if overview:
        texts.append(("overview", overview))

    for i, branch in enumerate(field_map.get("branches", [])):
        narrative = branch.get("narrative", "")
        if narrative:
            texts.append((f"branches[{i}].narrative", narrative))

        for j, shift in enumerate(branch.get("paradigm_shifts", [])):
            from_ = shift.get("from_approach", "")
            to_ = shift.get("to_approach", "")
            if from_:
                texts.append((f"branches[{i}].shifts[{j}].from_approach", from_))
            if to_:
                texts.append((f"branches[{i}].shifts[{j}].to_approach", to_))

        for j, fork in enumerate(branch.get("technical_forks", [])):
            desc = fork.get("description", "")
            if desc:
                texts.append((f"branches[{i}].forks[{j}].description", desc))

        for j, kp in enumerate(branch.get("key_papers", [])):
            sig = kp.get("significance", "")
            if sig:
                texts.append((f"branches[{i}].key_papers[{j}].significance", sig))

    for i, rel in enumerate(field_map.get("cross_branch_relationships", [])):
        desc = rel.get("description", "")
        if desc:
            texts.append((f"relationships[{i}].description", desc))

    if not texts:
        return field_map

    formatted = "\n\n".join(
        f"--- {path} ---\n{text}" for path, text in texts
    )

    prompt = textwrap.dedent(f"""\
        Translate the following academic research analysis texts from English
        to Simplified Chinese.

        Rules:
        1. Keep ALL paper titles in their original English form
        2. Keep ALL technical terms and proper nouns in English
           (e.g., "BEV", "Transformer", "end-to-end learning",
           "Lift-Splat-Shoot", "NeRF", etc.)
        3. Keep ALL URLs and links unchanged
        4. Keep branch names in English
        5. Translate all other text naturally to Simplified Chinese

        Respond ONLY with valid JSON (no markdown fences):
        {{
          "overview": "translated text...",
          "branches[0].narrative": "translated text...",
          "branches[0].shifts[0].from_approach": "translated text...",
          ...
        }}

        Only include keys for texts that were provided in the input.
        Texts to translate:

        {formatted}""")

    model_name = _resolve_model(model)
    if not model_name:
        return field_map

    try:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=4096,
        )
        raw = response.choices[0].message.content or ""
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: Chinese translation API call failed ({exc}). "
              f"Keeping English text.")
        return field_map

    translations = _extract_json_object(raw)
    if translations is None:
        print("Warning: Chinese translation JSON parse failed. "
              "Keeping English text.")
        return field_map

    # Apply translations to a deep copy
    import copy
    result = copy.deepcopy(field_map)

    if "overview" in translations and translations["overview"]:
        result["overview"] = translations["overview"]

    for key, translated_text in translations.items():
        if key == "overview" or not translated_text:
            continue
        parts = key.split(".")
        if len(parts) < 2:
            continue

        try:
            if parts[0] == "branches":
                idx = int(parts[1].strip("[]"))
                branch = result["branches"][idx]
                if parts[2] == "narrative":
                    branch["narrative"] = translated_text
                elif parts[2] == "shifts":
                    shift_idx = int(parts[3].strip("[]"))
                    field_name = parts[4]
                    branch["paradigm_shifts"][shift_idx][field_name] = translated_text
                elif parts[2] == "forks":
                    fork_idx = int(parts[3].strip("[]"))
                    branch["technical_forks"][fork_idx]["description"] = translated_text
                elif parts[2] == "key_papers":
                    kp_idx = int(parts[3].strip("[]"))
                    branch["key_papers"][kp_idx]["significance"] = translated_text
            elif parts[0] == "relationships":
                idx = int(parts[1].strip("[]"))
                result["cross_branch_relationships"][idx][
                    "description"
                ] = translated_text
        except (IndexError, ValueError, KeyError):
            continue

    return result
