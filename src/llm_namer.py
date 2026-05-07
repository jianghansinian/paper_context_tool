import sys
import textwrap
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI

from config import LLM_API_KEY, LLM_BASE_URL, LLM_BRANCH_NAMING_MODEL


def refine_query(keyword: str, client: Optional[OpenAI] = None) -> str:
    """Use LLM to disambiguate acronyms and expand keyword into a precise search query."""
    if client is None:
        return keyword

    prompt = textwrap.dedent(f"""\
        You are helping refine a keyword for searching academic paper databases
        (arXiv, OpenAlex). The user typed: "{keyword}"

       1. Identify any ambiguous acronyms (e.g. BEV = Battery Electric Vehicle
          OR Bird's Eye View). Resolve based on the most common academic usage.
       2. If the keyword is vague, add 1-2 clarifying terms to improve precision.
       3. Limit output to 12 words maximum.

       Return ONLY the refined search query string, nothing else.""")

    try:
        response = client.chat.completions.create(
            model=LLM_BRANCH_NAMING_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=40,
        )
        refined = (response.choices[0].message.content or "").strip().strip("\"'")
        if refined and refined != keyword:
            print(f"Query refined: {keyword} -> {refined}")
            return refined
        return keyword
    except Exception:  # noqa: BLE001
        return keyword


def build_llm_client() -> Optional[OpenAI]:
    """Create an OpenAI-compatible client for LLM calls (chat completions)."""
    key = LLM_API_KEY
    if not key:
        print("No LLM API key configured; LLM branch naming disabled.")
        return None
    return OpenAI(api_key=key, base_url=LLM_BASE_URL, timeout=10.0, max_retries=0)


def name_branch_with_llm(
    papers: list,
    client: Optional[OpenAI] = None,
    *,
    model: Optional[str] = None,
) -> Optional[str]:
    """Generate a branch name via LLM from representative paper titles.

    Returns None when the API call fails or client is None, so callers can
    fall back to keyword-based naming.
    """
    if client is None:
        return None

    model = model or LLM_BRANCH_NAMING_MODEL
    if not model:
        return None

    sample = papers[:5]
    paper_desc = ""
    for idx, paper in enumerate(sample, 1):
        title = paper.get("title", "")
        abstract = paper.get("abstract", "")
        abstract = textwrap.shorten(abstract, width=200, placeholder="...")
        paper_desc += f"{idx}. {title}\n   {abstract}\n\n"

    prompt = textwrap.dedent(f"""\
        You are analyzing a cluster of academic papers in a research field.
        Below are representative papers from this cluster.

{paper_desc}\
        Generate a concise, technical branch name (3-8 words) that accurately
        describes the shared research direction of these papers.
        Return ONLY the branch name, nothing else.""")

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=30,
        )
        name = response.choices[0].message.content
        if name:
            return name.strip().strip("\"'")
        return None
    except Exception:  # noqa: BLE001
        return None
