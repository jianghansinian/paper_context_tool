"""Worldview extraction experiment — test whether LLM can stably extract
each paper's "core belief" (worldview) from its claims.

Usage:
    cd /home/pnc/ws/paper_context_tool
    python experiments/worldview_experiment.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from openai import OpenAI

from paper import Paper, Claim
from claim_extractor import extract_claims
from llm_analyzer import build_analyzer_client
from paper_resolver import _fetch_arxiv_metadata

# ── Same 12 MVP papers ────────────────────────────────────────────────────
MVP_PAPERS = [
    {"arxiv_id": "2008.05711", "title": "LSS"},
    {"arxiv_id": "2112.11790", "title": "BEVDet"},
    {"arxiv_id": "2206.10092", "title": "BEVDepth"},
    {"arxiv_id": "2203.17054", "title": "BEVDet4D"},
    {"arxiv_id": "2203.17270", "title": "BEVFormer"},
    {"arxiv_id": "2211.10439", "title": "BEVFormerV2"},
    {"arxiv_id": "2308.09244", "title": "SparseBEV"},
    {"arxiv_id": "2211.10581", "title": "Sparse4D"},
    {"arxiv_id": "2305.14018", "title": "Sparse4Dv2"},
    {"arxiv_id": "2212.10156", "title": "UniAD"},
    {"arxiv_id": "2303.12077", "title": "VAD"},
    {"arxiv_id": "2405.19620", "title": "SparseDrive"},
]

# ── Worldview extraction system prompt ──────────────────────────────────────
_WORLDVIEW_SYSTEM = """\
You extract the CORE BELIEF that underlies a research paper's approach.

A core belief is the author's conviction about the STRUCTURE of the problem —
what kind of problem IS this, and therefore what kind of solution MUST work.

LITMUS TEST: If your sentence could be replaced by "We propose a method for X"
without losing insight, it's a METHOD DESCRIPTION, not a belief.
Rewrite it until it captures the CONVICTION behind the method.

A core belief is NOT:
- What problem the paper solves → that's problem_addressed
- What technique it uses → that's method, not belief
- What result it achieves → that's outcome, not belief
- "X is important" or "X can be improved" → too vague
- "A can achieve B by using C" → this is method description, not belief

A core belief IS:
- The author's stance on HOW this class of problem MUST be approached
- A falsifiable conviction: the opposite belief must also be defensible
- The "philosophy" that, if you believed it too, would make the paper's approach
  feel inevitable rather than arbitrary

GOOD beliefs (conviction, falsifiable):
  "Depth determines BEV quality; explicit supervision is essential, not optional"
    → Opposite belief: "Depth can be learned implicitly; explicit supervision is unnecessary"
  "The world is object instances; dense BEV grids are a wasteful intermediate artifact"
    → Opposite belief: "Dense BEV grids are necessary for complete scene coverage"
  "Planning is the organizing principle; perception must serve planning, not lead it"
    → Opposite belief: "Perception is independent; planning consumes its output"

BAD beliefs (method description, not conviction):
  "Temporal fusion from a single frame improves detection with minimal changes"
    → This describes WHAT they did, not WHAT THEY BELIEVE
    → Better: "Temporal cues are simple to exploit; you don't need complex architectures to benefit from them"
  "Recurrent temporal fusion reduces complexity from O(T) to O(1)"
    → This describes efficiency gain, not belief
    → Better: "Temporal fusion should be persistent and recurrent, not multi-frame stacking and discarding"
  "Sparse 4D sampling can achieve detection without dense view transformation"
    → This describes capability, not belief
    → Better: "Dense view transformation is unnecessary; direct sparse-to-sparse mapping is sufficient"

CRITICAL: Every belief must have a plausible OPPOSITE belief. If you can't imagine
someone holding the opposite conviction, you haven't found the belief yet.

Return ONLY a JSON object. No other text."""

_WORLDVIEW_PROMPT = """\
Extract the CORE BELIEF for each paper below. Each paper has claims that
reveal its underlying worldview. Read all claims for a paper, then distill
them into ONE sentence that captures the paper's core belief.

PAPERS AND THEIR CLAIMS:
{claims_text}

For each paper, output:
- paper: full paper title (exactly as given)
- belief: ONE sentence — the core conviction about HOW this problem should be solved
- why_this_belief: ONE sentence — what in the claims reveals this belief

Return JSON:
```json
{{"beliefs": [
  {{"paper": "full title", "belief": "one sentence conviction", "why_this_belief": "evidence from claims"}}
]}}
```"""


def main():
    print("=" * 60)
    print("  Worldview Extraction Experiment (5-run stability)")
    print("=" * 60)

    client = build_analyzer_client()
    if not client:
        print("ERROR: No LLM client")
        sys.exit(1)

    # ── Step 1: Resolve papers + extract claims (once) ──
    print(f"\n[1/3] Resolving {len(MVP_PAPERS)} papers + extracting claims...")
    papers: list[Paper] = []
    for i, mp in enumerate(MVP_PAPERS):
        arxiv_id = mp["arxiv_id"]
        meta = _fetch_arxiv_metadata(arxiv_id)
        paper = Paper(
            id=f"arxiv:{arxiv_id}",
            arxiv_id=arxiv_id,
            title=meta.get("title", mp["title"]) if meta else mp["title"],
            authors=meta.get("authors", []) if meta else [],
            year=meta.get("year", 0) if meta else 0,
            month=meta.get("month", 0) if meta else 0,
            abstract=meta.get("abstract", "") if meta else "",
            source="arxiv",
        )
        papers.append(paper)

    all_claims: list[Claim] = []
    for i, paper in enumerate(papers):
        claims = extract_claims(paper, client=client)
        all_claims.extend(claims)
        print(f"  [{i + 1}/12] {paper.title[:50]} — {len(claims)} claims")

    # Build claims text once (used for all runs)
    claims_by_paper: dict[str, list[Claim]] = {}
    for c in all_claims:
        claims_by_paper.setdefault(c.paper_title, []).append(c)

    papers_sorted = sorted(claims_by_paper.items(),
                           key=lambda x: (x[1][0].year, getattr(x[1][0], 'month', 0)))

    claims_lines = []
    short_names = {}  # title → short label
    for title, pclaims in papers_sorted:
        c0 = pclaims[0]
        m = getattr(c0, 'month', 0)
        date_str = f"{c0.year}-{m:02d}" if m > 0 else str(c0.year)
        claims_lines.append(f"\n[{date_str}] {title}")
        short = title.split(":")[0].strip()
        short_names[title] = short
        for j, c in enumerate(pclaims, 1):
            claims_lines.append(f"  Claim {j}: {c.statement}")
            if c.problem_addressed:
                claims_lines.append(f"    Problem addressed: {c.problem_addressed}")

    claims_text = "\n".join(claims_lines)
    print(f"  Total: {len(all_claims)} claims from {len(papers)} papers")

    # ── Step 2: Run worldview extraction 5 times ──
    N_RUNS = 5
    all_runs = []  # list[dict[short_name, belief_text]]

    print(f"\n[2/3] Running worldview extraction {N_RUNS} times...")
    for run_idx in range(N_RUNS):
        print(f"\n  Run {run_idx + 1}/{N_RUNS}...")
        beliefs = _extract_worldviews(client, claims_text)

        run_beliefs = {}
        for b in beliefs:
            title = b.get("paper", "")
            short = title.split(":")[0].strip()
            run_beliefs[short] = b.get("belief", "")
        all_runs.append(run_beliefs)

        # Print this run's beliefs
        for title, pclaims in papers_sorted:
            short = short_names[title]
            belief = run_beliefs.get(short, "MISSING")
            print(f"    {short}: {belief[:100]}...")

    # ── Step 3: Stability comparison ──
    print(f"\n[3/3] Stability analysis across {N_RUNS} runs")
    print("=" * 60)

    # Build comparison table
    paper_order = [short_names[t] for t, _ in papers_sorted]

    print(f"\n{'Paper':<18}", end="")
    for r in range(N_RUNS):
        print(f"  Run {r + 1:<50}", end="")
    print()
    print("-" * (18 + 54 * N_RUNS))

    for short in paper_order:
        print(f"{short:<18}", end="")
        for r in range(N_RUNS):
            belief = all_runs[r].get(short, "MISSING")
            belief_short = belief[:48] + ".." if len(belief) > 50 else belief
            print(f"  {belief_short:<52}", end="")
        print()

    # Semantic stability score: check if beliefs across runs share key concepts
    print(f"\n{'=' * 60}")
    print(f"  SEMANTIC STABILITY SCORE")
    print(f"{'=' * 60}")

    import difflib
    for short in paper_order:
        beliefs_across = [all_runs[r].get(short, "") for r in range(N_RUNS)]
        # Pairwise similarity
        scores = []
        for i in range(N_RUNS):
            for j in range(i + 1, N_RUNS):
                sim = difflib.SequenceMatcher(None, beliefs_across[i], beliefs_across[j]).ratio()
                scores.append(sim)
        avg_sim = sum(scores) / len(scores) if scores else 0
        bar = "█" * int(avg_sim * 20) + "░" * (20 - int(avg_sim * 20))
        print(f"  {short:<18} [{bar}] {avg_sim:.2f}")

    # Save all runs
    output = {
        "papers": paper_order,
        "runs": [{"run": i + 1, "beliefs": r} for i, r in enumerate(all_runs)],
    }
    out_path = Path("output/worldview_experiment_5runs.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    print(f"\nSaved to {out_path}")


def _extract_worldviews(client, claims_text: str) -> list[dict]:
    """Single worldview extraction call. Returns list of {paper, belief, why}."""
    prompt = _WORLDVIEW_PROMPT.format(claims_text=claims_text)
    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": _WORLDVIEW_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.1,
            max_tokens=2048,
        )
        raw = response.choices[0].message.content or ""
        raw = re.sub(r"^```(?:json)?\s*", "", raw.strip())
        raw = re.sub(r"\s*```$", "", raw)
        data = json.loads(raw)
        return data.get("beliefs", [])
    except Exception as exc:
        print(f"    Extraction failed: {exc}")
        return []


if __name__ == "__main__":
    main()
