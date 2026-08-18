"""Minimal problem extraction experiment.

Test: does a single LLM call (claims -> per-paper problem) produce stable
problem assignment across 5 runs?

Parallel to phase_minimal.py but for problem-driven framing.
Only extraction — no grouping, no edges, no hierarchy.

Usage:
    cd /home/pnc/ws/paper_context_tool
    python experiments/problem_minimal.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_analyzer import build_analyzer_client, _extract_json_object, _resolve_model
from config import LLM_ANALYZER_TIMEOUT_SEC

INPUT_NARRATIVE = "output/v8_mvp_069/narrative.json"
OUTPUT_DIR = Path("output/minimal_problem")
N_RUNS = 5

SYSTEM = "You are a research historian. Extract what research problem each paper addresses."

USER_TEMPLATE = """Here are claims from {n_papers} papers in the field of {field}.

{claims_text}

For EACH paper, extract the ONE research problem it addresses. A research problem
is the question the paper is trying to answer, NOT the method it uses.

Return ONLY a JSON object:
{{
  "papers": [
    {{
      "paper": "full paper title",
      "problem": "one-sentence research problem this paper addresses",
      "year": 2020
    }}
  ]
}}

Notes:
- Extract problems as questions or problem statements, not methods.
- Be specific: "how to estimate depth for BEV" is better than "3D perception".
- One problem per paper (the dominant one).
"""


def load_claims():
    data = json.load(open(INPUT_NARRATIVE))
    claims = data["claims"]
    field = data.get("field_name", "BEV perception")

    by_paper: dict[str, dict] = {}
    for c in claims:
        pid = c["paper_id"]
        if pid not in by_paper:
            by_paper[pid] = {
                "title": c["paper_title"],
                "year": c["year"],
                "month": c["month"],
                "claims": [],
            }
        by_paper[pid]["claims"].append(c["statement"])

    papers = sorted(by_paper.values(), key=lambda p: (p["year"], p["month"]))

    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(f"[Paper {i}] {p['title']} ({p['year']}-{p['month']:02d})")
        for c in p["claims"]:
            lines.append(f"  - {c}")
        lines.append("")

    return papers, "\n".join(lines), field


def run_once(client, claims_text: str, field: str, n_papers: int) -> dict | None:
    user_prompt = USER_TEMPLATE.format(
        n_papers=n_papers,
        field=field,
        claims_text=claims_text,
    )
    try:
        resp = client.chat.completions.create(
            model=_resolve_model(None),
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            temperature=1.0,
            timeout=LLM_ANALYZER_TIMEOUT_SEC,
        )
        text = resp.choices[0].message.content or ""
        return _extract_json_object(text)
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return None


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading claims from {INPUT_NARRATIVE}")
    papers, claims_text, field = load_claims()
    n_papers = len(papers)
    print(f"  {n_papers} papers, {sum(len(p['claims']) for p in papers)} claims")
    print(f"  field: {field}")

    client = build_analyzer_client()
    if not client:
        print("ERROR: no LLM client.")
        sys.exit(1)

    print(f"\nRunning {N_RUNS} times...")
    results = []
    for i in range(1, N_RUNS + 1):
        t0 = time.time()
        result = run_once(client, claims_text, field, n_papers)
        dt = time.time() - t0
        if result is None:
            print(f"  run {i}: FAILED ({dt:.1f}s)")
            continue
        n_extracted = len(result.get("papers", []))
        print(f"  run {i}: {n_extracted} papers extracted ({dt:.1f}s)")
        out_path = OUTPUT_DIR / f"run_{i}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        results.append((i, result))
        time.sleep(2)

    if len(results) < 2:
        print("\nNot enough successful runs to compare.")
        return

    # ── Build paper -> problem list across runs ──
    title_to_problems: dict[str, list[str]] = defaultdict(list)
    title_to_year = {}
    for _, r in results:
        for p in r.get("papers", []):
            t = p.get("paper", "")
            prob = p.get("problem", "")
            title_to_problems[t].append(prob)
            if t not in title_to_year:
                title_to_year[t] = p.get("year", 0)

    # ── Per-paper stability ──
    print("\n" + "=" * 70)
    print("PER-PAPER PROBLEM STABILITY")
    print("=" * 70)
    # sort by year
    sorted_titles = sorted(title_to_problems.keys(), key=lambda t: title_to_year.get(t, 9999))
    stable_count = 0
    for t in sorted_titles:
        probs = title_to_problems[t]
        unique = set(probs)
        n_unique = len(unique)
        marker = "STABLE" if n_unique == 1 else f"VARIES ({n_unique})"
        if n_unique == 1:
            stable_count += 1
        print(f"\n  [{marker}] {t[:70]}")
        for prob in probs:
            print(f"    - {prob}")

    print(f"\n{'=' * 70}")
    print(f"Stability: {stable_count}/{len(sorted_titles)} papers have identical problem across all runs")

    # ── Cross-paper problem set comparison ──
    print(f"\n{'=' * 70}")
    print("UNIQUE PROBLEM STRINGS PER RUN (raw, before semantic clustering):")
    for i, r in results:
        probs = [p.get("problem", "") for p in r.get("papers", [])]
        unique_probs = sorted(set(probs))
        print(f"\n  run {i} ({len(unique_probs)} unique problems):")
        for p in unique_probs:
            print(f"    - {p}")

    # ── Papers grouped by exact-match problem string, per run ──
    print(f"\n{'=' * 70}")
    print("PAPER GROUPINGS BY EXACT PROBLEM STRING (per run):")
    for i, r in results:
        groups: dict[str, list[str]] = defaultdict(list)
        for p in r.get("papers", []):
            groups[p.get("problem", "")].append(p.get("paper", "")[:45])
        print(f"\n  run {i} ({len(groups)} distinct problem strings → {len(groups)} groups):")
        for prob, papers_list in sorted(groups.items(), key=lambda x: -len(x[1])):
            print(f"    [{len(papers_list)}] {prob[:70]}")
            for pt in papers_list:
                print(f"        - {pt}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
