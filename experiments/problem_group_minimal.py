"""Minimal problem-grouped partition experiment.

Test: does a single LLM call (claims -> per-paper problem + group papers sharing
the same problem) produce stable partitioning across 5 runs?

Parallel to phase_minimal.py but with problem as the grouping anchor instead of
"phase". Output format is identical to phase_minimal for direct comparison.

Usage:
    cd /home/pnc/ws/paper_context_tool
    python experiments/problem_group_minimal.py
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
OUTPUT_DIR = Path("output/minimal_problem_group")
N_RUNS = 5

SYSTEM = "You are a research historian. Extract the research problem each paper addresses, then group papers that share the same problem."

USER_TEMPLATE = """Here are claims from {n_papers} papers in the field of {field}.

{claims_text}

Step 1: For EACH paper, identify the ONE research problem it addresses (a problem is
the question the paper tries to answer, NOT the method it uses).

Step 2: Group papers that share the SAME research problem. Two papers belong to the
same group if they address the same core question, even if they propose different
methods. Two papers belong to different groups if they address different questions,
even if they use similar techniques.

Each paper must appear in exactly one group.

Return ONLY a JSON object:
{{
  "problem_groups": [
    {{
      "problem": "one-sentence research problem shared by this group",
      "papers": ["full paper titles in this group"]
    }}
  ]
}}
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


def normalize_partition(result: dict) -> frozenset[frozenset[str]]:
    """Convert problem_groups to set-of-sets of paper titles (order-insensitive)."""
    if not result or "problem_groups" not in result:
        return frozenset()
    groups = result["problem_groups"]
    return frozenset(frozenset(g.get("papers", [])) for g in groups)


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
        n_groups = len(result.get("problem_groups", []))
        print(f"  run {i}: {n_groups} problem groups ({dt:.1f}s)")
        out_path = OUTPUT_DIR / f"run_{i}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        results.append((i, result))
        time.sleep(2)

    if len(results) < 2:
        print("\nNot enough successful runs to compare.")
        return

    # ── Compare ──
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    counts = [len(r.get("problem_groups", [])) for _, r in results]
    print(f"\nProblem-group counts: {counts}")
    print(f"  min={min(counts)}, max={max(counts)}, mode={max(set(counts), key=counts.count)}")

    # problem names per run
    print("\nProblem labels per run:")
    for i, r in results:
        labels = [g.get("problem", "??") for g in r.get("problem_groups", [])]
        print(f"  run {i}: ({len(labels)} groups)")
        for n in labels:
            print(f"    - {n}")

    # paper set per group per run
    print("\nGroup paper-sets per run:")
    for i, r in results:
        print(f"\n  run {i}:")
        for g in r.get("problem_groups", []):
            papers_in = g.get("papers", [])
            print(f"    [{len(papers_in)}] {g.get('problem', '??')[:70]}")
            for t in papers_in:
                short = t.split(":")[0][:45] if ":" in t else t[:45]
                print(f"        - {short}")

    # ── Stability: same partition across runs? ──
    partitions = [normalize_partition(r) for _, r in results]
    unique_partitions = set(partitions)
    print(f"\n{'=' * 70}")
    print(f"Unique partitions: {len(unique_partitions)} out of {len(partitions)} runs")
    for i, p in enumerate(partitions, 1):
        match = [j + 1 for j, q in enumerate(partitions) if q == p]
        print(f"  run {i}: same as runs {match}")

    # ── Pairwise Jaccard ──
    print(f"\n{'=' * 70}")
    print("Pairwise partition overlap (Jaccard on group-sets):")
    print("      " + "  ".join(f"r{i}" for i, _ in results))
    for i, p1 in enumerate(partitions, 1):
        row = []
        for j, p2 in enumerate(partitions, 1):
            if p1 == p2:
                row.append("1.00")
            elif not p1 or not p2:
                row.append(" ?? ")
            else:
                inter = len(p1 & p2)
                union = len(p1 | p2)
                row.append(f"{inter/union:.2f}" if union else " ?? ")
        print(f"  r{i}  " + "  ".join(row))

    # ── Cross-run comparison with phase_minimal ──
    print(f"\n{'=' * 70}")
    print("COMPARISON WITH phase_minimal.py RESULTS:")
    print("  phase_minimal counts:      [4, 3, 4, 3, 3]  (4 unique partitions)")
    print(f"  problem_group counts:      {counts}  ({len(unique_partitions)} unique partitions)")
    print()
    print("  If problem_group is MORE stable (fewer unique partitions, higher Jaccard):")
    print("    → problem-driven grouping is more reliable than phase-based")
    print("  If problem_group is LESS stable or similar:")
    print("    → grouping step introduces same variance regardless of anchor")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
