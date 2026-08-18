"""Fine-grained grouping experiment: does finer granularity produce more stable,
more correct partitioning?

Hypothesis: finer-grained grouping is more stable because each group's boundary
is less ambiguous. Coarse grouping forces "big decisions" with no single correct
answer; fine grouping makes many small, low-ambiguity decisions.

Compared to pure_group.py, the only change is the prompt — it explicitly asks
for SPECIFIC sub-problems rather than broad research problems.

Usage:
    cd /home/pnc/ws/paper_context_tool
    python experiments/fine_group.py
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
OUTPUT_DIR = Path("output/fine_group")
N_RUNS = 5

USER_TEMPLATE = """Below are claims from {n_papers} papers in the field of {field}.

{claims_text}

Group these papers by their research problem AND innovation approach. Two papers
belong in the same group ONLY if they address the same specific problem AND share
the same core innovation approach.

Prefer FINE-GRAINED grouping: when in doubt, separate. But avoid creating groups
with only 1 paper — if a paper is truly unique, try to find the closest match
among other papers.

Each paper must appear in exactly one group.

Return ONLY a valid JSON object, no markdown fences, no other text:
{{
  "groups": [
    {{
      "problem": "one sentence: what specific problem does this group address?",
      "innovation": "one sentence: what shared innovation approach do these papers take?",
      "papers": ["full paper title as it appears above"]
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
            by_paper[pid] = {"title": c["paper_title"], "year": c["year"], "month": c["month"], "claims": []}
        by_paper[pid]["claims"].append(c["statement"])
    papers = sorted(by_paper.values(), key=lambda p: (p["year"], p["month"]))
    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(f"[{i}] {p['title']} ({p['year']}-{p['month']:02d})")
        for c in p["claims"]:
            lines.append(f"  Claim: {c}")
        lines.append("")
    return papers, "\n".join(lines), field


def run_once(client, claims_text: str, field: str, n_papers: int) -> dict | None:
    user_prompt = USER_TEMPLATE.format(n_papers=n_papers, field=field, claims_text=claims_text)
    try:
        resp = client.chat.completions.create(
            model=_resolve_model(None),
            messages=[{"role": "user", "content": user_prompt}],
            temperature=1.0,
            timeout=LLM_ANALYZER_TIMEOUT_SEC,
        )
        text = resp.choices[0].message.content or ""
        return _extract_json_object(text)
    except Exception as e:
        print(f"  FAIL: {e}")
        return None


def normalize(result: dict) -> frozenset[frozenset[str]]:
    if not result or "groups" not in result:
        return frozenset()
    return frozenset(frozenset(g.get("papers", [])) for g in result["groups"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    papers, claims_text, field = load_claims()
    n_papers = len(papers)
    print(f"{n_papers} papers, {sum(len(p['claims']) for p in papers)} claims, field={field}")

    client = build_analyzer_client()
    if not client:
        print("ERROR: no LLM client")
        sys.exit(1)

    results = []
    for i in range(1, N_RUNS + 1):
        t0 = time.time()
        result = run_once(client, claims_text, field, n_papers)
        dt = time.time() - t0
        if result is None:
            print(f"  run {i}: FAILED ({dt:.1f}s)")
            continue
        ng = len(result.get("groups", []))
        sizes = [len(g.get("papers", [])) for g in result.get("groups", [])]
        print(f"  run {i}: {ng} groups, sizes={sizes}, paper_total={sum(sizes)} ({dt:.1f}s)")
        out_path = OUTPUT_DIR / f"run_{i}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        results.append((i, result))
        time.sleep(2)

    if len(results) < 2:
        print("\nNot enough runs.")
        return

    partitions = [normalize(r) for _, r in results]
    unique_partitions = set(partitions)

    print(f"\n{'='*70}")
    print("RESULT SUMMARY")
    print(f"{'='*70}")
    counts = [len(r.get("groups", [])) for _, r in results]
    print(f"Group counts: {counts}")
    print(f"Unique partitions: {len(unique_partitions)} / {N_RUNS}")
    print()

    # ── Per-run detail ──
    for i, r in results:
        groups = r.get("groups", [])
        print(f"--- run {i} ({len(groups)} groups) ---")
        for g in groups:
            prob = g.get("problem", "??")
            innov = g.get("innovation", "")
            papers_in = g.get("papers", [])
            print(f"  [{len(papers_in)}] {prob[:70]}")
            if innov:
                print(f"        approach: {innov[:70]}")
            for p in papers_in:
                short = p.split(":")[0][:50]
                print(f"      - {short}")
        print()

    # ── Jaccard matrix ──
    print(f"{'='*70}")
    print("PAIRWISE JACCARD")
    print(f"{'='*70}")
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
                row.append(f"{inter/union:.2f}")
        print(f"  r{i}  " + "  ".join(row))

    # ── Per-paper assignment stability ──
    print(f"\n{'='*70}")
    print("PER-PAPER ASSIGNMENT STABILITY")
    print(f"{'='*70}")
    # For each paper, collect which other papers it was grouped with across runs
    paper_co_occurrence: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for _, r in results:
        for g in r.get("groups", []):
            papers_in = g.get("papers", [])
            for a in papers_in:
                for b in papers_in:
                    if a != b:
                        paper_co_occurrence[a][b] += 1

    sorted_papers = sorted(paper_co_occurrence.keys(), key=lambda t: t[:50])
    for paper in sorted_papers:
        co = paper_co_occurrence[paper]
        always_with = [p for p, c in co.items() if c == len(results)]
        sometimes_with = [p for p, c in co.items() if 0 < c < len(results)]
        never_with = [p for p, c in co.items() if c == 0 and p in paper_co_occurrence]
        short = paper.split(":")[0][:50]
        print(f"\n  {short}")
        if always_with:
            print(f"    always with: {[p.split(':')[0][:40] for p in always_with]}")
        if sometimes_with:
            print(f"    sometimes with ({[(p.split(':')[0][:30], f'{co[p]}/{len(results)}') for p in sometimes_with[:5]]})")

    # ── Comparison with pure_group ──
    print(f"\n{'='*70}")
    print("COMPARISON WITH pure_group.py")
    print(f"{'='*70}")
    print(f"  pure_group:   group counts [4,4,4,4,5], 4/5 identical, Jaccard ~0.7-1.0")
    print(f"  fine_group:   group counts {counts}, {len(unique_partitions)}/{N_RUNS} unique partitions")
    print()
    print("  Key question: does fine-grained grouping eliminate catastrophic errors")
    print("  (e.g. Sparse4D + SparseDrive in same group)?")

    # Check for known catastrophic error
    for i, r in results:
        for g in r.get("groups", []):
            papers_in = [p.lower() for p in g.get("papers", [])]
            has_sparse4d = any("sparse4d" in p and "v2" not in p and "sparse4dv2" not in p and "sparsedrive" not in p for p in papers_in)
            has_sparsedrive = any("sparsedrive" in p for p in papers_in)
            if has_sparse4d and has_sparsedrive:
                print(f"\n  ⚠ CATASTROPHIC ERROR in run {i}: Sparse4D + SparseDrive in same group!")
                print(f"    Group: {g.get('problem', '??')[:70]}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
