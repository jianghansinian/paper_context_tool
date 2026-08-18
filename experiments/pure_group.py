"""Pure grouping experiment: single LLM call, no constraints, no post-processing.

Minimal prompt. Just claims -> groups. Run 5 times, show all results.

Usage:
    cd /home/pnc/ws/paper_context_tool
    python experiments/pure_group.py
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
OUTPUT_DIR = Path("output/pure_group")
N_RUNS = 5

SYSTEM = ""

USER_TEMPLATE = """Below are claims from {n_papers} papers in the field of {field}.

{claims_text}

Group these papers into research problem groups. Papers that address the SAME
research problem should be in the SAME group. Papers that address DIFFERENT
research problems should be in DIFFERENT groups.

Each paper must appear in exactly one group.

Return ONLY a valid JSON object, no markdown fences, no other text:
{{
  "groups": [
    {{
      "problem": "one sentence: what research problem does this group address?",
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
        # check singletons
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
    print(f"Group counts: {[len(r.get('groups',[])) for _,r in results]}")
    print(f"Unique partitions: {len(unique_partitions)} / {N_RUNS}")
    print()

    for i, r in results:
        groups = r.get("groups", [])
        print(f"--- run {i} ({len(groups)} groups) ---")
        for g in groups:
            prob = g.get("problem", "??")
            papers_in = g.get("papers", [])
            print(f"  [{len(papers_in)}] {prob[:80]}")
            for p in papers_in:
                short = p.split(":")[0][:50]
                print(f"      - {short}")
        print()

    # Jaccard matrix
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

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
