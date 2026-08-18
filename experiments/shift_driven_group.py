"""Shift-driven grouping: identify paradigm shifts first, then derive stages.

Hypothesis: Doubao achieves stable grouping because it follows narrative causality
(paradigm shifts are objective historical facts), not pairwise similarity (subjective).

Pipeline:
  Step 1: LLM identifies turning points / paradigm shifts in the field's evolution
  Step 2: Turning points naturally define stage boundaries
  Step 3: Papers are assigned to stages by which paradigm they belong to

This is "narrative-first, extract structure second" — the opposite of our previous
"cluster-first" approach.

Usage:
    cd /home/pnc/ws/paper_context_tool
    python experiments/shift_driven_group.py
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
OUTPUT_DIR = Path("output/shift_driven_group")
N_RUNS = 5

USER_TEMPLATE = """Below are claims from {n_papers} papers in the field of {field}.

{claims_text}

Identify the major PARADIGM SHIFTS in this field's evolution. A paradigm shift is
a moment when the community fundamentally changed direction — not just an incremental
improvement, but a change in what kind of solution is considered viable.

Each shift creates a NEW stage. If you identify N shifts, you MUST produce N+1 stages.
Each shift is a boundary between exactly two consecutive stages.

CRITICAL RULES:
1. Different shifts MUST NOT be merged into one stage. Each shift produces its own
   boundary, creating a separate stage on each side.
2. A shift about methodology (e.g. dense→sparse) and a shift about task scope
   (e.g. perception→planning) are DIFFERENT shifts that create DIFFERENT stages.
   "Sparse detection" and "end-to-end planning" are separate stages.
3. Every paper must appear in exactly one stage.

Return ONLY a valid JSON object:
{{
  "shifts": [
    {{
      "shift": "one sentence: what changed?",
      "trigger": "which paper(s) triggered this shift?",
      "before_paradigm": "one sentence: what was the community doing before?",
      "after_paradigm": "one sentence: what did the community do after?"
    }}
  ],
  "stages": [
    {{
      "name": "short stage name",
      "paradigm": "one sentence: what paradigm defines this stage?",
      "papers": ["full paper titles in this stage"],
      "core_problem": "one sentence: what problem did this stage focus on?"
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


def normalize_partition(result: dict) -> frozenset[frozenset[str]]:
    if not result or "stages" not in result:
        return frozenset()
    return frozenset(frozenset(s.get("papers", [])) for s in result["stages"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    papers, claims_text, field = load_claims()
    n_papers = len(papers)
    all_titles = set(p["title"] for p in papers)
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

        n_shifts = len(result.get("shifts", []))
        n_stages = len(result.get("stages", []))
        sizes = [len(s.get("papers", [])) for s in result.get("stages", [])]
        # Check coverage
        covered = set()
        for s in result.get("stages", []):
            covered.update(s.get("papers", []))
        missing = all_titles - covered
        extra = covered - all_titles

        print(f"  run {i}: {n_shifts} shifts → {n_stages} stages, sizes={sizes} "
              f"(covered={len(covered)}/{n_papers}, missing={len(missing)}, extra={len(extra)}) ({dt:.1f}s)")

        out_path = OUTPUT_DIR / f"run_{i}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        results.append((i, result))
        time.sleep(2)

    if len(results) < 2:
        print("\nNot enough runs.")
        return

    # ── Shift analysis ──
    print(f"\n{'='*70}")
    print("PARADIGM SHIFTS IDENTIFIED")
    print(f"{'='*70}")
    for i, r in results:
        shifts = r.get("shifts", [])
        print(f"\n  run {i} ({len(shifts)} shifts):")
        for j, s in enumerate(shifts, 1):
            print(f"    {j}. {s.get('shift', '??')[:80]}")
            print(f"       trigger: {s.get('trigger', '??')[:60]}")
            print(f"       before: {s.get('before_paradigm', '??')[:60]}")
            print(f"       after:  {s.get('after_paradigm', '??')[:60]}")

    # ── Stage detail ──
    print(f"\n{'='*70}")
    print("STAGES (per run)")
    print(f"{'='*70}")
    for i, r in results:
        stages = r.get("stages", [])
        print(f"\n  run {i} ({len(stages)} stages):")
        for j, s in enumerate(stages, 1):
            papers_in = s.get("papers", [])
            short_titles = [p.split(":")[0][:35] for p in papers_in]
            print(f"    Stage {j} [{len(papers_in)}]: {s.get('name', '??')}")
            print(f"      paradigm: {s.get('paradigm', '??')[:70]}")
            print(f"      problem:  {s.get('core_problem', '??')[:70]}")
            print(f"      papers:   {', '.join(short_titles)}")

    # ── Partition stability ──
    partitions = [normalize_partition(r) for _, r in results]
    unique_partitions = set(partitions)

    print(f"\n{'='*70}")
    print("PARTITION STABILITY")
    print(f"{'='*70}")
    counts = [len(r.get("stages", [])) for _, r in results]
    print(f"Stage counts: {counts}")
    print(f"Unique partitions: {len(unique_partitions)} / {N_RUNS}")

    # Jaccard
    print(f"\n  PAIRWISE JACCARD:")
    print("      " + "  ".join(f"r{i}" for i, _ in results))
    for i, p1 in enumerate(partitions):
        row = []
        for j, p2 in enumerate(partitions):
            if p1 == p2:
                row.append("1.00")
            elif not p1 or not p2:
                row.append(" ?? ")
            else:
                inter = len(p1 & p2)
                union = len(p1 | p2)
                row.append(f"{inter/union:.2f}")
        print(f"  r{i+1}  " + "  ".join(row))

    # ── Per-paper assignment stability ──
    print(f"\n{'='*70}")
    print("PER-PAPER STAGE ASSIGNMENT")
    print(f"{'='*70}")
    paper_stage_labels: dict[str, list[str]] = defaultdict(list)
    for _, r in results:
        for s in r.get("stages", []):
            stage_name = s.get("name", "??")
            for p in s.get("papers", []):
                paper_stage_labels[p].append(stage_name)

    stable_count = 0
    for paper in sorted(paper_stage_labels.keys(), key=lambda t: t[:50]):
        labels = paper_stage_labels[paper]
        unique = set(labels)
        marker = "STABLE" if len(unique) == 1 else f"VARIES({len(unique)})"
        if len(unique) == 1:
            stable_count += 1
        short = paper.split(":")[0][:45]
        print(f"  [{marker}] {short}: {labels}")

    print(f"\n  Stability: {stable_count}/{len(paper_stage_labels)} papers have same stage across all runs")

    # ── Check catastrophic errors ──
    print(f"\n{'='*70}")
    print("CORRECTNESS CHECK")
    print(f"{'='*70}")
    for i, r in results:
        for s in r.get("stages", []):
            papers_in = [p.lower() for p in s.get("papers", [])]
            has_sparse4d = any("sparse4d" in p and "v2" not in p and "sparsedrive" not in p for p in papers_in)
            has_sparsedrive = any("sparsedrive" in p for p in papers_in)
            if has_sparse4d and has_sparsedrive:
                print(f"  ⚠ run {i}: Sparse4D + SparseDrive in same stage '{s.get('name', '??')}'")

    # ── Shift consistency across runs ──
    print(f"\n{'='*70}")
    print("SHIFT CONSISTENCY (semantic comparison)")
    print(f"{'='*70}")
    # Collect all shift descriptions
    all_shifts = []
    for i, r in results:
        for s in r.get("shifts", []):
            all_shifts.append((i, s.get("shift", "")))
    print(f"  Total shifts across runs: {len(all_shifts)}")
    for i, desc in all_shifts:
        print(f"    run {i}: {desc[:80]}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
