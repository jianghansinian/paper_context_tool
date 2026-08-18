"""Minimal phase detection experiment.

Test: does a single LLM call (claims -> phases) produce stable phase structure
across 5 runs?

Deliberately minimal prompt: no litmus test, no belief extraction, no
validation heuristics, no paradigm shift detection. Just raw claims -> phases.

Usage:
    cd /home/pnc/ws/paper_context_tool
    python experiments/phase_minimal.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from llm_analyzer import build_analyzer_client, _extract_json_object, _resolve_model
from config import LLM_ANALYZER_TIMEOUT_SEC

# ── Config ──
INPUT_NARRATIVE = "output/v8_mvp_069/narrative.json"
OUTPUT_DIR = Path("output/minimal_phase")
N_RUNS = 5

# ── Prompt (intentionally bare) ──
SYSTEM = "You are a research historian. Group papers into phases of technical evolution."

USER_TEMPLATE = """Here are claims from {n_papers} papers in the field of {field}.

{claims_text}

Group these papers into phases of technical evolution. Each paper must appear in exactly one phase.

Return ONLY a JSON object:
{{
  "phases": [
    {{
      "name": "short phase name",
      "papers": ["full paper titles in this phase"],
      "time_range": "YYYY-MM—YYYY-MM"
    }}
  ]
}}
"""


def load_claims() -> tuple[list[dict], str]:
    """Load claims and build claims text grouped by paper."""
    data = json.load(open(INPUT_NARRATIVE))
    claims = data["claims"]
    field = data.get("field_name", "BEV perception")

    # group by paper
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

    # sort by time
    papers = sorted(by_paper.values(), key=lambda p: (p["year"], p["month"]))

    lines = []
    for i, p in enumerate(papers, 1):
        lines.append(f"[Paper {i}] {p['title']} ({p['year']}-{p['month']:02d})")
        for c in p["claims"]:
            lines.append(f"  - {c}")
        lines.append("")

    return papers, "\n".join(lines), field


def run_once(client, claims_text: str, field: str, n_papers: int) -> dict | None:
    """One LLM call -> phases JSON."""
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
            temperature=1.0,  # default sampling — don't force determinism
            timeout=LLM_ANALYZER_TIMEOUT_SEC,
        )
        text = resp.choices[0].message.content or ""
        return _extract_json_object(text)
    except Exception as e:
        print(f"  LLM call failed: {e}")
        return None


def normalize_phase_set(result: dict) -> frozenset[frozenset[str]]:
    """Convert phases to a set-of-sets of paper titles (order-insensitive)."""
    if not result or "phases" not in result:
        return frozenset()
    phases = result["phases"]
    return frozenset(frozenset(p.get("papers", [])) for p in phases)


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
        n_phases = len(result.get("phases", []))
        print(f"  run {i}: {n_phases} phases ({dt:.1f}s)")
        out_path = OUTPUT_DIR / f"run_{i}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
        results.append((i, result))
        # small delay to avoid rate limit
        time.sleep(2)

    if len(results) < 2:
        print("\nNot enough successful runs to compare.")
        return

    # ── Compare ──
    print("\n" + "=" * 70)
    print("COMPARISON")
    print("=" * 70)

    # phase counts
    counts = [len(r.get("phases", [])) for _, r in results]
    print(f"\nPhase counts: {counts}")
    print(f"  min={min(counts)}, max={max(counts)}, mode={max(set(counts), key=counts.count)}")

    # phase names
    print("\nPhase names per run:")
    for i, r in results:
        names = [p.get("name", "??") for p in r.get("phases", [])]
        print(f"  run {i}: ({len(names)} phases)")
        for n in names:
            print(f"    - {n}")

    # paper set per phase per run
    print("\nPhase paper-sets per run:")
    for i, r in results:
        print(f"\n  run {i}:")
        for p in r.get("phases", []):
            papers_in = p.get("papers", [])
            # short label
            short = [t.split(":")[0][:40] if ":" in t else t[:40] for t in papers_in]
            print(f"    [{len(papers_in)}] {p.get('name', '??')[:60]}")
            for s in short:
                print(f"        - {s}")

    # ── Stability check: how many runs share the same phase partition ──
    partitions = [normalize_phase_set(r) for _, r in results]
    unique_partitions = set(partitions)
    print(f"\n{'=' * 70}")
    print(f"Unique partitions: {len(unique_partitions)} out of {len(partitions)} runs")
    for i, p in enumerate(partitions, 1):
        match = [j + 1 for j, q in enumerate(partitions) if q == p]
        print(f"  run {i}: same as runs {match}")

    # ── Pairwise overlap matrix ──
    print(f"\n{'=' * 70}")
    print("Pairwise partition overlap (Jaccard on phase-sets):")
    print("      " + "  ".join(f"r{i}" for i, _ in results))
    for i, p1 in enumerate(partitions, 1):
        row = []
        for j, p2 in enumerate(partitions, 1):
            if p1 == p2:
                row.append("1.00")
            elif not p1 or not p2:
                row.append(" ?? ")
            else:
                # Jaccard on sets of phases (each phase is a frozenset of papers)
                inter = len(p1 & p2)
                union = len(p1 | p2)
                row.append(f"{inter/union:.2f}" if union else " ?? ")
        print(f"  r{i}  " + "  ".join(row))

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
