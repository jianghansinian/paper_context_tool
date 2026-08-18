"""Shift-then-assign pipeline: paradigm shifts define stages, then papers are
assigned to those stages deterministically.

Pipeline:
  Step 1: Identify paradigm shifts (reuse paradigm_shift_detector logic)
  Step 2: N shifts → N+1 stages (deterministic)
  Step 3: Assign papers to stages (1 LLM call, boundaries are fixed)

This separates the unstable decision (how many stages?) from the stable one
(paper → stage assignment given fixed boundaries).

Usage:
    cd /home/pnc/ws/paper_context_tool
    python experiments/shift_then_assign.py
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
OUTPUT_DIR = Path("output/shift_then_assign")
N_RUNS = 5

# ── Step 1: Shift detection prompt ──
SHIFT_SYSTEM = """You are a research historian who identifies PARADIGM SHIFTS in a field's evolution.

A PARADIGM SHIFT is when the field's fundamental assumptions, research questions, or
success criteria changed — NOT when a method incrementally improved.

LITMUS TEST: Would a researcher from before this shift find the new approach
UNINTELLIGIBLE or OBVIOUSLY WRONG without understanding the shift itself?
If yes → paradigm shift. If they'd see it as a natural improvement → technique evolution.

Return ONLY a JSON object. No other text."""

SHIFT_USER = """Identify the PARADIGM SHIFTS in this field's evolution.

FIELD: {field_name}

PAPERS AND THEIR CLAIMS (chronological):
{claims_text}

Identify the paradigm shifts in this field's evolution. There is no fixed limit —
identify as many genuine shifts as the evidence supports. For each shift, provide:
- What changed (old paradigm → new paradigm)
- Which paper(s) triggered this shift
- When the shift occurred (year range)

Return JSON:
{{
  "shifts": [
    {{
      "shift_name": "short label, e.g. 'Dense BEV → Sparse Representation'",
      "old_paradigm": "what the field believed before",
      "new_paradigm": "what the field believed after",
      "catalyst_papers": ["paper titles"],
      "year_range": "e.g. 2022-2023"
    }}
  ]
}}
"""

# ── Step 3: Paper assignment prompt ──
ASSIGN_USER = """Below are {n_papers} papers in the field of {field}.

{claims_text}

The field's evolution is divided into {n_stages} stages, defined by {n_shifts}
paradigm shifts:

{stages_text}

Assign each paper to EXACTLY ONE stage based on which paradigm it belongs to.
A paper belongs to the stage whose paradigm it follows. If a paper TRIGGERS a
shift, it belongs to the NEW stage (after the shift).

CRITICAL RULES:
- "Sparse detection" and "end-to-end planning" are DIFFERENT stages
- A paper about sparse object detection belongs to the "sparse" stage, NOT the
  "end-to-end" stage, even if both use sparse representations
- The trigger paper of a shift belongs to the stage AFTER the shift

Return ONLY a JSON object:
{{
  "assignments": [
    {{
      "paper": "full paper title",
      "stage_index": 0,
      "reason": "one sentence: why this paper belongs to this stage"
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


def llm_call(client, system: str, user: str) -> dict | None:
    try:
        msgs = []
        if system:
            msgs.append({"role": "system", "content": system})
        msgs.append({"role": "user", "content": user})
        resp = client.chat.completions.create(
            model=_resolve_model(None),
            messages=msgs,
            temperature=1.0,
            timeout=LLM_ANALYZER_TIMEOUT_SEC,
        )
        text = resp.choices[0].message.content or ""
        return _extract_json_object(text)
    except Exception as e:
        print(f"  LLM FAIL: {e}")
        return None


def shifts_to_stages(shifts: list[dict]) -> list[dict]:
    """Convert N shifts to N+1 stages deterministically."""
    stages = []
    # Stage 0: before all shifts
    if shifts:
        stages.append({
            "index": 0,
            "name": f"Before: {shifts[0].get('old_paradigm', '??')[:50]}",
            "paradigm": shifts[0].get("old_paradigm", ""),
        })
        # Intermediate stages
        for i in range(len(shifts) - 1):
            stages.append({
                "index": i + 1,
                "name": f"After: {shifts[i].get('new_paradigm', '??')[:50]}",
                "paradigm": shifts[i].get("new_paradigm", ""),
            })
        # Last stage: after final shift
        stages.append({
            "index": len(shifts),
            "name": f"After: {shifts[-1].get('new_paradigm', '??')[:50]}",
            "paradigm": shifts[-1].get("new_paradigm", ""),
        })
    else:
        stages.append({
            "index": 0,
            "name": "Single stage",
            "paradigm": "All papers in one group",
        })
    return stages


def format_stages_for_assign(stages: list[dict], shifts: list[dict]) -> str:
    """Format stages + shift boundaries for the assignment prompt."""
    lines = []
    for i, stage in enumerate(stages):
        lines.append(f"Stage {i}: {stage['name']}")
        lines.append(f"  Paradigm: {stage['paradigm']}")
        if i < len(shifts):
            shift = shifts[i]
            lines.append(f"  → SHIFT: {shift.get('shift_name', '??')}")
            lines.append(f"    Trigger: {', '.join(shift.get('catalyst_papers', []))}")
        lines.append("")
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    papers, claims_text, field = load_claims()
    n_papers = len(papers)
    all_titles = [p["title"] for p in papers]
    print(f"{n_papers} papers, {sum(len(p['claims']) for p in papers)} claims, field={field}")

    client = build_analyzer_client()
    if not client:
        print("ERROR: no LLM client")
        sys.exit(1)

    results = []
    for run_i in range(1, N_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"  RUN {run_i}/{N_RUNS}")
        print(f"{'='*60}")

        # ── Step 1: Detect shifts ──
        t0 = time.time()
        shift_prompt = SHIFT_USER.format(field_name=field, claims_text=claims_text)
        shift_result = llm_call(client, SHIFT_SYSTEM, shift_prompt)
        dt1 = time.time() - t0

        if not shift_result or "shifts" not in shift_result:
            print(f"  Step 1 FAILED ({dt1:.1f}s)")
            continue

        shifts = shift_result["shifts"]
        print(f"  Step 1: {len(shifts)} shifts ({dt1:.1f}s)")
        for j, s in enumerate(shifts, 1):
            print(f"    {j}. {s.get('shift_name', '??')[:70]}")

        # ── Step 2: Build stages (deterministic) ──
        stages = shifts_to_stages(shifts)
        print(f"  Step 2: {len(stages)} stages (deterministic)")

        # ── Step 3: Assign papers to stages ──
        t1 = time.time()
        stages_text = format_stages_for_assign(stages, shifts)
        assign_prompt = ASSIGN_USER.format(
            n_papers=n_papers,
            n_stages=len(stages),
            n_shifts=len(shifts),
            field=field,
            claims_text=claims_text,
            stages_text=stages_text,
        )
        assign_result = llm_call(client, "", assign_prompt)
        dt2 = time.time() - t1

        if not assign_result or "assignments" not in assign_result:
            print(f"  Step 3 FAILED ({dt2:.1f}s)")
            continue

        assignments = assign_result["assignments"]
        print(f"  Step 3: {len(assignments)} assignments ({dt2:.1f}s)")

        # Build partition from assignments
        stage_papers: dict[int, list[str]] = defaultdict(list)
        for a in assignments:
            idx = a.get("stage_index", -1)
            paper = a.get("paper", "")
            if 0 <= idx < len(stages) and paper:
                stage_papers[idx].append(paper)

        sizes = [len(stage_papers.get(i, [])) for i in range(len(stages))]
        print(f"  Result: sizes={sizes}, total={sum(sizes)}")

        out_data = {
            "shifts": shifts,
            "stages": stages,
            "assignments": assignments,
        }
        out_path = OUTPUT_DIR / f"run_{run_i}.json"
        out_path.write_text(json.dumps(out_data, indent=2, ensure_ascii=False))

        results.append((run_i, shifts, stages, stage_papers, assignments))
        time.sleep(2)

    if len(results) < 2:
        print("\nNot enough runs.")
        return

    # ── Shift consistency ──
    print(f"\n{'='*70}")
    print("SHIFT CONSISTENCY ACROSS RUNS")
    print(f"{'='*70}")
    shift_counts = [len(s) for _, s, _, _, _ in results]
    print(f"  Shift counts: {shift_counts}")
    for run_i, shifts, _, _, _ in results:
        print(f"\n  run {run_i}:")
        for j, s in enumerate(shifts, 1):
            print(f"    {j}. {s.get('shift_name', '??')[:70]}")
            print(f"       catalyst: {', '.join(s.get('catalyst_papers', []))[:70]}")

    # ── Partition stability ──
    print(f"\n{'='*70}")
    print("PARTITION STABILITY")
    print(f"{'='*70}")

    def normalize_partition(stage_papers: dict[int, list[str]]) -> frozenset[frozenset[str]]:
        return frozenset(frozenset(v) for v in stage_papers.values() if v)

    partitions = [normalize_partition(sp) for _, _, _, sp, _ in results]
    unique_partitions = set(partitions)
    print(f"  Unique partitions: {len(unique_partitions)} / {N_RUNS}")

    # Jaccard
    print(f"\n  PAIRWISE JACCARD:")
    print("      " + "  ".join(f"r{i}" for i, _, _, _, _ in results))
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
    paper_stages: dict[str, list[tuple[int, int]]] = defaultdict(list)  # title -> [(run, stage_idx)]
    for run_i, _, _, stage_papers, _ in results:
        for idx, paper_list in stage_papers.items():
            for p in paper_list:
                paper_stages[p].append((run_i, idx))

    stable_count = 0
    for paper in sorted(paper_stages.keys(), key=lambda t: t[:50]):
        entries = paper_stages[paper]
        stage_indices = [idx for _, idx in entries]
        unique_stages = set(stage_indices)
        # Normalize: compare relative position (stage 0 vs 1 vs 2) not absolute index
        # since different runs may have different total stage counts
        short = paper.split(":")[0][:45]
        if len(unique_stages) == 1:
            stable_count += 1
            print(f"  [STABLE]   {short}: always stage {stage_indices[0]}")
        else:
            print(f"  [VARIES]   {short}: stages={stage_indices}")

    print(f"\n  Stability: {stable_count}/{len(paper_stages)} papers always in same stage")

    # ── Correctness check ──
    print(f"\n{'='*70}")
    print("CORRECTNESS CHECK")
    print(f"{'='*70}")
    for run_i, shifts, stages, stage_papers, _ in results:
        for idx, paper_list in stage_papers.items():
            titles_lower = [p.lower() for p in paper_list]
            has_sparse4d = any("sparse4d" in p and "v2" not in p and "sparsedrive" not in p for p in titles_lower)
            has_sparsedrive = any("sparsedrive" in p for p in titles_lower)
            if has_sparse4d and has_sparsedrive:
                print(f"  ⚠ run {run_i} stage {idx}: Sparse4D + SparseDrive together!")

    # ── Detail per run ──
    print(f"\n{'='*70}")
    print("STAGE DETAIL PER RUN")
    print(f"{'='*70}")
    for run_i, shifts, stages, stage_papers, assignments in results:
        print(f"\n  run {run_i} ({len(stages)} stages, {len(shifts)} shifts):")
        for idx in range(len(stages)):
            papers_in = stage_papers.get(idx, [])
            short_titles = [p.split(":")[0][:35] for p in papers_in]
            print(f"    Stage {idx} [{len(papers_in)}]: {stages[idx]['name'][:60]}")
            print(f"      papers: {', '.join(short_titles)}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
