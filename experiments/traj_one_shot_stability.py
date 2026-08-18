"""Test v4.4 one-shot analyzer on trajectory prediction papers.

Usage:
    PYTHONPATH=src python experiments/traj_one_shot_stability.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

from llm_analyzer import build_analyzer_client
from one_shot_analyzer import analyze_field_one_shot
from paper_cache import PaperCache

# ── 19 Trajectory Prediction papers ──
# (arxiv_id, title, year, month)
TRAJ_PAPERS = [
    ("1604.02557", "Social LSTM: Human Trajectory Prediction in Crowded Spaces", 2016, 6),
    ("1803.10892", "Social GAN: Socially Acceptable Trajectories with Generative Adversarial Networks", 2018, 6),
    ("1805.05499", "Multi-modal Trajectory Prediction of Surrounding Vehicles with Maneuver based LSTMs", 2018, 6),
    ("1802.06338", "Sequence-to-Sequence Prediction of Vehicle Trajectory via LSTM Encoder-Decoder Architecture", 2018, 6),
    ("1704.04394", "DESIRE: Distant Future Prediction in Dynamic Scenes with Interacting Agents", 2017, 4),
    ("2007.13732", "LaneGCN: Learning Lane Graph Representations for Motion Forecasting", 2020, 8),
    ("2001.03093", "Trajectron++: Dynamically Feasible Trajectory Prediction with Heterogeneous Data", 2020, 8),
    ("1910.05449", "MultiPath: Multiple Probabilistic Anchor Trajectory Hypotheses for Behavior Prediction", 2019, 10),
    ("1911.10298", "CoverNet: Multimodal Behavior Prediction Using Trajectory Sets", 2020, 6),
    ("2004.12255", "TPNet: Trajectory Proposal Network for Motion Prediction", 2020, 6),
    ("2106.08417", "Scene Transformer: A Unified Architecture for Predicting Multiple Agent Trajectories", 2022, 3),
    ("2103.14023", "AgentFormer: Agent-Aware Transformers for Socio-Temporal Multi-Agent Forecasting", 2021, 10),
    ("2008.08294", "TNT: Target-driveN Trajectory Prediction", 2020, 8),
    ("2306.03083", "MotionDiffuser: Controllable Multi-Agent Motion Prediction Using Diffusion", 2023, 6),
    ("2303.05760", "GameFormer: Game-theoretic Modeling and Learning of Transformer-based Interactive Prediction and Planning for Autonomous Driving", 2023, 10),
    ("2111.14973", "MultiPath++: Efficient Information Fusion and Trajectory Aggregation for Behavior Prediction", 2022, 5),
    # Papers without confirmed arXiv IDs — metadata only
    ("", "SPAGHETTI: Shared Cross-Modal Trajectory Prediction", 0, 0),
    ("", "LTP: Lane-Based Trajectory Prediction for Autonomous Driving", 0, 0),
    ("", "EvolveGraph: Multi-Agent Trajectory Prediction with Dynamic Relational Reasoning", 0, 0),
]

DOMAIN = "Trajectory Prediction"
N_RUNS = 3


def normalize_phase_set(result: dict) -> frozenset:
    if not result or "phases" not in result:
        return frozenset()
    return frozenset(frozenset(p.get("papers", [])) for p in result["phases"])


def main():
    # ── Step 1: Ensure papers are cached ──
    cache = PaperCache()
    print("=" * 60)
    print("  Paper Cache Status")
    print("=" * 60)

    specs = [{"arxiv_id": aid, "title": t, "year": y, "month": m}
             for aid, t, y, m in TRAJ_PAPERS]

    missing = cache.check_missing([s["arxiv_id"] for s in specs if s["arxiv_id"]])
    if missing:
        print(f"\n{len(missing)} papers need downloading:")
        for aid in missing:
            spec = next(s for s in specs if s["arxiv_id"] == aid)
            print(f"  {aid}: {spec['title'][:60]}")
        print()

    papers = cache.ensure_papers(specs, domain=DOMAIN)
    print(f"\n  Loaded {len(papers)} papers:")

    has_text = sum(1 for p in papers if p.full_text)
    print(f"  {has_text}/{len(papers)} have full text extracted")
    for p in papers:
        text_len = len(p.full_text or "")
        y = f"{p.year}-{p.month:02d}" if p.year else "????"
        print(f"    {p.title[:55]}... ({y}) ~{text_len} chars")

    # ── Step 2: Run one-shot analysis ──
    client = build_analyzer_client()
    if not client:
        print("ERROR: no LLM client")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"  Running {N_RUNS} one-shot analyses")
    print(f"{'='*60}")

    results = []
    for i in range(1, N_RUNS + 1):
        print(f"\n{'='*70}")
        print(f"Run {i}/{N_RUNS}")
        print(f"{'='*70}")
        t0 = time.time()
        result = analyze_field_one_shot(papers, DOMAIN, client)
        dt = time.time() - t0

        if not result:
            print(f"  FAILED ({dt:.1f}s)")
            continue

        phases = result.get("phases", [])
        shifts = result.get("shifts", [])
        claims = result.get("claims", [])
        tensions = result.get("tensions", [])

        n_assigned = sum(len(p.get("papers", [])) for p in phases)
        print(f"  {len(phases)} stages, {len(shifts)} shifts, {len(claims)} claims, {len(tensions)} tensions ({dt:.0f}s)")
        if n_assigned != len(papers):
            missing_p = set(p.title for p in papers) - set(t for phase in phases for t in phase.get("papers", []))
            print(f"  WARNING: {n_assigned}/{len(papers)} papers assigned. Missing: {missing_p}")

        for p in phases:
            papers_in = p.get("papers", [])
            print(f"\n    Stage {p.get('index','?')} [{len(papers_in)}]: {p.get('name','?')[:80]}")
            print(f"      Year: {p.get('year_range','?')}")
            print(f"      Dominant Question: {p.get('dominant_question','?')[:120]}")
            print(f"      Core Tension: {p.get('core_tension','?')[:120]}")
            for t in papers_in:
                print(f"        - {t[:80]}")

        if shifts:
            print(f"\n    Shifts:")
            for s in shifts:
                print(f"      {s.get('shift_name','?')[:80]}")
                print(f"        {s.get('from_phase','?')} -> {s.get('to_phase','?')}: {s.get('trigger','?')[:120]}")

        results.append((i, result))

        if i < N_RUNS:
            time.sleep(3)

    if len(results) < 2:
        print("\nNot enough successful runs to compare.")
        return

    # ── Step 3: Comparison ──
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN COMPARISON — Trajectory Prediction (v4.4)")
    print("=" * 70)

    counts = [len(r.get("phases", [])) for _, r in results]
    shift_counts = [len(r.get("shifts", [])) for _, r in results]
    paper_counts = [sum(len(p.get("papers", [])) for p in r.get("phases", [])) for _, r in results]

    print(f"\nStage counts: {counts}")
    print(f"  min={min(counts)}  max={max(counts)}  mode={max(Counter(counts).items(), key=lambda x: x[1])[0]}")
    print(f"Shift counts: {shift_counts}")
    print(f"Coverage: {paper_counts}")

    print("\n=== Stage names per run ===")
    for i, r in results:
        for p in r.get("phases", []):
            print(f"  run{i} P{p.get('index','?')}: {p.get('name','?')[:80]}")
            print(f"    papers ({len(p.get('papers',[]))}): {', '.join(t[:40] for t in p.get('papers',[]))}")

    print("\n=== Stage paper-sets ===")
    for i, r in results:
        print(f"\n  Run {i}:")
        for p in r.get("phases", []):
            papers_in = p.get("papers", [])
            print(f"    [{len(papers_in)}] {p.get('name','?')[:60]}  {p.get('year_range','?')}")
            for t in papers_in:
                print(f"        {t[:70]}")

    # ── Stability ──
    partitions = [normalize_phase_set(r) for _, r in results]
    unique = set(partitions)

    print(f"\n=== Partition stability ===")
    print(f"Unique partitions: {len(unique)} out of {len(partitions)}")
    for i, p in enumerate(partitions, 1):
        matches = [j + 1 for j, q in enumerate(partitions) if q == p]
        print(f"  run {i}: same as runs {matches}")

    print(f"\nPairwise Jaccard:")
    header = "      " + "  ".join(f"r{r[0]}" for r in results)
    print(header)
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

    print("\n=== Shift names ===")
    for i, r in results:
        for s in r.get("shifts", []):
            print(f"  run{i}: {s.get('shift_name','?')[:80]}")

    OUT_DIR = PROJECT_ROOT / "output" / "traj_one_shot"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, r in results:
        (OUT_DIR / f"run_{i}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {OUT_DIR}/")

    # ── Step 4: Cache summary ──
    print(f"\n{'='*60}")
    print("  Updated Cache Summary")
    print(f"{'='*60}")
    cache.print_summary(domain=DOMAIN)


if __name__ == "__main__":
    main()
