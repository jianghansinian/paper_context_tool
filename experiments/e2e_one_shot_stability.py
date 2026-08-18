"""Test v4.4 one-shot analyzer on end-to-end autonomous driving papers.

Usage:
    PYTHONPATH=src python experiments/e2e_one_shot_stability.py
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

# ── 25 End-to-End Autonomous Driving papers ──
# (arxiv_id, title, year, month)
E2E_PAPERS = [
    # --- Pioneering (1) ---
    ("", "ALVINN: An Autonomous Land Vehicle in a Neural Network", 1989, 0),
    # --- Imitation Learning Era (2-6) ---
    ("1604.07316", "End-to-End Learning for Self-Driving Cars", 2016, 4),
    ("1710.02410", "End-to-End Driving via Conditional Imitation Learning", 2017, 10),
    ("1807.00412", "Learning to Drive in a Day", 2018, 7),
    ("1912.12294", "Learning by Cheating", 2019, 12),
    ("", "Exploring Data Aggregation in Policy Learning for Vision-Based Urban Autonomous Driving", 2020, 6),
    # --- BEV/Multi-task E2E (7-10) ---
    ("2109.04456", "NEAT: Neural Attention Fields for End-to-End Autonomous Driving", 2021, 9),
    ("2207.07601", "ST-P3: End-to-End Vision-Based Autonomous Driving via Spatial-Temporal Feature Learning", 2022, 7),
    ("2206.08129", "TCP: Trajectory-Guided Control Prediction for End-to-End Autonomous Driving", 2022, 6),
    ("2305.06242", "Think Twice Before Driving: Towards Scalable Decoders for End-to-End Autonomous Driving", 2023, 5),
    # --- Planning-Oriented E2E (11-16) ---
    ("2212.10156", "Planning-Oriented Autonomous Driving", 2022, 12),
    ("2303.13414", "VAD: Vectorized Scene Representation for Efficient Autonomous Driving", 2023, 3),
    ("2308.00398", "DriveAdapter: Breaking the Coupling Barrier of Perception and Planning in End-to-End Autonomous Driving", 2023, 8),
    ("2402.11502", "Gen-AD: Generative End-to-End Autonomous Driving", 2024, 2),
    ("2411.15139", "DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving", 2024, 11),
    ("2406.06978", "Hydra-MDP: End-to-End Multimodal Planning with Multi-Target Hydra-Distillation", 2024, 6),
    # --- LLM/VLA Era (17-23) ---
    ("2310.01412", "DriveGPT4: Interpretable End-to-End Autonomous Driving via Large Language Model", 2023, 10),
    ("2312.07488", "LMDrive: Closed-Loop End-to-End Driving with Large Language Models", 2023, 12),
    ("2312.03661", "Reason2Drive: Towards Interpretable and Chain-Based Reasoning for Autonomous Driving", 2023, 12),
    ("2310.01957", "Driving with LLMs: Fusing Object-Level Vector Modality for Explainable Autonomous Driving", 2023, 10),
    ("2602.06521", "DriveWorld-VLA: Unified Latent-Space World Modeling with Vision-Language-Action", 2026, 2),
    ("2603.27287", "Uni-World VLA: Interleaved World Modeling and Planning for Autonomous Driving", 2026, 3),
    ("", "E2E-MFD: End-to-End Multi-modal Foundation Driving Mode", 0, 0),
]

DOMAIN = "End-to-End Autonomous Driving"
N_RUNS = 3


def normalize_phase_set(result: dict) -> frozenset:
    if not result or "phases" not in result:
        return frozenset()
    return frozenset(frozenset(p.get("papers", [])) for p in result["phases"])


def main():
    cache = PaperCache()
    print("=" * 60)
    print("  Paper Cache Status")
    print("=" * 60)

    specs = [{"arxiv_id": aid, "title": t, "year": y, "month": m}
             for aid, t, y, m in E2E_PAPERS]

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
        print(f"    {p.title[:60]}... ({y}) ~{text_len} chars")

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
            print(f"  WARNING: {n_assigned}/{len(papers)} papers assigned. Missing: {len(missing_p)} papers")

        for p in phases:
            papers_in = p.get("papers", [])
            print(f"\n    Stage {p.get('index','?')} [{len(papers_in)}]: {p.get('name','?')[:90]}")
            print(f"      Year: {p.get('year_range','?')}")
            print(f"      Dominant Question: {p.get('dominant_question','?')[:130]}")
            print(f"      Core Tension: {p.get('core_tension','?')[:130]}")
            for t in papers_in:
                print(f"        - {t[:80]}")

        if shifts:
            print(f"\n    Shifts:")
            for s in shifts:
                print(f"      {s.get('shift_name','?')[:90]}")
                print(f"        {s.get('from_phase','?')} -> {s.get('to_phase','?')}: {s.get('trigger','?')[:120]}")

        results.append((i, result))

        if i < N_RUNS:
            time.sleep(3)

    if len(results) < 2:
        print("\nNot enough successful runs to compare.")
        return

    # ── Comparison ──
    print("\n" + "=" * 70)
    print("CROSS-DOMAIN COMPARISON — End-to-End Autonomous Driving (v4.4)")
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
            papers_list = [t[:40] for t in p.get('papers', [])]
            print(f"    [{len(papers_list)}] {', '.join(papers_list[:6])}{'...' if len(papers_list) > 6 else ''}")

    print("\n=== Stage paper-sets ===")
    for i, r in results:
        print(f"\n  Run {i}:")
        for p in r.get("phases", []):
            papers_in = p.get("papers", [])
            print(f"    [{len(papers_in)}] {p.get('name','?')[:60]}  {p.get('year_range','?')}")
            for t in papers_in:
                print(f"        {t[:70]}")

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

    OUT_DIR = PROJECT_ROOT / "output" / "e2e_one_shot"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for i, r in results:
        (OUT_DIR / f"run_{i}.json").write_text(json.dumps(r, indent=2, ensure_ascii=False))
    print(f"\nResults saved to {OUT_DIR}/")

    print(f"\n{'='*60}")
    print("  Updated Cache Summary")
    print(f"{'='*60}")
    cache.print_summary(domain=DOMAIN)


if __name__ == "__main__":
    main()
