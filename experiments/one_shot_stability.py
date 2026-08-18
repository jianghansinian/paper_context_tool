"""5-run stability test for v4.1 one-shot analyzer (no Paper Profile).

Usage:
    cd /home/pnc/ws/paper_context_tool
    PYTHONPATH=src python experiments/one_shot_stability.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from collections import Counter

# Ensure src is importable
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
os.chdir(PROJECT_ROOT)

from llm_analyzer import build_analyzer_client
from one_shot_analyzer import analyze_field_one_shot
from paper import Paper

# ── 12 BEV paper arxiv IDs with metadata ──
BEV_PAPERS_META = [
    ("2203.17270", "Lift, Splat, Shoot: Encoding Images From Arbitrary Camera Rigs by Implicitly Unprojecting to 3D", 2022, 3),
    ("2203.17020", "BEVDet: High-Performance Multi-Camera 3D Object Detection in Bird-Eye-View", 2022, 3),
    ("2210.08244", "BEVDet4D: Exploit Temporal Cues in Multi-camera 3D Object Detection", 2022, 10),
    ("2205.10573", "BEVDepth: Acquisition of Reliable Depth for Multi-view 3D Object Detection", 2022, 5),
    ("2204.04965", "BEVFormer: Learning Bird's-Eye-View Representation from Multi-Camera Images via Spatiotemporal Transformers", 2022, 4),
    ("2302.05971", "BEVFormer v2: Adapting Modern Image Backbones to Bird's-Eye-View Recognition via Perspective Supervision", 2023, 2),
    ("2206.07919", "Sparse4D: Multi-view 3D Object Detection with Sparse 4D Anchors", 2022, 6),
    ("2211.11750", "Sparse4D v2: Recurrent Temporal Fusion with Sparse Model", 2022, 11),
    ("2301.00904", "SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos", 2023, 1),
    ("2211.03677", "Planning-oriented Autonomous Driving", 2022, 11),
    ("2303.13414", "VAD: Vectorized Scene Representation for Efficient Autonomous Driving", 2023, 3),
    ("2304.11511", "SparseDrive: End-to-End Autonomous Driving via Sparse Scene Representation", 2023, 4),
]

N_RUNS = 5
OUTPUT_DIR = PROJECT_ROOT / "output" / "one_shot_stability"


def load_bev_papers():
    """Load BEV papers from cached PDFs, extracting text via PyMuPDF."""
    cache_dir = PROJECT_ROOT / "data/paper_cache"
    papers = []

    for arxiv_id, title, year, month in BEV_PAPERS_META:
        pdf_path = cache_dir / f"{arxiv_id}.pdf"
        if not pdf_path.exists():
            print(f"  WARNING: {arxiv_id}.pdf not found, skipping")
            continue

        # Read or extract full text
        txt_path = cache_dir / f"{arxiv_id}.txt"
        if txt_path.exists():
            full_text = txt_path.read_text(encoding="utf-8", errors="replace")
        else:
            import fitz
            doc = fitz.open(str(pdf_path))
            full_text = "\n\n".join(page.get_text() for page in doc)
            doc.close()
            txt_path.write_text(full_text, encoding="utf-8")

        # Try to load abstract from metadata
        meta_path = cache_dir / f"{arxiv_id}.json"
        abstract = ""
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8", errors="replace"))
            abstract = meta.get("abstract", "")

        paper = Paper(
            id=arxiv_id,
            arxiv_id=arxiv_id,
            title=title,
            year=year,
            month=month,
            abstract=abstract,
            full_text=full_text,
        )
        papers.append(paper)

    return papers


def normalize_phase_set(result: dict) -> frozenset:
    if not result or "phases" not in result:
        return frozenset()
    return frozenset(frozenset(p.get("papers", [])) for p in result["phases"])


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading BEV papers from cache...")
    papers = load_bev_papers()
    print(f"  Loaded {len(papers)} papers")

    if len(papers) < 8:
        print("ERROR: not enough papers loaded")
        sys.exit(1)

    for p in papers:
        text_len = len(p.full_text or "")
        print(f"  {p.title[:55]}... ({p.year}-{p.month:02d}) ~{text_len} chars")

    client = build_analyzer_client()
    if not client:
        print("ERROR: no LLM client")
        sys.exit(1)

    print(f"\nRunning {N_RUNS} times...")
    results = []
    for i in range(1, N_RUNS + 1):
        print(f"\n{'='*60}")
        print(f"Run {i}/{N_RUNS}")
        print(f"{'='*60}")
        t0 = time.time()
        result = analyze_field_one_shot(papers, "BEV Perception", client)
        dt = time.time() - t0

        if not result:
            print(f"  FAILED ({dt:.1f}s)")
            continue

        phases = result.get("phases", [])
        shifts = result.get("shifts", [])
        print(f"  {len(phases)} phases, {len(shifts)} shifts ({dt:.0f}s)")

        for p in phases:
            papers_in = p.get("papers", [])
            print(f"    Phase {p.get('index','?')} [{len(papers_in)}]: {p.get('name','?')[:70]}")
            print(f"      Q: {p.get('dominant_question','?')[:90]}")
            for t in papers_in:
                print(f"        - {t[:70]}")

        results.append((i, result))

        out_path = OUTPUT_DIR / f"run_{i}.json"
        out_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        if i < N_RUNS:
            time.sleep(3)

    if len(results) < 2:
        print("\nNot enough successful runs to compare.")
        return

    # ── Comparison ──
    print("\n" + "=" * 70)
    print("COMPARISON — v4.1 (no Paper Profile)")
    print("=" * 70)

    counts = [len(r.get("phases", [])) for _, r in results]
    shift_counts = [len(r.get("shifts", [])) for _, r in results]

    print(f"\nPhase counts: {counts}")
    print(f"  min={min(counts)}  max={max(counts)}  mode={max(Counter(counts).items(), key=lambda x: x[1])[0]}")
    print(f"Shift counts: {shift_counts}")

    # Phase names per run
    print("\nPhase names:")
    for i, r in results:
        for p in r.get("phases", []):
            print(f"  run{i} P{p.get('index','?')}: {p.get('name','?')[:60]}")

    # Phase paper assignments
    print("\nPhase paper-sets:")
    for i, r in results:
        print(f"\n  Run {i}:")
        for p in r.get("phases", []):
            papers_in = p.get("papers", [])
            short = [t.split(":")[0][:50] for t in papers_in]
            print(f"    [{len(papers_in)}] {p.get('name','?')[:50]}  {p.get('year_range','?')}")
            for s in short:
                print(f"        {s}")

    # Partition stability
    partitions = [normalize_phase_set(r) for _, r in results]
    unique = set(partitions)

    print(f"\nUnique partitions: {len(unique)} out of {len(partitions)}")
    for i, p in enumerate(partitions, 1):
        matches = [j + 1 for j, q in enumerate(partitions) if q == p]
        print(f"  run {i}: same as runs {matches}")

    # Pairwise Jaccard
    print(f"\nPairwise Jaccard:")
    print("      " + "  ".join(f"r{r[0]}" for r in results))
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

    # 3-phase hit rate
    target_3 = sum(1 for c in counts if c == 3)
    print(f"\n3-phase hit rate: {target_3}/{len(counts)} ({target_3/len(counts):.0%})")
    print(f"Lump rate (2 phases): {sum(1 for c in counts if c == 2)}/{len(counts)}")

    # Shift name consistency
    print("\nShift names:")
    for i, r in results:
        for s in r.get("shifts", []):
            print(f"  run{i}: {s.get('shift_name','?')[:80]}")

    print(f"\nResults saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
