"""V4 E2E Field Evolution Pipeline.

field_name → V3 retrieval → cache + full-text download → one-shot stage analysis
→ narrative generation → {slug}_evolution.md

Design: docs/design_pipeline_e2e.md (E2E V3).

Usage:
    python src/run_v4.py "BEV Perception"
    python src/run_v4.py "BEV Perception" --no-download
    python src/run_v4.py "BEV Perception" --fast
    python src/run_v4.py --resume output/v4/xxx/one_shot_result.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC))
os.chdir(PROJECT_ROOT)

import config
from llm_analyzer import build_analyzer_client
from one_shot_analyzer import analyze_field_one_shot
from one_shot_narrative import generate_evolution_md
from paper_cache import PaperCache
from paper_retriever_v3 import retrieve_field_papers_v3


def _interactive_download_callback(title: str, elapsed: float, remaining: int) -> str:
    """Called when a PDF download exceeds timeout. Prompts user interactively."""
    print(f"\n  ⚠ Downloading \"{title[:60]}\" is slow (>{elapsed:.0f}s).")
    print(f"  Remaining: {remaining} paper(s) to download.")
    while True:
        try:
            choice = input("  [C]ontinue waiting  [S]kip remaining downloads  [Q]uit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "skip"
        if choice in ("c", "continue"):
            return "continue"
        if choice in ("s", "skip"):
            return "skip"
        if choice in ("q", "quit"):
            return "quit"
        print("  Please enter C, S, or Q.")


def _make_specs(selected: list[dict]) -> list[dict]:
    """Map V3 selected dicts → PaperCache specs (abstract passthrough)."""
    specs = []
    for p in selected:
        specs.append({
            "arxiv_id": p.get("arxiv_id", "") or "",
            "title": p.get("title", ""),
            "year": p.get("year", 0),
            "month": p.get("month", 0),
            "abstract": p.get("abstract", "") or "",
        })
    return specs


def _load_paper_objs(cache: PaperCache, selected: list[dict], args) -> list:
    """Load/download full text for selected papers. Returns list of Paper objects."""
    specs = _make_specs(selected)

    if args.no_download:
        print("\n--no-download: skipping PDF download, using cache + metadata only")
        from paper import Paper
        paper_objs = []
        for spec in specs:
            full_text = ""
            aid = spec["arxiv_id"]
            if aid:
                txt_path = cache.dir / f"{aid}.txt"
                if txt_path.exists():
                    full_text = txt_path.read_text(encoding="utf-8", errors="replace")
            paper_objs.append(Paper(
                id=aid or spec["title"], arxiv_id=aid or None,
                title=spec["title"], year=spec["year"], month=spec["month"],
                abstract=spec["abstract"], full_text=full_text,
            ))
        return paper_objs

    print(f"\nChecking cache for {len(specs)} papers...")
    arxiv_ids = [s["arxiv_id"] for s in specs if s["arxiv_id"]]
    missing = cache.check_missing(arxiv_ids)
    if missing:
        print(f"  {len(missing)}/{len(arxiv_ids)} papers need downloading")
    else:
        print("  All papers already cached")

    return cache.ensure_papers(
        specs,
        domain=args.domain,
        download_timeout=args.download_timeout,
        on_slow_download=_interactive_download_callback,
    )


def _print_paper_status(paper_objs) -> int:
    n_full = 0
    for p in paper_objs:
        text_len = len(p.full_text or "")
        if text_len > 500:
            flag, n_full = "FULL", n_full + 1
        elif text_len == 0:
            flag = "META"
        else:
            flag = "PART"
        print(f"    [{flag}] {p.title[:60]}... ({p.year}) ~{text_len} chars")
    return n_full


def main():
    ap = argparse.ArgumentParser(description="V4 E2E Field Evolution Pipeline")
    ap.add_argument("field", nargs="?", default="",
                    help='Field name, e.g. "BEV Perception"')
    ap.add_argument("--fast", action="store_true",
                    help="Skip LLM seed generation in retrieval (OpenAlex top by citations)")
    ap.add_argument("--max-papers", type=int, default=config.V3_MAX_PAPERS,
                    help=f"Max papers to retrieve (default: {config.V3_MAX_PAPERS})")
    ap.add_argument("--no-download", action="store_true",
                    help="Skip PDF download, use cache + metadata (abstract) only")
    ap.add_argument("--download-timeout", type=int, default=60,
                    help="Seconds per PDF before asking user to skip (default: 60)")
    ap.add_argument("--domain", default="",
                    help="Domain tag for cache (default: same as field name)")
    ap.add_argument("--resume", default="",
                    help="Resume from saved one_shot_result.json (skip retrieval + download + analysis)")
    args = ap.parse_args()

    client = build_analyzer_client()

    # ── Resume path: skip to Step 4 ──
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            print(f"ERROR: --resume file not found: {resume_path}")
            sys.exit(1)
        result = json.loads(resume_path.read_text())
        out_dir = resume_path.parent
        field_name = args.field or args.domain
        run_info_path = out_dir / "run_info.json"
        if not field_name and run_info_path.exists():
            field_name = json.loads(run_info_path.read_text()).get("field_name", "")
        if not field_name:
            field_name = "Unknown Field"
        print(f"Resuming from: {resume_path}")
    else:
        if not args.field:
            ap.error("Either --resume or a field name is required")
        if not client:
            print("ERROR: no LLM client available (LLM_API_KEY not configured)")
            sys.exit(1)

        field_name = args.field
        args.domain = args.domain or field_name

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = field_name.lower().replace(" ", "_").replace("-", "_")
        out_dir = Path("output") / "v4" / f"{ts}_{slug}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output dir: {out_dir}")

        # ── Step 1: V3 Retrieval ──
        print(f"\n{'='*60}")
        print(f"  Step 1: V3 Paper Retrieval: {field_name}")
        print(f"{'='*60}")

        selected, report = retrieve_field_papers_v3(
            field_name, client,
            max_papers=args.max_papers,
            fast=args.fast,
        )
        if not selected:
            print("ERROR: no papers retrieved.")
            sys.exit(1)

        retrieval_out = {
            "field_name": field_name,
            "selected": selected,
            "report": report,
        }
        (out_dir / "selected_papers.json").write_text(
            json.dumps(retrieval_out, ensure_ascii=False, indent=2))

        # ── Step 2: Cache + Download ──
        print(f"\n{'='*60}")
        print(f"  Step 2: Cache check + download")
        print(f"{'='*60}")

        cache = PaperCache()
        paper_objs = _load_paper_objs(cache, selected, args)

        n_full = _print_paper_status(paper_objs)
        print(f"\n  Loaded {len(paper_objs)} papers: {n_full} with full text, "
              f"{len(paper_objs) - n_full} metadata-only")
        if n_full == 0:
            print("  WARNING: 0 papers have full text. Analysis quality will be degraded.")

        # ── Step 3: One-Shot Analysis ──
        print(f"\n{'='*60}")
        print(f"  Step 3: One-shot analysis")
        print(f"{'='*60}")

        t0 = time.time()
        result = analyze_field_one_shot(paper_objs, field_name, client)
        dt = time.time() - t0

        if not result:
            print("ERROR: one-shot analysis failed")
            sys.exit(1)

        phases = result.get("phases", [])
        shifts = result.get("shifts", [])
        claims = result.get("claims", [])
        tensions = result.get("tensions", [])
        n_assigned = sum(len(p.get("papers", [])) for p in phases)
        print(f"\n  {len(phases)} phases, {len(shifts)} shifts, {len(claims)} claims, "
              f"{len(tensions)} tensions ({dt:.0f}s)")
        print(f"  Coverage: {n_assigned}/{len(paper_objs)} papers assigned")

        for p in phases:
            papers_in = p.get("papers", [])
            print(f"\n    Phase {p.get('index','?')} [{len(papers_in)}]: {p.get('name','?')[:80]}")
            print(f"      {p.get('year_range','?')} | {p.get('dominant_question','?')[:100]}")
            for t in papers_in:
                print(f"        - {t[:70]}")

        (out_dir / "one_shot_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2))
        print(f"\n  One-shot result saved: {out_dir / 'one_shot_result.json'}")

        run_info = {
            "field_name": field_name,
            "ts": ts,
            "n_papers": len(paper_objs),
            "n_full_text": n_full,
        }
        (out_dir / "run_info.json").write_text(
            json.dumps(run_info, ensure_ascii=False, indent=2))

    # ── Step 4: Narrative Generation ──
    print(f"\n{'='*60}")
    print(f"  Step 4: Narrative generation")
    print(f"{'='*60}")

    if not client:
        print("ERROR: no LLM client available for narrative generation")
        sys.exit(1)

    md = generate_evolution_md(result, field_name, client)
    if not md:
        print("ERROR: narrative generation failed")
        sys.exit(1)

    slug = field_name.lower().replace(" ", "_").replace("-", "_")
    md_path = out_dir / f"{slug}_evolution.md"
    md_path.write_text(md)
    print(f"\n  Evolution narrative: {md_path} ({len(md):,} chars)")

    if not args.resume:
        print(f"\n  Cache Summary:")
        cache.print_summary(domain=args.domain)

    print(f"\n{'='*60}")
    print(f"  Done!")
    print(f"{'='*60}")
    print(f"  Output: {out_dir}/")
    print(f"  - selected_papers.json")
    print(f"  - one_shot_result.json")
    print(f"  - {slug}_evolution.md")


if __name__ == "__main__":
    main()
