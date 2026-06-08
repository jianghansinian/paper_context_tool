"""V3 pipeline entry point: Seed-paper-centric structured understanding.

Usage:
    python src/run_v3.py <arxiv_url_or_pdf_path> [description] [--route]

Options:
    --route     Enable technical route analysis + comparative analysis (default: off)

Examples:
    python src/run_v3.py https://arxiv.org/abs/2203.17270
    python src/run_v3.py 2203.17270 "focus on temporal attention"
    python src/run_v3.py /path/to/paper.pdf --route
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import config
from paper_resolver import resolve_paper, _download_arxiv_pdf
from structured_analyzer import analyze_paper_structure
from citation_miner import CitationMiner
from route_analyzer import analyze_routes, compare_with_mainstream
from markdown_exporter_v3 import export_markdown, translate_markdown_to_zh
from llm_analyzer import build_analyzer_client
from text_extractor import extract_text_from_pdf
from domains.ai_ml import EXPERIMENTAL_PROFILE
from domain_detector import detect_domain
from paper_type_detector import detect_paper_type


def _print_separator(title: str) -> None:
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}")


def _init_v3_run_dir(paper) -> Path:
    """Create timestamped output directory for this V3 run."""
    slug = paper.title[:80].replace(" ", "-").replace("/", "-")
    # Strip non-filesystem-safe chars
    slug = "".join(c for c in slug if c.isalnum() or c in "._-")
    if not slug:
        slug = "paper"

    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = config.V3_OUTPUT_DIR / f"{timestamp}_{slug}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def main():
    if len(sys.argv) < 2:
        print('Usage: python src/run_v3.py <arxiv_url_or_pdf_path> [description] [--route]')
        print()
        print('Options:')
        print('  --route   Enable technical route analysis + comparative analysis (default: off)')
        print()
        print('Examples:')
        print('  python src/run_v3.py https://arxiv.org/abs/2203.17270')
        print('  python src/run_v3.py 2203.17270 "focus on temporal attention"')
        print('  python src/run_v3.py /path/to/paper.pdf --route')
        sys.exit(1)

    # Parse --route flag
    args = [a for a in sys.argv[1:] if a != "--route"]
    route_enabled = "--route" in sys.argv

    if route_enabled:
        config.V3_ROUTE_ANALYSIS_ENABLED = True

    target = args[0].strip()
    user_description = args[1].strip() if len(args) > 1 else ""

    # ── Build LLM client ──
    llm_client = build_analyzer_client()

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Paper Resolution
    # ══════════════════════════════════════════════════════════════
    _print_separator("Phase 1: Resolving paper")
    try:
        seed_paper = resolve_paper(target, user_description)
    except Exception as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    print(f"Title:    {seed_paper.title}")
    print(f"Authors:  {', '.join(seed_paper.authors[:5])}"
          f"{'...' if len(seed_paper.authors) > 5 else ''}")
    print(f"Year:     {seed_paper.year}")
    print(f"Citations: {seed_paper.citation_count}")
    print(f"Full text: {'yes' if seed_paper.full_text else 'no'}"
          f" ({len(seed_paper.full_text)} chars)" if seed_paper.full_text else "")

    if not seed_paper.full_text:
        print("")
        print("  ⚠ WARNING: Full text is unavailable. Analysis quality will be significantly")
        print("  degraded — only abstract-based analysis is possible. Formulas, detailed")
        print("  architecture, training procedures, and experimental results may be missing.")
        print("  Possible causes: arXiv PDF not yet indexed (new paper), rate-limiting,")
        print("  or network error. Retry later or provide a local PDF file.")
        print("  Set V3_STRUCTURED_ANALYSIS_ENABLED=0 to skip analysis and inspect metadata only.")

    run_dir = _init_v3_run_dir(seed_paper)
    print(f"Output:   {run_dir}")

    # ══════════════════════════════════════════════════════════════
    # Phase 1.2: Domain detection + Paper type detection
    # ══════════════════════════════════════════════════════════════
    _print_separator("Phase 1.2: Domain & paper type detection")
    domain = detect_domain(seed_paper, llm_client)
    print(f"Detected domain: {domain.domain_name} ({domain.domain_description[:60]}...)")

    paper_type = detect_paper_type(seed_paper, domain, llm_client)
    profile = domain.get_paper_type(paper_type)
    if profile is None:
        profile = domain.paper_types[0]  # fallback to first defined type
    print(f"Detected type: {paper_type}")

    # ══════════════════════════════════════════════════════════════
    # Phase 1.5: Structured Understanding (seed paper)
    # ══════════════════════════════════════════════════════════════
    if config.V3_STRUCTURED_ANALYSIS_ENABLED:
        _print_separator("Phase 1.5: Structured Understanding (seed paper)")
        structured = analyze_paper_structure(seed_paper, llm_client,
                                             profile=profile,
                                             domain_name=domain.domain_name)
        if structured:
            seed_paper.structured = structured
            print(f"Structured understanding complete.")
            print(f"  Problem: {structured.problem[:120]}...")
            print(f"  Components: {len(structured.components)}")
            print(f"  Formulas: {len(structured.formulas)}")
            print(f"  Results: {len(structured.main_results)}")
        else:
            print("Structured understanding unavailable — continuing without it.")
    else:
        print("Structured understanding disabled (V3_STRUCTURED_ANALYSIS_ENABLED=0).")

    # ══════════════════════════════════════════════════════════════
    # Phase 2-3: Citation Mining
    # ══════════════════════════════════════════════════════════════
    miner = CitationMiner()
    all_refs: list = []

    if config.V3_CITATION_MINING_ENABLED:
        _print_separator("Phase 2: Backward citation mining")
        try:
            miner.mine_references(seed_paper,
                                  max_depth=config.REFERENCE_MAX_DEPTH,
                                  llm_client=llm_client)
            all_refs = miner.classify_references(seed_paper, llm_client=llm_client)
            print(f"Found {len(miner.get_all_papers())} papers via backward mining")
            print(f"Classified {len(all_refs)} references")

            supporting = sum(1 for r in all_refs if r.citation_type.value == "supporting")
            contrasting = sum(1 for r in all_refs if r.citation_type.value == "contrasting")
            foundational = sum(1 for r in all_refs if r.citation_type.value == "foundational")
            print(f"  Supporting: {supporting}, Contrasting: {contrasting}, Foundational: {foundational}")
        except Exception as exc:
            print(f"Backward citation mining failed: {exc}")

        _print_separator("Phase 3: Forward citation mining")
        try:
            miner.mine_citations(seed_paper)
            print(f"Total papers in pool: {len(miner.get_all_papers())}")
        except Exception as exc:
            print(f"Forward citation mining failed: {exc}")
    else:
        print("Citation mining disabled (V3_CITATION_MINING_ENABLED=0).")

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Key paper analysis (structured understanding)
    # ══════════════════════════════════════════════════════════════
    key_papers = miner.get_key_papers() if config.V3_CITATION_MINING_ENABLED else []

    if key_papers and config.V3_STRUCTURED_ANALYSIS_ENABLED:
        _print_separator(f"Phase 4: Analyzing {len(key_papers)} key papers")
        analyzed = 0
        for i, paper in enumerate(key_papers):
            # Directly download PDF (skip redundant metadata fetch that hits arXiv API)
            if paper.arxiv_id and not paper.full_text:
                try:
                    pdf_path = _download_arxiv_pdf(paper.arxiv_id)
                    if pdf_path:
                        paper.full_text = extract_text_from_pdf(pdf_path)
                except Exception:
                    pass
                time.sleep(0.5)  # Avoid hammering arXiv servers

            struct = analyze_paper_structure(paper, llm_client,
                                               profile=profile,
                                               domain_name=domain.domain_name)
            if struct:
                paper.structured = struct
                analyzed += 1
            if (i + 1) % 3 == 0:
                print(f"  Analyzed {i + 1}/{len(key_papers)} key papers...")
        print(f"Successfully analyzed {analyzed}/{len(key_papers)} key papers")
    elif key_papers:
        print("Structured analysis disabled — skipping key paper analysis.")
    else:
        print("No key papers to analyze.")

    # ══════════════════════════════════════════════════════════════
    # Phase 5: Technical Route Analysis
    # ══════════════════════════════════════════════════════════════
    routes = None
    comparison = None

    if key_papers and config.V3_ROUTE_ANALYSIS_ENABLED:
        _print_separator("Phase 5: Technical route analysis")
        all_analyzed = [seed_paper] + key_papers if seed_paper.structured else key_papers
        analyzed_papers = [p for p in all_analyzed if p.structured]

        if len(analyzed_papers) >= 2:
            routes = analyze_routes(analyzed_papers, seed_paper, llm_client)
            if routes:
                branches = routes.get("branches", [])
                mainstream = [b for b in branches if b.get("is_mainstream")]
                print(f"Identified {len(branches)} technical branches")
                print(f"  Mainstream: {len(mainstream)}")
                for b in branches:
                    ms_label = " [MAINSTREAM]" if b.get("is_mainstream") else ""
                    print(f"  - {b['name']} ({len(b.get('paper_ids', []))} papers){ms_label}")
            else:
                print("Route analysis unavailable — skipping.")
        else:
            print("Not enough analyzed papers for route analysis.")

        # Comparative analysis
        if routes:
            _print_separator("Phase 5b: Comparative analysis")
            comparison = compare_with_mainstream(seed_paper, routes, llm_client)
            if comparison:
                matrix = comparison.get("comparison_matrix", [])
                print(f"Comparison complete: {len(matrix)} dimensions compared")
            else:
                print("Comparative analysis unavailable.")
    elif config.V3_ROUTE_ANALYSIS_ENABLED:
        print("Route analysis skipped (no key papers available).")
    else:
        print("Route analysis disabled (V3_ROUTE_ANALYSIS_ENABLED=0).")

    # ══════════════════════════════════════════════════════════════
    # Phase 6: Export
    # ══════════════════════════════════════════════════════════════
    _print_separator("Phase 6: Exporting markdown")

    en_path = run_dir / "paper_analysis.md"
    zh_path = run_dir / "paper_analysis.zh.md"

    # Generate English report first
    export_markdown(seed_paper, routes, comparison, all_refs, en_path, lang="en",
                    profile=profile)
    print(f"English report: {en_path}")

    # Chinese: post-hoc LLM translation of the full English report
    en_text = en_path.read_text(encoding="utf-8")
    zh_text = translate_markdown_to_zh(en_text, llm_client)
    zh_path.write_text(zh_text, encoding="utf-8")
    print(f"Chinese report: {zh_path}")

    # Save seed paper data for reference
    import json as _json
    seed_path = run_dir / "seed_paper.json"
    with seed_path.open("w", encoding="utf-8") as f:
        _json.dump(seed_paper.to_dict(), f, ensure_ascii=False, indent=2)

    # Save citation graph
    graph_path = run_dir / "citation_graph.json"
    graph_data = {
        "seed_id": seed_paper.id,
        "papers": {pid: {"title": p.title, "year": p.year,
                          "citation_count": p.citation_count}
                     for pid, p in miner.get_all_papers().items()},
        "references": {pid: [r.to_dict() for r in refs]
                        for pid, refs in miner._ref_map.items()},
    }
    with graph_path.open("w", encoding="utf-8") as f:
        _json.dump(graph_data, f, ensure_ascii=False, indent=2)

    print(f"Seed data:     {seed_path}")
    print(f"Citation graph: {graph_path}")

    # ── Summary ──
    _print_separator("Pipeline complete")
    print(f"Output directory: {run_dir}")
    print(f"Files:")
    print(f"  {en_path.name}     — English report")
    print(f"  {zh_path.name}  — Chinese report")
    print(f"  seed_paper.json     — Seed paper metadata")
    print(f"  citation_graph.json — Citation relationship data")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
