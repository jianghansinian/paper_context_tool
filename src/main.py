import json

from config import (
    MIN_PAPER_YEAR,
    BRANCH_ANALYSIS_ENABLED,
    EVOLUTION_ANALYSIS_ENABLED,
    OUTPUT_VALIDATION_ENABLED,
    RELEVANCE_FILTER_ENABLED,
)
import config
from crawler import fetch_papers
from embedding import build_embedding_client, generate_embeddings
from cluster import cluster_embeddings
from branch_discovery import discover_branches
from citation_graph import build_citation_graph, export_graph
from key_paper import rank_key_papers
from llm_namer import build_llm_client, name_branch_with_llm, refine_query
from llm_analyzer import (
    analyze_branch,
    analyze_evolution,
    build_analyzer_client,
    filter_relevant_papers,
    print_validation_report,
    translate_field_map_for_zh,
    validate_output,
)
from markdown_export import export_markdown
from timeline import build_timeline


def _attach_embeddings(papers, embedding_matrix):
    for idx, paper in enumerate(papers):
        paper["_embedding"] = embedding_matrix[idx]


def _save_raw_papers(papers):
    path = config.OUTPUT_PAPERS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(
            [{"title": p.get("title"), "year": p.get("year"),
              "citation_count": p.get("citation_count"), "source": p.get("source"),
              "link": p.get("link")}
             for p in papers],
            f, ensure_ascii=False, indent=2)


def _print_summary(papers, labels, branches, field_map):
    n = len(papers)
    n_clusters = len(set(labels)) - (1 if -1 in set(labels) else 0)
    print(f"\n{'='*50}")
    print(f"Pipeline summary")
    print(f"{'='*50}")
    print(f"  Papers crawled:           {n}")
    print(f"  Clusters discovered:      {n_clusters}  (labels: {sorted(set(labels))})")
    if MIN_PAPER_YEAR:
        pre = sum(1 for p in papers if p.get("year", 0) < MIN_PAPER_YEAR)
        print(f"  Papers < {MIN_PAPER_YEAR}:         {pre} (excluded)")
    for branch in field_map["branches"]:
        source = branch.get("_analysis_source", "heuristic")
        kp_count = len(branch["key_papers"]) if isinstance(branch["key_papers"], list) else 0
        narrative = " [narrative]" if branch.get("narrative") else ""
        shifts = len(branch.get("paradigm_shifts", []))
        forks = len(branch.get("technical_forks", []))
        extra = ""
        if shifts:
            extra += f", {shifts} paradigm shift(s)"
        if forks:
            extra += f", {forks} technical fork(s)"
        print(f"    [{branch['branch_name']}]: {branch['paper_count']} papers "
              f"-> {kp_count} key (source: {source}{extra}){narrative}")
    print(f"{'='*50}")
    print(f"Output files:")
    print(f"  {config.OUTPUT_PAPERS_PATH}    — all crawled papers")
    print(f"  {config.OUTPUT_CLUSTERS_PATH}  — per-cluster paper rankings + score breakdown")
    print(f"  {config.OUTPUT_MARKDOWN_EN_PATH}         — final research map (EN)")
    print(f"  {config.OUTPUT_MARKDOWN_ZH_PATH}         — final research map (ZH)")
    print(f"  {config.OUTPUT_GRAPH_PATH}      — citation graph (node-link JSON)")
    print(f"{'='*50}\n")


def main():
    import sys

    if len(sys.argv) < 2:
        print('Usage: python src/main.py "<keyword>" [seed_paper_title]')
        return

    keyword = sys.argv[1].strip()
    if not keyword:
        print("Keyword is empty.")
        return

    seed_paper_title = sys.argv[2].strip() if len(sys.argv) > 2 else None

    # ── Build clients ──
    embedding_client = build_embedding_client()
    llm_client = build_llm_client()
    analyzer_client = build_analyzer_client()

    # ── Refine query ──
    keyword = refine_query(keyword, llm_client)

    # ── Initialize timestamped output directory ──
    config.init_run_output(keyword)

    # ── Crawl papers ──
    papers = fetch_papers(keyword)
    if not papers:
        print("No papers available.")
        return

    if MIN_PAPER_YEAR:
        before = len(papers)
        papers = [p for p in papers if p.get("year", 0) >= MIN_PAPER_YEAR]
        if len(papers) < before:
            print(f"Year filter (>= {MIN_PAPER_YEAR}): {len(papers)} / {before} papers kept")

    _save_raw_papers(papers)

    # ── LLM Relevance Filtering ──
    if RELEVANCE_FILTER_ENABLED and analyzer_client is not None:
        papers = filter_relevant_papers(papers, keyword, analyzer_client,
                                        min_score="borderline")
        if not papers:
            print("All papers were filtered out by relevance filter. Nothing to analyse.")
            return
    else:
        print("Relevance filter disabled or no LLM client available. Skipping.")

    # ── Embedding + Clustering ──
    embedding_matrix = generate_embeddings(papers, embedding_client)
    _attach_embeddings(papers, embedding_matrix)

    labels, _reduced = cluster_embeddings(embedding_matrix)
    branches = discover_branches(papers, labels)

    # ── Build field_map ──
    field_map = {"field": keyword, "branches": []}
    for branch in branches:
        branch_papers = [papers[idx] for idx in branch["paper_indices"]]
        branch_name = name_branch_with_llm(branch_papers, llm_client) or branch["branch_name"]

        for paper in branch_papers:
            paper["branch"] = branch_name

        # LLM Branch Analysis ── with fallback to heuristic
        if BRANCH_ANALYSIS_ENABLED and analyzer_client is not None:
            branch_analysis = analyze_branch(
                branch_papers,
                {"branch_name": branch_name, "branch_id": branch["branch_id"],
                 "keywords": branch["keywords"]},
                analyzer_client,
            )
        else:
            branch_analysis = None

        if branch_analysis is not None:
            key_papers = branch_analysis.get("key_papers", [])
            timeline = build_timeline(branch_papers)
            field_map["branches"].append(
                {
                    "branch_id": branch["branch_id"],
                    "branch_name": branch_analysis.get("branch_name", branch_name),
                    "keywords": branch["keywords"],
                    "paper_count": len(branch_papers),
                    "key_papers": key_papers,
                    "timeline": timeline,
                    "narrative": branch_analysis.get("narrative", ""),
                    "paradigm_shifts": branch_analysis.get("paradigm_shifts", []),
                    "technical_forks": branch_analysis.get("technical_forks", []),
                    "_analysis_source": "llm",
                }
            )
        else:
            # Heuristic fallback
            key_papers = rank_key_papers(branch_papers)
            timeline = build_timeline(branch_papers)
            field_map["branches"].append(
                {
                    "branch_id": branch["branch_id"],
                    "branch_name": branch_name,
                    "keywords": branch["keywords"],
                    "paper_count": len(branch_papers),
                    "key_papers": key_papers,
                    "timeline": timeline,
                    "_analysis_source": "heuristic",
                }
            )

    if not field_map["branches"]:
        print("No branches discovered.")
        return

    # ── Cross-Branch Evolution Analysis ──
    if EVOLUTION_ANALYSIS_ENABLED and analyzer_client is not None:
        evolution = analyze_evolution(field_map["branches"], keyword, analyzer_client)
        if evolution:
            field_map["overview"] = evolution.get("overview", "")
            field_map["cross_branch_relationships"] = evolution.get("cross_branch_relationships", [])
            field_map["temporal_ordering"] = evolution.get("temporal_ordering", [])

    # ── Export ──
    export_markdown(field_map, config.OUTPUT_MARKDOWN_EN_PATH, lang="en")

    # ── Chinese translation (post-hoc, preserves paper titles/tech terms) ──
    field_map_zh = field_map
    if analyzer_client is not None:
        translated = translate_field_map_for_zh(field_map, analyzer_client)
        if translated is not None:
            field_map_zh = translated

    export_markdown(field_map_zh, config.OUTPUT_MARKDOWN_ZH_PATH, lang="zh")

    # ── Citation graph ──
    graph = build_citation_graph(papers)
    export_graph(graph)

    # ── Validation (non‑blocking) ──
    if OUTPUT_VALIDATION_ENABLED and analyzer_client is not None:
        validation = validate_output(field_map, analyzer_client)
        if validation:
            field_map["_validation"] = validation
            print_validation_report(validation)

    _print_summary(papers, labels, branches, field_map)


if __name__ == "__main__":
    main()
