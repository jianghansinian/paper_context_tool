import json

from config import (
    MIN_PAPER_YEAR,
    OUTPUT_CLUSTERS_PATH,
    OUTPUT_GRAPH_PATH,
    OUTPUT_MARKDOWN_PATH,
    OUTPUT_PAPERS_PATH,
)
from crawler import fetch_papers
from embedding import build_embedding_client, generate_embeddings
from cluster import cluster_embeddings
from branch_discovery import discover_branches
from citation_graph import build_citation_graph, export_graph
from key_paper import rank_key_papers
from llm_namer import build_llm_client, name_branch_with_llm, refine_query
from markdown_export import export_markdown
from timeline import build_timeline


def _attach_embeddings(papers, embedding_matrix):
    for idx, paper in enumerate(papers):
        paper["_embedding"] = embedding_matrix[idx]


def _save_raw_papers(papers):
    OUTPUT_PAPERS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PAPERS_PATH.open("w", encoding="utf-8") as f:
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
    total_kp = sum(len(b["key_papers"]) for b in field_map["branches"])
    print(f"  Key papers selected:      {total_kp}")
    for branch in field_map["branches"]:
        print(f"    [{branch['branch_name']}]: {branch['paper_count']} papers "
              f"-> {len(branch['key_papers'])} key")
    print(f"{'='*50}")
    print(f"Output files:")
    print(f"  {OUTPUT_PAPERS_PATH}    — all crawled papers")
    print(f"  {OUTPUT_CLUSTERS_PATH}  — per-cluster paper rankings + score breakdown")
    print(f"  {OUTPUT_MARKDOWN_PATH}   — final research map")
    print(f"  {OUTPUT_GRAPH_PATH}      — citation graph (node-link JSON)")
    print(f"{'='*50}\n")


def main():
    import sys

    if len(sys.argv) < 2:
        print('Usage: python src/main.py "<keyword>"')
        return

    keyword = sys.argv[1].strip()
    if not keyword:
        print("Keyword is empty.")
        return

    llm_client = build_llm_client()
    keyword = refine_query(keyword, llm_client)

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

    client = build_embedding_client()
    embedding_matrix = generate_embeddings(papers, client)
    _attach_embeddings(papers, embedding_matrix)

    labels, _reduced = cluster_embeddings(embedding_matrix)
    branches = discover_branches(papers, labels)

    field_map = {"field": keyword, "branches": []}
    for branch in branches:
        branch_papers = [papers[idx] for idx in branch["paper_indices"]]
        branch_name = branch["branch_name"]

        llm_name = name_branch_with_llm(branch_papers, llm_client)
        if llm_name:
            branch_name = llm_name

        for paper in branch_papers:
            paper["branch"] = branch_name

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
            }
        )

    if not field_map["branches"]:
        print("No branches discovered.")
        return

    export_markdown(field_map, OUTPUT_MARKDOWN_PATH)
    graph = build_citation_graph(papers)
    export_graph(graph)

    _print_summary(papers, labels, branches, field_map)


if __name__ == "__main__":
    main()