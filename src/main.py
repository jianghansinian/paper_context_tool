from config import (
    OUTPUT_MARKDOWN_PATH,
)
from crawler import fetch_papers
from embedding import build_embedding_client, generate_embeddings
from cluster import cluster_embeddings
from branch_discovery import discover_branches
from citation_graph import build_citation_graph, export_graph
from key_paper import rank_key_papers
from markdown_export import export_markdown
from timeline import build_timeline


def _attach_embeddings(papers, embedding_matrix):
    for idx, paper in enumerate(papers):
        paper["_embedding"] = embedding_matrix[idx]


def main():
    import sys

    if len(sys.argv) < 2:
        print('Usage: python src/main.py "<keyword>"')
        return

    keyword = sys.argv[1].strip()
    if not keyword:
        print("Keyword is empty.")
        return

    papers = fetch_papers(keyword)
    if not papers:
        print("No papers available.")
        return

    client = build_embedding_client()
    embedding_matrix = generate_embeddings(papers, client)
    _attach_embeddings(papers, embedding_matrix)

    labels, _reduced = cluster_embeddings(embedding_matrix)
    branches = discover_branches(papers, labels)

    field_map = {"field": keyword, "branches": []}
    for branch in branches:
        branch_papers = [papers[idx] for idx in branch["paper_indices"]]
        for paper in branch_papers:
            paper["branch"] = branch["branch_name"]

        key_papers = rank_key_papers(branch_papers)
        timeline = build_timeline(branch_papers)
        field_map["branches"].append(
            {
                "branch_id": branch["branch_id"],
                "branch_name": branch["branch_name"],
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

    print(f"Done. Output saved to {OUTPUT_MARKDOWN_PATH}")
    print("Done. Graph saved to output/research_graph.json")


if __name__ == "__main__":
    main()
