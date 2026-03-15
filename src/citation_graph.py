import json
from difflib import SequenceMatcher
from typing import Dict, List

import networkx as nx
from networkx.readwrite import json_graph

from config import OUTPUT_GRAPH_PATH


def _title_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, (a or "").lower(), (b or "").lower()).ratio()


def build_citation_graph(papers: List[Dict]) -> nx.DiGraph:
    graph = nx.DiGraph()

    for idx, paper in enumerate(papers):
        graph.add_node(
            idx,
            title=paper.get("title", ""),
            year=paper.get("year", 0),
            citation_count=paper.get("citation_count", 0),
            link=paper.get("link", ""),
            branch=paper.get("branch", ""),
        )

    # Heuristic inferred citation edges due missing explicit references in MVP input.
    for i in range(len(papers)):
        for j in range(len(papers)):
            if i == j:
                continue
            src = papers[i]
            dst = papers[j]
            if int(src.get("year", 0)) <= int(dst.get("year", 0)):
                continue
            sim = _title_similarity(src.get("title", ""), dst.get("title", ""))
            if sim >= 0.35:
                graph.add_edge(i, j, relation="inferred_citation", similarity=round(sim, 4))

    return graph


def export_graph(graph: nx.DiGraph) -> None:
    OUTPUT_GRAPH_PATH.parent.mkdir(parents=True, exist_ok=True)
    payload = json_graph.node_link_data(graph)
    with OUTPUT_GRAPH_PATH.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
