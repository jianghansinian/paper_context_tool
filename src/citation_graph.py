import json
import re
from difflib import SequenceMatcher
from typing import Dict, List, Set

import networkx as nx
from networkx.readwrite import json_graph

import config


def _extract_openalex_id(openalex_url: str) -> str:
    m = re.search(r"openalex\.org/(W\d+)", str(openalex_url))
    return m.group(1) if m else ""


def _openalex_citation_edges(papers: List[Dict]) -> Set:
    oa_id_to_idx = {}
    for idx, paper in enumerate(papers):
        oa_id = _extract_openalex_id(paper.get("openalex_id", ""))
        if oa_id:
            oa_id_to_idx[oa_id] = idx

    if len(oa_id_to_idx) < 2:
        return set()

    edges = set()
    for src_idx, paper in enumerate(papers):
        refs = paper.get("referenced_works", [])
        if not refs:
            continue
        for ref_url in refs:
            ref_id = _extract_openalex_id(ref_url)
            dst_idx = oa_id_to_idx.get(ref_id)
            if dst_idx is not None and dst_idx != src_idx:
                edges.add((src_idx, dst_idx))

    return edges


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

    real_edges = _openalex_citation_edges(papers)

    if real_edges:
        for src_idx, dst_idx in real_edges:
            graph.add_edge(src_idx, dst_idx, relation="citation")
    else:
        # Fallback: heuristic inferred citation edges.
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
                    graph.add_edge(
                        i, j, relation="inferred_citation", similarity=round(sim, 4)
                    )

    return graph


def export_graph(graph: nx.DiGraph) -> None:
    path = config.OUTPUT_GRAPH_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json_graph.node_link_data(graph, edges="links")
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
