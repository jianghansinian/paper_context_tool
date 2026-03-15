import math
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import CENTRALITY_WEIGHT, CITATION_WEIGHT, TOP_K_PAPERS


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    v_min = float(values.min())
    v_max = float(values.max())
    if math.isclose(v_min, v_max):
        return np.ones_like(values, dtype=float)
    return (values - v_min) / (v_max - v_min)


def rank_key_papers(branch_papers: List[Dict], top_k: int = TOP_K_PAPERS) -> List[Dict]:
    if not branch_papers:
        return []

    citations = np.array([max(0, int(paper.get("citation_count", 0))) for paper in branch_papers], dtype=float)
    citation_score = _normalize(np.log1p(citations))

    embeddings = np.array([paper["_embedding"] for paper in branch_papers], dtype=float)
    centroid = embeddings.mean(axis=0, keepdims=True)
    centrality = cosine_similarity(embeddings, centroid).reshape(-1)
    centrality_score = _normalize(centrality)

    combined_score = CITATION_WEIGHT * citation_score + CENTRALITY_WEIGHT * centrality_score

    ranked = sorted(
        zip(branch_papers, combined_score),
        key=lambda item: item[1],
        reverse=True,
    )

    output = []
    for paper, score in ranked[:top_k]:
        paper_copy = {k: v for k, v in paper.items() if not k.startswith("_")}
        paper_copy["score"] = round(float(score), 4)
        output.append(paper_copy)
    return output
