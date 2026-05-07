import json
import math
from typing import Dict, List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from config import (
    CENTRALITY_WEIGHT,
    CITATION_WEIGHT,
    OUTPUT_CLUSTERS_PATH,
    RECENCY_WEIGHT,
    TOP_K_PAPERS,
)


def _normalize(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    v_min = float(values.min())
    v_max = float(values.max())
    if math.isclose(v_min, v_max):
        return np.ones_like(values, dtype=float)
    return (values - v_min) / (v_max - v_min)


def rank_key_papers(
    branch_papers: List[Dict],
    top_k: int = TOP_K_PAPERS,
) -> List[Dict]:
    if not branch_papers:
        return []

    citations = np.array(
        [max(0, int(paper.get("citation_count", 0))) for paper in branch_papers],
        dtype=float,
    )
    citation_score = _normalize(np.log1p(citations))

    embeddings = np.array(
        [paper["_embedding"] for paper in branch_papers], dtype=float
    )
    centroid = embeddings.mean(axis=0, keepdims=True)
    centrality = cosine_similarity(embeddings, centroid).reshape(-1)
    centrality_score = _normalize(centrality)

    years = np.array(
        [int(paper.get("year", 0)) for paper in branch_papers], dtype=float
    )
    year_score = _normalize(years)

    combined_score = (
        CITATION_WEIGHT * citation_score
        + CENTRALITY_WEIGHT * centrality_score
        + RECENCY_WEIGHT * year_score
    )

    ranked = sorted(
        zip(branch_papers, combined_score, citation_score, centrality_score, year_score),
        key=lambda item: item[1],
        reverse=True,
    )

    output = []
    all_ranked = []
    for paper, score, cit, cent, yr in ranked:
        paper_copy = {k: v for k, v in paper.items() if not k.startswith("_")}
        paper_copy["score"] = round(float(score), 4)
        paper_copy["score_breakdown"] = {
            "citation": round(float(cit), 4),
            "centrality": round(float(cent), 4),
            "recency": round(float(yr), 4),
        }
        all_ranked.append(paper_copy)
        if len(output) < top_k:
            output.append(paper_copy)

    _save_clusters_data(all_ranked, len(output))

    return output


def _save_clusters_data(papers_ranked: List[Dict], top_k_count: int) -> None:
    """Persist per-cluster paper rankings as intermediate output."""
    OUTPUT_CLUSTERS_PATH.parent.mkdir(parents=True, exist_ok=True)

    existing = []
    if OUTPUT_CLUSTERS_PATH.exists():
        try:
            with OUTPUT_CLUSTERS_PATH.open(encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:  # noqa: BLE001
            pass

    existing.append(
        {
            "total_papers": len(papers_ranked),
            "top_k": top_k_count,
            "papers": papers_ranked,
            "date": None,  # caller-side timestamp
        }
    )

    with OUTPUT_CLUSTERS_PATH.open("w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)