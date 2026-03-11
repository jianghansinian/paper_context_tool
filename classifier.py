import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from embedding import get_embedding


def _seed_text(branch):
    return " ".join(branch.get("seed_papers", []))


def build_branch_embeddings(branches, client):
    """Precompute branch embeddings from seed papers."""
    embeddings = []
    for branch in branches:
        seed = _seed_text(branch)
        if not seed:
            embeddings.append(np.zeros(1))
            continue
        embeddings.append(get_embedding(seed, client))
    return embeddings


def classify_paper(paper, branches, branch_embeddings, client):
    """Assign one paper to the most similar branch."""

    paper_text = paper["title"] + " " + paper["abstract"]
    paper_emb = get_embedding(paper_text, client)

    best_branch = None
    best_score = -1

    for branch, seed_emb in zip(branches, branch_embeddings):
        if seed_emb.shape[0] == 1 and seed_emb[0] == 0:
            continue

        score = cosine_similarity(
            [paper_emb], [seed_emb]
        )[0][0]

        if score > best_score:
            best_score = score
            best_branch = branch

    return best_branch