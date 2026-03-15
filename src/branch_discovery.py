from collections import defaultdict
from typing import Dict, List

from sklearn.feature_extraction.text import CountVectorizer


def _top_terms(texts: List[str], top_k: int = 3) -> List[str]:
    docs = [text for text in texts if text.strip()]
    if not docs:
        return ["general"]
    vectorizer = CountVectorizer(stop_words="english", ngram_range=(1, 2), max_features=2000)
    matrix = vectorizer.fit_transform(docs)
    weights = matrix.sum(axis=0).A1
    vocab = vectorizer.get_feature_names_out()
    ranked = sorted(zip(vocab, weights), key=lambda item: item[1], reverse=True)
    return [term for term, _ in ranked[:top_k]] or ["general"]


def discover_branches(papers: List[Dict], labels) -> List[Dict]:
    groups = defaultdict(list)
    for idx, label in enumerate(labels):
        label_id = int(label)
        groups[label_id].append(idx)

    branches = []
    branch_order = 1
    for label_id, indices in sorted(groups.items(), key=lambda item: item[0]):
        cluster_papers = [papers[i] for i in indices]
        texts = [f"{paper.get('title', '')} {paper.get('abstract', '')}" for paper in cluster_papers]
        keywords = _top_terms(texts, top_k=2)
        name = f"Branch {branch_order}: {' / '.join(word.title() for word in keywords)}"
        branches.append(
            {
                "branch_id": label_id,
                "branch_name": name,
                "paper_indices": indices,
                "keywords": keywords,
            }
        )
        branch_order += 1
    return branches
