import math
from typing import Tuple

import numpy as np


def _estimate_k(n_samples: int) -> int:
    return max(2, min(8, int(math.sqrt(max(n_samples, 2)))))


def cluster_embeddings(embeddings: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if embeddings.shape[0] == 0:
        return np.array([], dtype=int), np.empty((0, 0), dtype=float)
    if embeddings.shape[0] < 3:
        return np.zeros(embeddings.shape[0], dtype=int), embeddings

    try:
        import umap
        import hdbscan

        reducer = umap.UMAP(
            n_neighbors=min(15, max(2, embeddings.shape[0] - 1)),
            n_components=5,
            metric="cosine",
            random_state=42,
        )
        reduced = reducer.fit_transform(embeddings)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=max(2, embeddings.shape[0] // 12),
            metric="euclidean",
            prediction_data=False,
        )
        labels = clusterer.fit_predict(reduced)
        if np.all(labels == -1):
            labels = np.zeros(embeddings.shape[0], dtype=int)
        return labels.astype(int), reduced
    except Exception:  # noqa: BLE001
        from sklearn.cluster import KMeans
        from sklearn.decomposition import PCA

        reduced = PCA(n_components=min(5, embeddings.shape[1], embeddings.shape[0])).fit_transform(
            embeddings
        )
        k = _estimate_k(embeddings.shape[0])
        labels = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(reduced)
        return labels.astype(int), reduced
