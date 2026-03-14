from typing import Optional

import numpy as np
from openai import APIConnectionError, APITimeoutError, NotFoundError, OpenAI
from sklearn.feature_extraction.text import HashingVectorizer

from config import (
    EMBEDDING_API_KEY,
    EMBEDDING_BASE_URL,
    EMBEDDING_MODEL,
    ENABLE_LOCAL_EMBEDDING_FALLBACK,
)


_LOCAL_VECTORIZER = HashingVectorizer(
    n_features=1024,
    alternate_sign=False,
    norm="l2",
    ngram_range=(1, 2),
)


def _local_embedding(text: str) -> np.ndarray:
    vec = _LOCAL_VECTORIZER.transform([text]).toarray()[0]
    return vec.astype(float)


def build_embedding_client(api_key: Optional[str] = None) -> Optional[OpenAI]:
    """Create an embeddings client via OpenAI-compatible API."""
    key = api_key or EMBEDDING_API_KEY
    if not key:
        if ENABLE_LOCAL_EMBEDDING_FALLBACK:
            print("No embedding API key detected, using local embedding fallback.")
            return None
        raise ValueError(
            "Embedding API key is missing. Set EMBEDDING_API_KEY (or OPENAI_API_KEY)."
        )
    return OpenAI(
        api_key=key,
        base_url=EMBEDDING_BASE_URL,
        timeout=5.0,
        max_retries=0,
    )


def get_embedding(text: str, client: Optional[OpenAI]) -> np.ndarray:
    """Generate embedding for the input text."""
    if client is None:
        return _local_embedding(text)

    try:
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=text)
    except (APIConnectionError, APITimeoutError):
        if ENABLE_LOCAL_EMBEDDING_FALLBACK:
            print("Embedding API unreachable, falling back to local embedding.")
            return _local_embedding(text)
        raise
    except NotFoundError as exc:
        if ENABLE_LOCAL_EMBEDDING_FALLBACK:
            print("Embedding endpoint/model not found, falling back to local embedding.")
            return _local_embedding(text)
        raise ValueError(
            "Embeddings endpoint/model not found (404). "
            "Check EMBEDDING_BASE_URL and EMBEDDING_MODEL. "
            "Example: EMBEDDING_BASE_URL=https://api.openai.com/v1, "
            "EMBEDDING_MODEL=text-embedding-3-small."
        ) from exc
    return np.array(response.data[0].embedding, dtype=float)
