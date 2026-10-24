"""Pure NumPy retrieval functions mirrored from the Week 3 lab contract."""

from collections.abc import Sequence

import numpy as np


def cosine_similarity(query_vector, corpus_vectors) -> np.ndarray:
    """Return one finite cosine score for each corpus row; zero norms score 0."""
    query = np.asarray(query_vector, dtype=np.float64).reshape(-1)
    corpus = np.asarray(corpus_vectors, dtype=np.float64)
    if corpus.ndim != 2 or corpus.shape[1] != query.size:
        raise ValueError("query and corpus vectors must have matching dimensions")
    denominators = np.linalg.norm(corpus, axis=1) * np.linalg.norm(query)
    return np.divide(
        corpus @ query,
        denominators,
        out=np.zeros(corpus.shape[0], dtype=np.float64),
        where=denominators != 0,
    )


def stable_top_k(scores, post_ids, k: int) -> np.ndarray:
    """Return indices ordered by score descending, then post_id ascending."""
    scores = np.asarray(scores)
    post_ids = np.asarray(post_ids)
    if scores.ndim != 1 or post_ids.ndim != 1 or len(scores) != len(post_ids):
        raise ValueError("scores and post_ids must be matching one-dimensional arrays")
    if k < 1:
        raise ValueError("k must be positive")
    return np.lexsort((post_ids, -scores))[:k]


def semantic_search(query_embedding, posts: Sequence[dict], corpus_embeddings, k: int = 5) -> list[dict]:
    """Return the top posts and numeric cosine score, without loading a model or I/O."""
    if len(posts) == 0:
        return []
    corpus = np.asarray(corpus_embeddings)
    if corpus.ndim != 2 or corpus.shape[0] != len(posts):
        raise ValueError("corpus_embeddings must have one row per post")
    scores = cosine_similarity(query_embedding, corpus)
    indices = stable_top_k(scores, [post["post_id"] for post in posts], k)
    return [{**posts[i], "score": float(scores[i])} for i in indices]
