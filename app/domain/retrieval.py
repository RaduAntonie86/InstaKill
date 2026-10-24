"""Pure NumPy cosine retrieval and deterministic top-k selection."""

from collections.abc import Sequence

import numpy as np


def top_k_by_cosine(
    query_vector: Sequence[float] | np.ndarray,
    post_vectors: Sequence[Sequence[float]] | np.ndarray,
    posts: Sequence[dict],
    limit: int,
) -> list[dict]:
    """Return posts ordered by cosine score, then ascending post_id.

    This function has no model, database, HTTP, or filesystem dependencies.
    """
    if limit < 1:
        raise ValueError("limit must be positive")
    if len(posts) == 0:
        return []

    query = np.asarray(query_vector, dtype=np.float64).reshape(-1)
    matrix = np.asarray(post_vectors, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != len(posts):
        raise ValueError("post_vectors must have one row per post")
    if matrix.shape[1] != query.size:
        raise ValueError("query and post vectors must have matching dimensions")

    query_norm = np.linalg.norm(query)
    post_norms = np.linalg.norm(matrix, axis=1)
    denominators = post_norms * query_norm
    scores = np.divide(
        matrix @ query,
        denominators,
        out=np.zeros(len(posts), dtype=np.float64),
        where=denominators != 0,
    )
    # lexsort uses the last key as the primary key.
    order = np.lexsort((np.asarray([p["post_id"] for p in posts]), -scores))[:limit]
    return [{**posts[i], "score": float(scores[i])} for i in order]
