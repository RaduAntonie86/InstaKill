"""Shared feed ranking service used by the API and offline evaluation."""

import numpy as np

from app.domain.learned_feed import FEATURES, build_features, predict
from app.domain.recommendations import DEFAULT_WEIGHTS, build_signals, item_signals


def rank_feed(user_id: int, posts: list[dict], history: list[dict], now: float,
              limit: int = 10, strategy: str = "for_you",
              model: dict[str, np.ndarray] | None = None,
              weights: dict[str, float] | None = None) -> list[dict]:
    """Rank a fixed catalogue at an explicit time for reproducible replay."""
    if limit < 1:
        raise ValueError("limit must be positive")
    prior_likes = {(like["user_id"], like["post_id"]) for like in history if like["created_at"] < now}
    if strategy in ("learned", "for_you"):
        if model is None:
            raise ValueError(f"{strategy} strategy requires the shared signal model constants")
        features, liked = build_features(user_id, posts, history, now, model)
        if strategy == "learned":
            chosen_weights = DEFAULT_WEIGHTS if weights is None else weights
            if (set(chosen_weights) != set(DEFAULT_WEIGHTS)
                    or any(value < 0 for value in chosen_weights.values())
                    or not any(chosen_weights.values())):
                raise ValueError("weights must contain non-negative freshness, affinity, and popularity values, at least one positive")
            scores = features[:, :3] @ np.asarray([chosen_weights[name] for name in ("freshness", "affinity", "popularity")])
        else:
            scores = predict(model, features)
    elif strategy == "chronological":
        scores = np.asarray([post["created_at"] for post in posts], dtype=np.float64)
        features = None
        liked = np.fromiter(((user_id, post["post_id"]) in prior_likes for post in posts), dtype=bool, count=len(posts))
    elif strategy == "popularity":
        shared = build_signals(history, posts, now)
        scores = np.asarray([item_signals(user_id, post, shared, now)["popularity"] for post in posts], dtype=np.float64)
        features = None
        liked = np.fromiter(((user_id, post["post_id"]) in prior_likes for post in posts), dtype=bool, count=len(posts))
    else:
        raise ValueError(f"unknown feed strategy: {strategy}")

    ids = np.asarray([post["post_id"] for post in posts], dtype=np.int64)
    created = np.asarray([post["created_at"] for post in posts], dtype=np.float64)
    candidates = np.flatnonzero((created < now) & ~liked)
    order = np.lexsort((ids[candidates], -scores[candidates]))[:limit]
    result = []
    for i in candidates[order]:
        item = {**posts[i], "score": float(scores[i])}
        if strategy == "for_you":
            item["signals"] = {name: float(value) for name, value in zip(FEATURES, features[i])}
        elif strategy == "learned":
            item["signals"] = {name: float(value) for name, value in zip(FEATURES[:3], features[i, :3])}
        result.append(item)
    return result
