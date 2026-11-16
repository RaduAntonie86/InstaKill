"""Deterministic three-signal feed ranking, based on the Week 4 lab."""

from collections import Counter, defaultdict
from heapq import nsmallest

DAY = 86_400
HALF_LIFE_DAYS = 3 * 365
POPULARITY_DAYS = 365
DEFAULT_WEIGHTS = {"freshness": 1.0, "affinity": 1.0, "popularity": 1.0}


def build_signals(history: list[dict], posts: list[dict], now: float) -> dict:
    """Summarize prior likes once: liked posts, topic preferences, and popularity."""
    post_by_id = {post["post_id"]: post for post in posts}
    movie_post_ids = {
        post["post_id"] for post in posts
        if "[MovieLens movieId=" in str(post.get("content", ""))
    }
    movie_horizon = max(
        (like["created_at"] for like in history
         if like["created_at"] < now and like["post_id"] in movie_post_ids),
        default=now,
    )
    liked = defaultdict(set)
    topic_counts = defaultdict(Counter)
    recent_likes = Counter()
    for like in history:
        if like["created_at"] >= now:
            continue
        post_id = like["post_id"]
        post = post_by_id.get(post_id)
        if post is None:
            continue
        liked[like["user_id"]].add(post_id)
        topic_counts[like["user_id"]][post["topic"]] += 1
        popularity_now = movie_horizon if post_id in movie_post_ids else now
        if like["created_at"] >= popularity_now - POPULARITY_DAYS * DAY:
            recent_likes[post_id] += 1
    return {
        "liked": liked,
        "topic_counts": topic_counts,
        "recent_likes": recent_likes,
        "max_recent": max(recent_likes.values(), default=1),
    }


def item_signals(user_id: int, post: dict, signals: dict, now: float) -> dict[str, float]:
    age_days = max(0.0, (now - post["created_at"]) / DAY)
    freshness = 0.5 ** (age_days / HALF_LIFE_DAYS)
    counts = signals["topic_counts"].get(user_id)
    total = sum(counts.values()) if counts else 0
    affinity = counts[post["topic"]] / total if total else 0.0
    popularity = signals["recent_likes"][post["post_id"]] / signals["max_recent"]
    return {"freshness": freshness, "affinity": affinity, "popularity": popularity}


def recommend(user_id: int, posts: list[dict], history: list[dict], now: float,
              limit: int = 10, weights: dict[str, float] | None = None) -> list[dict]:
    """Rank unseen, already-published posts with O(C log k) top-k selection."""
    if limit < 1:
        raise ValueError("limit must be positive")
    weights = DEFAULT_WEIGHTS if weights is None else weights
    if set(weights) != set(DEFAULT_WEIGHTS) or any(value < 0 for value in weights.values()) or not any(weights.values()):
        raise ValueError("weights must contain non-negative freshness, affinity, and popularity values, at least one positive")
    signals = build_signals(history, posts, now)
    candidates = []
    for post in posts:
        if post["created_at"] >= now or post["post_id"] in signals["liked"].get(user_id, set()):
            continue
        parts = item_signals(user_id, post, signals, now)
        score = sum(weights[name] * value for name, value in parts.items())
        candidates.append((score, post["post_id"], post, parts))
    best = nsmallest(limit, candidates, key=lambda row: (-row[0], row[1]))
    return [{**post, "score": float(score), "signals": parts} for score, _, post, parts in best]
