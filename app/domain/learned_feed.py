"""Vectorized six-signal feed model exported by the Week 5 notebook."""
from pathlib import Path

import numpy as np

FEATURES = ("freshness", "affinity", "popularity", "activity", "lifetime", "age")
DAY = 86_400.0


def load_model(path: Path) -> dict[str, np.ndarray]:
    """Load and validate the notebook artifact. Call once during application startup."""
    with np.load(path, allow_pickle=False) as archive:
        model = {name: archive[name].copy() for name in archive.files}
    if tuple(model.get("features", ())) != FEATURES:
        raise ValueError("like_model.npz feature order must be freshness, affinity, popularity, activity, lifetime, age")
    if model["mu"].shape != (6,) or model["sd"].shape != (6,):
        raise ValueError("like_model.npz must contain six-element mu and sd arrays")
    hidden = model["W1"].shape[1] if model["W1"].ndim == 2 else 0
    if model["W1"].shape != (6, hidden) or model["b1"].shape != (hidden,):
        raise ValueError("like_model.npz has inconsistent first-layer shapes")
    if model["W2"].shape != (hidden, 1) or model["b2"].shape != (1,):
        raise ValueError("like_model.npz has inconsistent output-layer shapes")
    if any(name not in model for name in ("half_life_days", "popularity_days", "activity_cap", "age_cap_days")):
        raise ValueError("like_model.npz is missing signal constants")
    return model


def predict(model: dict[str, np.ndarray], features: np.ndarray) -> np.ndarray:
    """Pure NumPy prediction for an (n, 6) feature matrix; returns P(like) per row."""
    x = (np.asarray(features) - model["mu"]) / model["sd"]
    hidden = np.tanh(x @ model["W1"] + model["b1"])
    z = (hidden @ model["W2"] + model["b2"])[:, 0]
    return 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))


def _genre_matrix(topics: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Expand topic strings to genre indicators, iterating only over distinct labels."""
    topic_values = np.asarray([topic or "general" for topic in topics], dtype=str)
    unique_topics, inverse = np.unique(topic_values, return_inverse=True)
    labels_by_topic = [tuple(part.strip() for part in value.replace("|", ",").split(",") if part.strip()) or ("general",)
                       for value in unique_topics]
    labels = sorted({label for group in labels_by_topic for label in group})
    label_index = {label: index for index, label in enumerate(labels)}
    distinct = np.zeros((len(unique_topics), len(labels)), dtype=np.float64)
    for topic_index, group in enumerate(labels_by_topic):
        distinct[topic_index, [label_index[label] for label in group]] = 1.0
    return distinct[inverse], np.maximum(distinct.sum(axis=1)[inverse], 1.0)


def build_features(user_id: int, posts: list[dict], history: list[dict], now: float,
                   model: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Build the notebook's six signals for all posts at once; returns (features, liked mask).

    The bundled MovieLens corpus is a historical snapshot. Its one-year popularity
    window is evaluated at the corpus's last rating timestamp; live posts use `now`.
    """
    post_ids = np.asarray([post["post_id"] for post in posts], dtype=np.int64)
    created = np.asarray([post["created_at"] for post in posts], dtype=np.float64)
    topics = [post.get("topic", "general") for post in posts]
    genres, n_genres = _genre_matrix(topics)

    if history:
        h_user = np.asarray([like["user_id"] for like in history], dtype=np.int64)
        h_post = np.asarray([like["post_id"] for like in history], dtype=np.int64)
        h_time = np.asarray([like["created_at"] for like in history], dtype=np.float64)
        valid = h_time < now
        h_user, h_post, h_time = h_user[valid], h_post[valid], h_time[valid]
        positions = np.searchsorted(post_ids, h_post)
        in_catalogue = positions < len(post_ids)
        safe_positions = np.minimum(positions, max(len(post_ids) - 1, 0))
        if len(post_ids):
            in_catalogue &= post_ids[safe_positions] == h_post
        else:
            in_catalogue[:] = False
        h_user, h_time, positions = h_user[in_catalogue], h_time[in_catalogue], positions[in_catalogue]
    else:
        h_user = np.empty(0, dtype=np.int64)
        h_time = np.empty(0, dtype=np.float64)
        positions = np.empty(0, dtype=np.int64)

    movie_posts = np.fromiter(
        ("[MovieLens movieId=" in str(post.get("content", "")) for post in posts),
        dtype=bool,
        count=len(posts),
    )
    movie_likes = movie_posts[positions] if len(positions) else np.zeros(0, dtype=bool)
    movie_horizon = float(h_time[movie_likes].max()) if movie_likes.any() else now

    liked_mask = np.zeros(len(posts), dtype=bool)
    user_rows = h_user == user_id
    liked_mask[positions[user_rows]] = True
    user_positions = positions[user_rows]
    user_count = len(user_positions)

    genre_counts = genres[user_positions].sum(axis=0) if user_count else np.zeros(genres.shape[1])
    genre_total = genre_counts.sum()
    affinity = (genres @ genre_counts) / (genre_total * n_genres) if genre_total else np.zeros(len(posts))

    window = float(model["popularity_days"]) * DAY
    recent_mask = ((h_time >= now - window) & ~movie_likes) | ((h_time >= movie_horizon - window) & movie_likes)
    recent_counts = np.bincount(positions[recent_mask], minlength=len(posts)).astype(np.float64)
    lifetime_counts = np.bincount(positions, minlength=len(posts)).astype(np.float64)
    recent_max = max(float(recent_counts.max(initial=0.0)), 1.0)
    lifetime_max = max(float(lifetime_counts.max(initial=0.0)), 1.0)

    # Score the static MovieLens corpus at its own latest observed event, just as
    # the notebook scores candidates at a historical interaction moment. This
    # prevents its 25-year cap from flattening nearly every movie to age=1.
    signal_now = np.where(movie_posts, movie_horizon, now)
    age_days = (signal_now - created) / DAY
    freshness = 0.5 ** (age_days / float(model["half_life_days"]))
    popularity = recent_counts / recent_max
    activity = np.full(len(posts), min(1.0, np.log1p(user_count) / np.log1p(float(model["activity_cap"]))))
    lifetime = np.log1p(lifetime_counts) / np.log1p(lifetime_max)
    age = np.minimum(1.0, age_days / float(model["age_cap_days"]))
    features = np.column_stack((freshness, affinity, popularity, activity, lifetime, age))
    return features, liked_mask
