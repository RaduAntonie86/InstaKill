"""Import a bounded MovieLens ml-latest-small archive into the local SQLite DB.

Usage: python scripts/seed_movielens.py path/to/ml-latest-small.zip
The archive is read in place and is never copied into the repository.
"""
import argparse
import csv
import io
import re
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from app.db.database import get_connection, init_db

MAX_USERS = 200
MAX_POSTS = 2_000


def _member(archive: zipfile.ZipFile, filename: str) -> str:
    matches = [name for name in archive.namelist() if name == filename or name.endswith("/" + filename)]
    if not matches:
        raise ValueError(f"Archive does not contain {filename}")
    return matches[0]


def _read_csv(archive: zipfile.ZipFile, filename: str) -> list[dict[str, str]]:
    with archive.open(_member(archive, filename)) as raw:
        return list(csv.DictReader(io.TextIOWrapper(raw, encoding="utf-8", newline="")))


def _created_at(timestamp: str) -> str:
    # MovieLens timestamps are Unix seconds; store UTC ISO text for consistent API output.
    return datetime.fromtimestamp(int(timestamp), tz=timezone.utc).isoformat(timespec="seconds")


def seed(zip_path: Path) -> tuple[int, int, int]:
    if not zip_path.is_file():
        raise FileNotFoundError(f"MovieLens zip not found: {zip_path}")
    with zipfile.ZipFile(zip_path) as archive:
        movies = _read_csv(archive, "movies.csv")
        ratings = _read_csv(archive, "ratings.csv")

    # Stable numeric ordering makes the exact selected cohort independent of CSV row order.
    movies.sort(key=lambda row: int(row["movieId"]))
    ratings.sort(key=lambda row: (int(row["userId"]), int(row["movieId"]), int(row["timestamp"])))
    users = sorted({int(row["userId"]) for row in ratings})[:MAX_USERS]
    if not movies or not users:
        raise ValueError("MovieLens archive must include movie and rating rows")
    user_set = set(users)
    selected_movies = movies[:MAX_POSTS]

    # The small MovieLens archive has no free-text review column. Treat its rating
    # events as review metadata: summarize the community rating and use the first
    # recorded rating as the post's publication/first-seen time.
    rating_stats: dict[int, dict[str, float | int]] = {}
    for row in ratings:
        movie_id = int(row["movieId"])
        stats = rating_stats.setdefault(movie_id, {"sum": 0.0, "count": 0, "first": int(row["timestamp"])})
        stats["sum"] = float(stats["sum"]) + float(row["rating"])
        stats["count"] = int(stats["count"]) + 1
        stats["first"] = min(int(stats["first"]), int(row["timestamp"]))

    init_db()
    with get_connection() as conn:
        user_ids: dict[int, int] = {}
        for source_user_id in users:
            username = f"movielens_user_{source_user_id}"
            conn.execute("INSERT OR IGNORE INTO user(username) VALUES (?)", (username,))
            user_ids[source_user_id] = conn.execute(
                "SELECT id FROM user WHERE username = ?", (username,)
            ).fetchone()["id"]
        catalog_owner_id = user_ids[users[0]] if users else None

        post_ids: dict[int, int] = {}
        for movie in selected_movies:
            movie_id = int(movie["movieId"])
            stats = rating_stats.get(movie_id)
            if stats:
                average = float(stats["sum"]) / int(stats["count"])
                review = f"Movie review: {movie['title']}\nMovieLens community rating: {average:.1f}/5 from {stats['count']} ratings."
                created_at = _created_at(str(stats["first"]))
            else:
                review = f"Movie review: {movie['title']}\nGenres: {movie['genres'].replace('|', ', ')}."
                year = re.search(r"\((\d{4})\)\s*$", movie["title"])
                created_at = f"{year.group(1)}-01-01T00:00:00+00:00" if year else "1970-01-01T00:00:00+00:00"
            content = f"{review}\n[MovieLens movieId={movie_id}]"
            topic = movie["genres"].replace("|", ", ") or "(no genres listed)"
            existing = conn.execute(
                "SELECT id FROM post WHERE instr(content, ?) > 0 ORDER BY id LIMIT 1",
                (f"[MovieLens movieId={movie_id}]",),
            ).fetchone()
            if existing:
                post_id = existing["id"]
                conn.execute("UPDATE post SET user_id = ?, content = ?, topic = ?, created_at = ? WHERE id = ?",
                             (catalog_owner_id, content, topic, created_at, post_id))
            else:
                cursor = conn.execute(
                    "INSERT INTO post(user_id, content, topic, created_at) VALUES (?, ?, ?, ?)",
                    (catalog_owner_id, content, topic, created_at),
                )
                post_id = cursor.lastrowid
            post_ids[movie_id] = post_id

        likes_added = 0
        for rating in ratings:
            source_user_id = int(rating["userId"])
            movie_id = int(rating["movieId"])
            if source_user_id not in user_set or movie_id not in post_ids or float(rating["rating"]) < 4.0:
                continue
            cursor = conn.execute(
                "INSERT OR IGNORE INTO post_like(user_id, post_id, created_at) VALUES (?, ?, ?)",
                (user_ids[source_user_id], post_ids[movie_id], _created_at(rating["timestamp"])),
            )
            likes_added += cursor.rowcount
        conn.commit()

    return len(users), len(selected_movies), likes_added


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("zip", type=Path, help="path to the downloaded ml-latest-small zip")
    args = parser.parse_args()
    users, posts, likes = seed(args.zip)
    print(f"MovieLens import complete: {users} source users, {posts} movie posts, {likes} new likes.")


if __name__ == "__main__":
    main()
