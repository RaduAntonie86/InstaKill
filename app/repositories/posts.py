"""SQLite read boundary for posts used by semantic search."""

from app.db.database import get_connection
from datetime import datetime, timezone


def _epoch(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


class SQLitePostRepository:
    def list_search_documents(self) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id AS post_id, user_id, content FROM post ORDER BY id ASC"
            ).fetchall()
        return [dict(row) for row in rows]

    def list_feed_posts(self) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id AS post_id, user_id, content, topic, created_at FROM post ORDER BY id ASC"
            ).fetchall()
        return [{**dict(row), "created_at": _epoch(row["created_at"])} for row in rows]

    def list_likes(self) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT user_id, post_id, created_at FROM post_like ORDER BY created_at, user_id, post_id"
            ).fetchall()
        return [{**dict(row), "created_at": _epoch(row["created_at"])} for row in rows]

    def user_exists(self, user_id: int) -> bool:
        with get_connection() as conn:
            return conn.execute("SELECT 1 FROM user WHERE id = ?", (user_id,)).fetchone() is not None
