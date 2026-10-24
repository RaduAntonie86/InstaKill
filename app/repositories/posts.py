"""SQLite read boundary for posts used by semantic search."""

from app.db.database import get_connection


class SQLitePostRepository:
    def list_search_documents(self) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                "SELECT id AS post_id, user_id, content FROM post ORDER BY id ASC"
            ).fetchall()
        return [dict(row) for row in rows]
