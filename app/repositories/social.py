"""Persistence operations for users, posts, follows, and likes."""
from app.db.database import get_connection


class SQLiteSocialRepository:
    def get_user(self, user_id: int):
        with get_connection() as conn:
            row = conn.execute("SELECT id, username, created_at FROM user WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if row else None

    def list_users(self):
        with get_connection() as conn:
            return [dict(r) for r in conn.execute("SELECT id, username, created_at FROM user ORDER BY id")]

    def create_user(self, username: str):
        with get_connection() as conn:
            cur = conn.execute("INSERT INTO user(username) VALUES (?)", (username,))
            row = conn.execute("SELECT id, username, created_at FROM user WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)

    def patch_user(self, user_id: int, username: str):
        with get_connection() as conn:
            cur = conn.execute("UPDATE user SET username = ? WHERE id = ?", (username, user_id))
            row = conn.execute("SELECT id, username, created_at FROM user WHERE id = ?", (user_id,)).fetchone()
        return dict(row) if cur.rowcount else None

    def delete_user(self, user_id: int) -> bool:
        with get_connection() as conn:
            conn.execute("DELETE FROM post_like WHERE user_id = ? OR post_id IN (SELECT id FROM post WHERE user_id = ?)", (user_id, user_id))
            conn.execute("DELETE FROM follow WHERE follower_id = ? OR followed_id = ?", (user_id, user_id))
            conn.execute("DELETE FROM post WHERE user_id = ?", (user_id,))
            cur = conn.execute("DELETE FROM user WHERE id = ?", (user_id,))
            return cur.rowcount > 0

    def get_post(self, post_id: int):
        with get_connection() as conn:
            row = conn.execute("SELECT id, user_id, content, created_at FROM post WHERE id = ?", (post_id,)).fetchone()
        return dict(row) if row else None

    def list_posts(self):
        with get_connection() as conn:
            return [dict(r) for r in conn.execute("SELECT id, user_id, content, created_at FROM post ORDER BY id")]

    def create_post(self, user_id: int, content: str):
        with get_connection() as conn:
            cur = conn.execute("INSERT INTO post(user_id, content) VALUES (?, ?)", (user_id, content))
            row = conn.execute("SELECT id, user_id, content, created_at FROM post WHERE id = ?", (cur.lastrowid,)).fetchone()
        return dict(row)

    def patch_post(self, post_id: int, content: str):
        with get_connection() as conn:
            cur = conn.execute("UPDATE post SET content = ? WHERE id = ?", (content, post_id))
            row = conn.execute("SELECT id, user_id, content, created_at FROM post WHERE id = ?", (post_id,)).fetchone()
        return dict(row) if cur.rowcount else None

    def delete_post(self, post_id: int) -> bool:
        with get_connection() as conn:
            conn.execute("DELETE FROM post_like WHERE post_id = ?", (post_id,))
            cur = conn.execute("DELETE FROM post WHERE id = ?", (post_id,))
            return cur.rowcount > 0

    def follow(self, follower_id: int, followed_id: int):
        with get_connection() as conn:
            conn.execute("INSERT INTO follow(follower_id, followed_id) VALUES (?, ?)", (follower_id, followed_id))
            row = conn.execute("SELECT follower_id, followed_id, created_at FROM follow WHERE follower_id = ? AND followed_id = ?", (follower_id, followed_id)).fetchone()
        return dict(row)

    def unfollow(self, follower_id: int, followed_id: int) -> bool:
        with get_connection() as conn:
            cur = conn.execute("DELETE FROM follow WHERE follower_id = ? AND followed_id = ?", (follower_id, followed_id))
            return cur.rowcount > 0

    def follows(self, user_id: int, direction: str):
        if direction == "following":
            sql = "SELECT u.id, u.username, f.created_at FROM follow f JOIN user u ON u.id = f.followed_id WHERE f.follower_id = ? ORDER BY u.id"
        else:
            sql = "SELECT u.id, u.username, f.created_at FROM follow f JOIN user u ON u.id = f.follower_id WHERE f.followed_id = ? ORDER BY u.id"
        with get_connection() as conn:
            return [dict(r) for r in conn.execute(sql, (user_id,))]

    def like(self, user_id: int, post_id: int):
        with get_connection() as conn:
            conn.execute("INSERT INTO post_like(user_id, post_id) VALUES (?, ?)", (user_id, post_id))
            row = conn.execute("SELECT user_id, post_id, created_at FROM post_like WHERE user_id = ? AND post_id = ?", (user_id, post_id)).fetchone()
        return dict(row)

    def unlike(self, user_id: int, post_id: int) -> bool:
        with get_connection() as conn:
            cur = conn.execute("DELETE FROM post_like WHERE user_id = ? AND post_id = ?", (user_id, post_id))
            return cur.rowcount > 0
