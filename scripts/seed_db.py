"""Seed repeatable demo users, topical posts, and likes for /feed testing."""
from datetime import datetime, timedelta, timezone

from app.db.database import get_connection, init_db

USERS = ["alice", "bob", "carol", "dave", "erin"]
POSTS = [
    ("alice", "hello world, this is my first post", "general"),
    ("alice", "Fresh pasta and roasted tomatoes for dinner", "food"),
    ("alice", "A quiet trail through the forest this morning", "outdoors"),
    ("bob", "Just shipped a tiny Python tool for my team", "technology"),
    ("bob", "My desk setup finally feels right", "technology"),
    ("bob", "The little bakery on the corner is excellent", "food"),
    ("carol", "A watercolor sketch from the weekend", "art"),
    ("carol", "Trying a new recipe with fresh basil", "food"),
    ("carol", "Museum day: the new printmaking exhibit is lovely", "art"),
    ("dave", "Sunrise ride along the river today", "outdoors"),
    ("dave", "Learning about batteries and electric cars", "technology"),
    ("dave", "Best picnic spot has a view of the whole city", "outdoors"),
    ("erin", "A quick sketch of the old train station", "art"),
    ("erin", "Homemade ramen on a rainy evening", "food"),
    ("erin", "Found a peaceful path near the lake", "outdoors"),
    ("alice", "Three useful keyboard shortcuts I learned today", "technology"),
    ("bob", "A gallery full of bold colors and shapes", "art"),
    ("carol", "The neighborhood garden is blooming again", "outdoors"),
    ("dave", "Testing a new local-first notes app", "technology"),
    ("erin", "Finally perfected the sourdough crust", "food"),
]
PREFERENCES = {
    "alice": ["food", "outdoors"], "bob": ["technology", "art"],
    "carol": ["art", "food"], "dave": ["outdoors", "technology"],
    "erin": ["food", "art"],
}


def seed() -> None:
    init_db()
    with get_connection() as conn:
        for username in USERS:
            conn.execute("INSERT OR IGNORE INTO user (username) VALUES (?)", (username,))
        user_ids = {row["username"]: row["id"] for row in conn.execute("SELECT id, username FROM user")}
        for username, content, topic in POSTS:
            conn.execute(
                "INSERT INTO post (user_id, content, topic) "
                "SELECT ?, ?, ? WHERE NOT EXISTS (SELECT 1 FROM post WHERE content = ?)",
                (user_ids[username], content, topic, content),
            )
        topic_posts = {}
        for row in conn.execute("SELECT id, topic FROM post ORDER BY id"):
            topic_posts.setdefault(row["topic"], []).append(row["id"])
        now = datetime.now(timezone.utc)
        for user_index, (username, topics) in enumerate(PREFERENCES.items()):
            for topic_index, topic in enumerate(topics):
                for post_index, post_id in enumerate(topic_posts.get(topic, [])[:2]):
                    created_at = (now - timedelta(days=2 + user_index * 3 + topic_index + post_index)).strftime("%Y-%m-%d %H:%M:%S")
                    conn.execute(
                        "INSERT OR IGNORE INTO post_like (user_id, post_id, created_at) VALUES (?, ?, ?)",
                        (user_ids[username], post_id, created_at),
                    )
        conn.commit()
    print(f"Seeded {len(USERS)} users, {len(POSTS)} topical demo posts, and preference likes (repeatable).")


if __name__ == "__main__":
    seed()
