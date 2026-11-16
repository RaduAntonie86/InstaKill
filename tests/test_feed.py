from app import main


def test_weighted_feed_does_not_require_learned_model(monkeypatch):
    class Repository:
        def user_exists(self, user_id):
            return user_id == 1

        def list_feed_posts(self):
            return [{"post_id": 1, "user_id": 2, "content": "post", "topic": "general", "created_at": 1.0}]

        def list_likes(self):
            return []

    monkeypatch.setattr(main, "SQLitePostRepository", Repository)
    response = main.feed(user_id=1, limit=10, strategy="weighted",
                         freshness=1.0, affinity=1.0, popularity=1.0)
    assert response["strategy"] == "weighted"
    assert [post["post_id"] for post in response["results"]] == [1]
