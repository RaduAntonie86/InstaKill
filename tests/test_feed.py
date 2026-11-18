from app import main
from pathlib import Path
from app.domain.learned_feed import load_model


def test_learned_feed_uses_three_signal_scorer(monkeypatch):
    class Repository:
        def user_exists(self, user_id):
            return user_id == 1

        def list_feed_posts(self):
            return [{"post_id": 1, "user_id": 2, "content": "post", "topic": "general", "created_at": 1.0}]

        def list_likes(self):
            return []

    monkeypatch.setattr(main, "SQLitePostRepository", Repository)
    model_path = Path(__file__).resolve().parents[1] / "like_model.npz"
    monkeypatch.setattr(main.app.state, "like_model", load_model(model_path), raising=False)
    response = main.feed(user_id=1, limit=10, strategy="learned",
                         freshness=1.0, affinity=1.0, popularity=1.0)
    assert response["strategy"] == "learned"
    assert [post["post_id"] for post in response["results"]] == [1]
    assert set(response["results"][0]["signals"]) == {"freshness", "affinity", "popularity"}
