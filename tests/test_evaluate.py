import csv
import sqlite3
from pathlib import Path

import numpy as np

from scripts import evaluate
from app.application.feed import rank_feed
from app.domain.learned_feed import build_features, load_model, predict


def test_ndcg_hand_made_rankings():
    assert evaluate.ndcg_at_10([1, 2], set()) == 0
    assert evaluate.ndcg_at_10([1, 2], {1, 2}) == 1
    expected = 1 / np.log2(5)
    assert np.isclose(evaluate.ndcg_at_10([8, 9, 7, 2], {2}), expected)


def _ranking_fixture():
    model = load_model(evaluate.ROOT / "like_model.npz")
    now = evaluate.epoch("2020-01-10T00:00:00+00:00")
    posts = [
        {"post_id": 1, "user_id": 2, "content": "liked", "topic": "Comedy, Drama", "created_at": now - 10 * evaluate.DAY},
        {"post_id": 2, "user_id": 2, "content": "one", "topic": "Comedy, Drama", "created_at": now - 2 * evaluate.DAY},
        {"post_id": 3, "user_id": 2, "content": "two", "topic": "Action", "created_at": now - evaluate.DAY},
    ]
    history = [{"user_id": 1, "post_id": 1, "created_at": now - 5 * evaluate.DAY}]
    return model, now, posts, history


def test_learned_uses_three_signals():
    model, now, posts, history = _ranking_fixture()
    features, _ = build_features(1, posts, history, now, model)
    expected_scores = features[:, :3].sum(axis=1)
    result = rank_feed(1, posts, history, now, 10, "learned", model)
    expected_ids = sorted((2, 3), key=lambda post_id: (-expected_scores[post_id - 1], post_id))
    assert [item["post_id"] for item in result] == expected_ids
    for item in result:
        assert np.isclose(item["score"], expected_scores[item["post_id"] - 1])
        assert set(item["signals"]) == {"freshness", "affinity", "popularity"}


def test_for_you_uses_six_signal_neural_model():
    model, now, posts, history = _ranking_fixture()
    features, _ = build_features(1, posts, history, now, model)
    expected_scores = predict(model, features)
    result = rank_feed(1, posts, history, now, 10, "for_you", model)
    expected_ids = sorted((2, 3), key=lambda post_id: (-expected_scores[post_id - 1], post_id))
    assert [item["post_id"] for item in result] == expected_ids
    for item in result:
        assert np.isclose(item["score"], expected_scores[item["post_id"] - 1])
        assert set(item["signals"]) == {"freshness", "affinity", "popularity", "activity", "lifetime", "age"}


def fixture_db(path: Path):
    with sqlite3.connect(path) as conn:
        conn.executescript("""
        PRAGMA foreign_keys=ON;
        CREATE TABLE user(id INTEGER PRIMARY KEY, username TEXT, created_at TEXT);
        CREATE TABLE post(id INTEGER PRIMARY KEY, user_id INTEGER, content TEXT, created_at TEXT, topic TEXT);
        CREATE TABLE post_like(user_id INTEGER, post_id INTEGER, created_at TEXT, PRIMARY KEY(user_id,post_id));
        CREATE TABLE follow(follower_id INTEGER, followed_id INTEGER, created_at TEXT, PRIMARY KEY(follower_id,followed_id));
        INSERT INTO user VALUES (1,'u','2020-01-01T00:00:00+00:00'),(2,'v','2020-01-01T00:00:00+00:00');
        INSERT INTO post VALUES (1,1,'a','2020-01-01T00:00:00+00:00','a'),(2,2,'b','2020-01-02T00:00:00+00:00','b'),
          (3,1,'c','2020-01-03T00:00:00+00:00','a'),(4,2,'d','2020-01-04T00:00:00+00:00','b');
        INSERT INTO post_like VALUES (1,1,'2020-01-05T00:00:00+00:00'),(1,2,'2020-01-06T00:00:00+00:00'),
          (1,3,'2020-01-07T00:00:00+00:00'),(1,4,'2020-01-10T00:00:00+00:00');
        INSERT INTO follow VALUES (1,2,'2020-01-09T00:00:00+00:00');
        """)


def test_snapshot_has_no_future_rows(tmp_path):
    db, snap = tmp_path / "source.db", tmp_path / "snapshot.db"
    fixture_db(db)
    evaluate._snapshot(db, snap, evaluate.epoch("2020-01-10T00:00:00+00:00"))
    with sqlite3.connect(snap) as conn:
        assert conn.execute("SELECT COUNT(*) FROM post_like WHERE created_at >= '2020-01-10'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM post WHERE created_at >= '2020-01-10'").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM follow WHERE created_at >= '2020-01-10'").fetchone()[0] == 0


def test_replay_deterministic_and_calls_shared_service(tmp_path, monkeypatch):
    db = tmp_path / "fixture.db"
    fixture_db(db)
    first, second = tmp_path / "one", tmp_path / "two"
    calls = []
    real = evaluate.rank_feed

    def counted(*args, **kwargs):
        calls.append(args[5])
        return real(*args, **kwargs)

    monkeypatch.setattr(evaluate, "rank_feed", counted)
    cutoff = "2020-01-10T00:00:00+00:00"
    evaluate.evaluate(db, cutoff, 42, first)
    evaluate.evaluate(db, cutoff, 42, second)
    assert calls == list(evaluate.STRATEGIES) * 2
    assert (first / "eval_results.csv").read_bytes() == (second / "eval_results.csv").read_bytes()
    assert (first / "baseline_table.md").read_bytes() == (second / "baseline_table.md").read_bytes()
    with (first / "eval_results.csv").open() as f:
        assert len(list(csv.DictReader(f))) == 4
