"""Time-split, offline evaluation of the four feed ranking strategies."""

import argparse
import base64
import csv
import hashlib
import io
import sqlite3
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.application.feed import rank_feed
from app.db import database
from app.domain.learned_feed import load_model
from app.repositories.posts import SQLitePostRepository

STRATEGIES = ("chronological", "popularity", "for_you", "learned")
STRATEGY_COLORS = {"chronological": "#4c78a8", "popularity": "#f58518", "for_you": "#54a24b", "learned": "#b279a2"}
DAY = 86400.0


def epoch(value: str) -> float:
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def ndcg_at_10(ranking: list[int], relevant: set[int]) -> float:
    if not relevant:
        return 0.0
    dcg = sum((1.0 / np.log2(position + 2)) for position, post_id in enumerate(ranking[:10]) if post_id in relevant)
    ideal = sum(1.0 / np.log2(position + 2) for position in range(min(10, len(relevant))))
    return float(dcg / ideal) if ideal else 0.0


def bootstrap_interval(values: np.ndarray, rng: np.random.Generator, draws: int = 5000) -> tuple[float, float]:
    if not len(values):
        return (0.0, 0.0)
    samples = rng.integers(0, len(values), size=(draws, len(values)))
    means = np.asarray(values)[samples].mean(axis=1)
    return tuple(float(x) for x in np.quantile(means, [0.025, 0.975]))


def _cutoff_epoch(conn: sqlite3.Connection, requested: str | None) -> float:
    if requested:
        parsed = datetime.fromisoformat(requested.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()
    values = [epoch(row[0]) for row in conn.execute("SELECT created_at FROM post_like ORDER BY created_at, user_id, post_id")]
    if not values:
        raise ValueError("database has no likes; provide a seeded database")
    return float(np.quantile(values, 0.8, method="higher"))


def _snapshot(source: Path, target: Path, cutoff: float) -> None:
    src = sqlite3.connect(source)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
        dst.execute("PRAGMA foreign_keys = ON")
        future_posts = [row[0] for row in dst.execute("SELECT id, created_at FROM post") if epoch(row[1]) >= cutoff]
        if future_posts:
            marks = ",".join("?" for _ in future_posts)
            dst.execute(f"DELETE FROM post_like WHERE post_id IN ({marks})", future_posts)
        for table in ("post_like", "follow", "post"):
            rows = dst.execute(f"SELECT rowid, created_at FROM {table}").fetchall()
            future = [rowid for rowid, created in rows if epoch(created) >= cutoff]
            if future:
                dst.executemany(f"DELETE FROM {table} WHERE rowid = ?", ((rowid,) for rowid in future))
        dst.commit()
    finally:
        dst.close()
        src.close()


def _draw_dashboard(rows: list[dict], slices: list[str], summaries: dict, deltas: dict, metadata: dict) -> str:
    figures = []
    fig, ax = plt.subplots(figsize=(8, 4.3))
    means = [summaries["overall"][s][0] for s in STRATEGIES]
    lows = [means[i] - summaries["overall"][s][1] for i, s in enumerate(STRATEGIES)]
    highs = [summaries["overall"][s][2] - means[i] for i, s in enumerate(STRATEGIES)]
    ax.bar(STRATEGIES, means, color=[STRATEGY_COLORS[s] for s in STRATEGIES])
    ax.errorbar(range(4), means, yerr=[lows, highs], fmt="none", ecolor="#222", capsize=4)
    ax.set(title="Which strategy has the highest mean NDCG@10?", ylabel="Mean NDCG@10", ylim=(0, 1))
    ax.set_xlabel("Feed strategy")
    fig.tight_layout(); figures.append(fig)

    fig, ax = plt.subplots(figsize=(8, 4.3))
    labels = slices
    vals = [deltas[g][0] for g in labels]
    err = [[vals[i] - deltas[g][1] for i, g in enumerate(labels)], [deltas[g][2] - vals[i] for i, g in enumerate(labels)]]
    colors = [plt.get_cmap("coolwarm")(0.5 + np.clip(v, -1, 1) * 0.45) for v in vals]
    ax.bar(labels, vals, color=colors); ax.errorbar(range(len(labels)), vals, yerr=err, fmt="none", ecolor="#222", capsize=4)
    ax.axhline(0, color="#222", linewidth=1); ax.set(title="Where does learned change the best baseline?", ylabel="NDCG@10 difference", xlabel="User activity group")
    fig.tight_layout(); figures.append(fig)

    fig, ax = plt.subplots(figsize=(8, 4.3))
    groups = ["new", "casual", "regular", "power"]
    x = np.arange(len(groups)); width = 0.19
    for j, strategy in enumerate(STRATEGIES):
        ys = [summaries[g][strategy][0] if g in summaries and strategy in summaries[g] else 0 for g in groups]
        ax.bar(x + (j - 1.5) * width, ys, width, color=STRATEGY_COLORS[strategy], label=strategy)
        bounds = [summaries[g][strategy][1:] if g in summaries and strategy in summaries[g] else (0, 0) for g in groups]
        err = [[ys[i] - bounds[i][0] for i in range(len(groups))], [bounds[i][1] - ys[i] for i in range(len(groups))]]
        ax.errorbar(x + (j - 1.5) * width, ys, yerr=err, fmt="none", ecolor="#222", capsize=2, linewidth=0.8)
    ax.set_xticks(x, groups); ax.set_ylim(0, 1); ax.set(title="How does mean NDCG@10 vary by activity?", ylabel="Mean NDCG@10", xlabel="Likes before cutoff"); ax.legend()
    fig.tight_layout(); figures.append(fig)

    encoded = []
    for fig in figures:
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130); plt.close(fig)
        encoded.append(base64.b64encode(buf.getvalue()).decode("ascii"))
    table = _markdown_table(rows)
    charts = "".join(f'<section><img alt="Evaluation chart {i+1}" src="data:image/png;base64,{image}"></section>' for i, image in enumerate(encoded))
    return f'<!doctype html><html><meta charset="utf-8"><title>InstaKill feed evaluation</title><style>body{{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem;color:#202124}}table{{border-collapse:collapse}}td,th{{border:1px solid #bbb;padding:.45rem}}img{{max-width:100%;height:auto}}code{{overflow-wrap:anywhere}}</style><h1>InstaKill feed evaluation</h1><p>Cutoff: {metadata["cutoff"]} · Seed: {metadata["seed"]} · Users: {metadata["users"]}</p><p>Database SHA-256: <code>{metadata["fingerprint"]}</code><br>Git commit: <code>{metadata["commit"]}</code></p><h2>Baseline table</h2>{table}{charts}</html>'


def _markdown_table(rows: list[dict]) -> str:
    header = "| Scope | Strategy / comparison | Mean / delta | 95% interval | Users / verdict |\n|---|---|---:|---:|---|"
    body = []
    for row in rows:
        body.append(f'| {row["scope"]} | {row["name"]} | {row["value"]:.4f} | [{row["low"]:.4f}, {row["high"]:.4f}] | {row["count"] if row["verdict"] is None else row["verdict"]} |')
    return header + "\n" + "\n".join(body)


def evaluate(db_path: Path, cutoff_text: str | None, seed: int, out: Path) -> None:
    raw = sqlite3.connect(db_path)
    cutoff = _cutoff_epoch(raw, cutoff_text)
    raw.close()
    cutoff_iso = datetime.fromtimestamp(cutoff, timezone.utc).isoformat()
    with tempfile.TemporaryDirectory(prefix="instakill-eval-") as tmp:
        snapshot = Path(tmp) / "snapshot.db"
        _snapshot(db_path, snapshot, cutoff)
        original = database.DB_PATH
        database.DB_PATH = snapshot
        try:
            conn = sqlite3.connect(db_path)
            try:
                conn.row_factory = sqlite3.Row
                posts_future = [dict(r) for r in conn.execute("SELECT id AS post_id, user_id, content, topic, created_at FROM post ORDER BY id")]
                likes_future = [dict(r) for r in conn.execute("SELECT user_id, post_id, created_at FROM post_like ORDER BY created_at, user_id, post_id")]
            finally:
                conn.close()
            for post in posts_future:
                post["created_at"] = epoch(post["created_at"])
            for like in likes_future:
                like["created_at"] = epoch(like["created_at"])
            repository = SQLitePostRepository()
            posts = repository.list_feed_posts()
            history = repository.list_likes()
            post_ids = {p["post_id"] for p in posts}
            before: dict[int, list[dict]] = {}
            after: dict[int, set[int]] = {}
            for like in likes_future:
                if like["created_at"] < cutoff:
                    before.setdefault(like["user_id"], []).append(like)
                elif like["post_id"] in post_ids:
                    after.setdefault(like["user_id"], set()).add(like["post_id"])
            selected = sorted(u for u in before if len(before[u]) >= 3 and after.get(u))
            model = load_model(ROOT / "like_model.npz")
            rows = []
            by_user = {}
            for user_id in selected:
                relevant = after[user_id]
                by_user[user_id] = {}
                for strategy in STRATEGIES:
                    ranked = rank_feed(user_id, posts, history, cutoff, 10, strategy, model if strategy == "learned" else None)
                    score = ndcg_at_10([r["post_id"] for r in ranked], relevant)
                    by_user[user_id][strategy] = score
                    count = len(before[user_id])
                    group = "new" if count < 10 else "casual" if count < 50 else "regular" if count < 200 else "power"
                    rows.append({"user_id": user_id, "strategy": strategy, "ndcg_at_10": f"{score:.12f}", "n_relevant": len(relevant), "user_group": group})
        finally:
            database.DB_PATH = original

    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "eval_results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["user_id", "strategy", "ndcg_at_10", "n_relevant", "user_group"])
        writer.writeheader(); writer.writerows(rows)
    rng = np.random.default_rng(seed)
    groups_by_user = {r["user_id"]: r["user_group"] for r in rows}
    potential_future_likes = sum(1 for like in likes_future if like["user_id"] in selected and like["created_at"] >= cutoff and like["post_id"] in post_ids)
    groups = ["overall", "new", "casual", "regular", "power"]
    summaries = {}; deltas = {}; table_rows = []
    for group in groups:
        users = selected if group == "overall" else [u for u in selected if groups_by_user.get(u) == group]
        summaries[group] = {}
        for strategy in STRATEGIES:
            values = np.asarray([by_user[u][strategy] for u in users])
            low, high = bootstrap_interval(values, rng)
            mean = float(values.mean()) if len(values) else 0.0
            summaries[group][strategy] = (mean, low, high)
            table_rows.append({"scope": group, "name": strategy, "value": mean, "low": low, "high": high, "count": len(users), "verdict": None})
        baseline_name = max(STRATEGIES[:-1], key=lambda s: summaries[group][s][0])
        paired = np.asarray([by_user[u]["learned"] - by_user[u][baseline_name] for u in users])
        low, high = bootstrap_interval(paired, rng)
        delta = float(paired.mean()) if len(paired) else 0.0
        deltas[group] = (delta, low, high)
        verdict = "wins" if low > 0 else "loses" if high < 0 else "unclear"
        table_rows.append({"scope": group, "name": f"learned − {baseline_name}", "value": delta, "low": low, "high": high, "count": len(users), "verdict": verdict})

    fingerprint = hashlib.sha256(db_path.read_bytes()).hexdigest()
    try:
        # Pin the implementation commit so adding the generated reports does
        # not change the metadata and break byte-for-byte reproduction.
        commit = subprocess.check_output(["git", "log", "-1", "--format=%H", "--", "scripts/evaluate.py"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        if not commit:
            commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unavailable"
    metadata = {"cutoff": cutoff_iso, "seed": seed, "users": len(selected), "potential_future_likes": potential_future_likes, "fingerprint": fingerprint, "commit": commit}
    report = "# Feed baseline evaluation\n\n" + _markdown_table(table_rows) + "\n\n## Run metadata\n\n" + "\n".join(f"- **{k}:** {v}" for k, v in metadata.items()) + "\n"
    (out / "baseline_table.md").write_text(report, encoding="utf-8", newline="\n")
    slices = ["overall", "new", "casual", "regular", "power"]
    (out / "dashboard.html").write_text(_draw_dashboard(table_rows, slices, summaries, deltas, metadata), encoding="utf-8", newline="\n")
    print(f"Evaluated {len(selected)} users at {cutoff_iso}; reports written to {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", type=Path, default=ROOT / "instakill.db")
    parser.add_argument("--cutoff", help="UTC ISO date or timestamp; default is the 80th percentile like time")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=Path, default=ROOT / "reports")
    args = parser.parse_args()
    evaluate(args.db.resolve(), args.cutoff, args.seed, args.out.resolve())


if __name__ == "__main__":
    main()
