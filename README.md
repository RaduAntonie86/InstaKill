# InstaKill — Starter Repo

Mini social-media platform. Mandatory backend: Python + FastAPI + SQLite (default) / PostgreSQL (allowed). See the full brief and grading rubric in `masterplan.md` (course root).

What the instructor provides (this starter):

- `app/main.py` — FastAPI hello-world with `GET /health`
- `app/db/database.py` — SQLite bootstrap (creates tables on startup)
- `app/domain/models.py` — `User` / `Post` schema
- `scripts/seed_db.py` — 5-10 synthetic users/posts
- `tests/test_health.py` — test skeleton

What it does **not** provide: the feed, ranking, ML models, search. Those are the team's work, in the following weeks.

## Semantic search (Week 3)

`GET /search?q=<text>&limit=<1..20>` searches the posts currently in SQLite.
The application builds an in-memory index during FastAPI startup: it reads
posts through the SQLite repository and encodes their content with the local
CPU `all-MiniLM-L6-v2` Sentence Transformers model. The pure NumPy retrieval
function computes cosine similarity and breaks score ties by ascending post
ID. The first run may download the pretrained model; no hosted inference or
vector database is used. See [docs/api-contract.md](docs/api-contract.md).

When post creation is implemented, the application should refresh the index
after saving a new post (or use an index-refresh event). Until then, restart
the application after changing the seeded database to rebuild the index.

## Running it

```bash
pip install -e .
python scripts/seed_db.py
uvicorn app.main:app --reload
# open http://localhost:8000 for the 2012-style API playground, or use the API directly:
curl localhost:8000/health
curl "localhost:8000/feed?user_id=1"

pytest
```

## Personalized feed (Week 4)

`GET /feed?user_id=1` ranks posts using freshness, topic affinity from the
user's past likes, and recent post popularity. It excludes posts the user has
already liked and posts created in the future. Defaults match the notebook
weights `(1, 1, 1)`; tune them with `freshness`, `affinity`, and `popularity`
query parameters, for example `/feed?user_id=1&freshness=2&affinity=2&popularity=3`.
Results include each signal and the combined score. The sample database contains
five users, topical posts, and likes; regenerate it with `python scripts/seed_db.py`.

## Offline feed evaluation (Week 7)

Install the evaluation tools with `pip install -e ".[eval]"`, then run
`python scripts/evaluate.py --db instakill.db --seed 42 --out reports/`. The
script takes a temporary time-split snapshot, evaluates chronological,
popularity, for_you, and learned feeds through the shared ranking service, and
writes per-user results, an interval table, and a self-contained HTML dashboard.
An explicit cutoff can be supplied with `--cutoff 2009-09-10T09:26:52+00:00`.
See [docs/evaluation-protocol.md](docs/evaluation-protocol.md) for the protocol
and known limitations.

## Milestones (see masterplan.md for full details)

W1 Repo+API → W3 CRUD+DB → W4 Feed v0 → W7 Baseline+eval plan → W9 ML+pretrained feature → W11 Profile+optimize → W13 Feature freeze+E2E → W14 Red-team+Demo Day.

At the end of each week: a demonstrable commit/tag.
