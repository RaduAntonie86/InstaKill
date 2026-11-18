# API contract: semantic search

## `GET /search`

Search the seeded posts by semantic similarity using the pretrained
`all-MiniLM-L6-v2` encoder, running locally on CPU. Corpus and query embeddings
are normalized by the encoder adapter; cosine scoring remains implemented in
NumPy in the application domain.

### Query parameters

- `q` — required, non-empty search text. Whitespace-only values are rejected.
- `limit` — optional integer from 1 through 20; defaults to 10.

### Successful response

Return **200 OK** with the original query and up to `limit` results:

```json
{
  "query": "weekend hiking",
  "results": [
    {
      "post_id": 12,
      "user_id": 3,
      "content": "Found a quiet trail in the hills today",
      "score": 0.63
    }
  ]
}
```

Scores are cosine similarity values. Results are sorted by descending score;
equal scores are ordered by ascending `post_id`.

### Validation errors

- **422 Unprocessable Content** — `q` is missing or empty, or `limit` is not
  between 1 and 20. The response includes FastAPI's validation detail.

## Search implementation boundaries

At application startup, the API initializes SQLite, reads all current posts
through `SQLitePostRepository`, and builds an in-memory embedding index through
`SentenceTransformerEncoder`. The index is held for the process lifetime. The
NumPy cosine and deterministic top-k function is pure and independent of HTTP,
SQLite, and model loading.

When post creation is added, its application service should refresh the index
after a successful database write (or publish an index-refresh event). A later
production version can replace the in-memory index with a rebuildable cache;
this milestone intentionally indexes the small seeded corpus at startup.

## `GET /feed`

Returns up to `limit` (1–50, default 10) posts for a positive `user_id`.
Already-liked and future posts are excluded, ties resolve by ascending post ID,
and unknown users return 404. Four strategies are available:

- `strategy=learned` (default weights 1, 1, 1) ranks with a three-signal
  linear score: freshness + affinity + recent popularity. Optional non-negative
  `freshness`, `affinity`, and `popularity` query weights tune this scorer; at
  least one must be positive. Results include the three signals and score.
- `strategy=for_you` uses the Week 5 one-hidden-layer NumPy neural model over
  all six signals (freshness, affinity, popularity, activity, lifetime, age)
  and returns predicted like probability plus the six signals.
- `strategy=popularity` orders by the product's last-365-day popularity signal.
- `strategy=chronological` orders newest eligible posts first.

The model is loaded from the project-root `like_model.npz` once at startup.
`for_you` and `learned` use the exact signal builder and constants from that
artifact; MovieLens posts use the archive's latest rating time as their feature
reference time, while live posts use the current request time. If the model
file is missing, those two strategies return 503. Run `python scripts/seed_db.py`
to create the demo users, topical posts, and likes.

## Social resources

Users support `POST /users` (`{"username":"alice"}`), `GET /users`,
`GET /users/{user_id}`, `PATCH /users/{user_id}` (`{"username":"new-name"}`),
and `DELETE /users/{user_id}`. Posts support the same CRUD shape at `/posts`; a
create body is `{"user_id":1,"content":"Hello"}` and a patch body is
`{"content":"Edited"}`. Collection GETs return arrays, item GETs return one
object, create returns 201, and delete returns 204. Missing items return 404;
duplicate usernames and duplicate relationships return 409.

`POST /follows` accepts `{"follower_id":1,"followed_id":2}` and returns the
relationship including `created_at`. `DELETE /follows/{follower_id}/{followed_id}`
removes it. `GET /users/{user_id}/following` and
`GET /users/{user_id}/followers` return arrays of users with the relationship's
`created_at`. Self-follows and duplicate follows return 409.

`POST /likes` accepts `{"user_id":1,"post_id":1}` and returns the like,
including `created_at`. `DELETE /likes/{user_id}/{post_id}` removes it. Duplicate
likes return 409. User and post deletion also removes dependent follows and likes.
