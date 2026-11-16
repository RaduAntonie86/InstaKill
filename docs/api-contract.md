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

Builds a personalized ranking from existing SQLite posts and likes. The
recommendation logic mirrors the Week 4 notebook: freshness decays with a
three-year half-life, affinity is the share of the user's historical likes in
the post's topic, and popularity is the post's share of all likes in the last
365 days. The default score is the sum of those three signals. The top results
are selected with a bounded heap and ties resolve by ascending post ID.

Parameters: required `user_id` (positive integer), `limit` (1–50, default 10),
and non-negative `freshness`, `affinity`, `popularity` weights (each default 1;
at least one must be positive). Users' already-liked posts and future posts are
excluded. The response contains the weights and ranked posts, each with its
combined `score` and individual `signals`. Unknown users return 404. Run
`python scripts/seed_db.py` to create the demo users, topical posts, and likes.

Add `strategy=learned` to rank with the Week 5 NumPy model instead of the
weighted Week 4 formula. The model is loaded from the project-root
`like_model.npz` once at startup. Learned results include a predicted like
probability in `score` and the six notebook signals in `signals`; if the model
file is missing, this strategy returns 503. The learned scorer computes scores
for all candidate posts in one NumPy batch. MovieLens posts use the archive's
latest rating timestamp as their feature reference time, preserving the
historical one-year popularity window and 25-year age cap; live posts use the
current request time.

`strategy=for_you` scores the shared Week 5 `freshness`, `affinity`, and
`popularity` signals with the Week 4 weights (all `1` by default). It returns
those three signal values with each post. `strategy=weighted` keeps the legacy
Week 4 scorer available. Use `strategy=chronological` to rank eligible posts
newest first. Popularity is an offline evaluation baseline only and is not
exposed by `GET /feed`.

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
