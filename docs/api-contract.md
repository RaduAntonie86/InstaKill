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
