from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query

from app.db.database import init_db
from app.application.search import SearchIndex
from app.adapters.embeddings import SentenceTransformerEncoder
from app.repositories.posts import SQLitePostRepository


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Build a small in-memory index from the seeded SQLite posts at startup.
    repository = SQLitePostRepository()
    encoder = SentenceTransformerEncoder()
    app.state.search_index = SearchIndex(repository, encoder)
    app.state.search_index.refresh()
    yield


app = FastAPI(title="InstaKill", lifespan=lifespan)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search")
def search(
    q: str = Query(..., min_length=1, description="Text to search for"),
    limit: int = Query(10, ge=1, le=20),
) -> dict:
    if not q.strip():
        raise HTTPException(status_code=422, detail="q must contain non-whitespace text")
    index: SearchIndex = app.state.search_index
    return {"query": q, "results": index.search(q, limit)}
