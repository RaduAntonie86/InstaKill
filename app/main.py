from contextlib import asynccontextmanager

import sqlite3
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, Query, Response
from datetime import datetime, timezone

from app.db.database import init_db
from app.application.search import SearchIndex
from app.adapters.embeddings import SentenceTransformerEncoder
from app.repositories.posts import SQLitePostRepository
from app.application.feed import rank_feed
from app.repositories.social import SQLiteSocialRepository
from app.domain.learned_feed import load_model
from pathlib import Path
from fastapi.responses import FileResponse


class UserCreate(BaseModel):
    username: str = Field(min_length=1, max_length=80)

    @field_validator("username")
    @classmethod
    def trim_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("username must contain non-whitespace text")
        return value


class UserPatch(BaseModel):
    username: str = Field(min_length=1, max_length=80)

    @field_validator("username")
    @classmethod
    def trim_username(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("username must contain non-whitespace text")
        return value


class PostCreate(BaseModel):
    user_id: int = Field(gt=0)
    content: str = Field(min_length=1)


class PostPatch(BaseModel):
    content: str = Field(min_length=1)


class FollowCreate(BaseModel):
    follower_id: int = Field(gt=0)
    followed_id: int = Field(gt=0)


class LikeCreate(BaseModel):
    user_id: int = Field(gt=0)
    post_id: int = Field(gt=0)


def _social_write(action):
    try:
        return action()
    except sqlite3.IntegrityError as exc:
        raise HTTPException(status_code=409, detail="Resource already exists or violates a relationship constraint") from exc


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    # Build a small in-memory index from the seeded SQLite posts at startup.
    repository = SQLitePostRepository()
    encoder = SentenceTransformerEncoder()
    app.state.search_index = SearchIndex(repository, encoder)
    app.state.search_index.refresh()
    model_path = Path(__file__).resolve().parent.parent / "like_model.npz"
    try:
        app.state.like_model = load_model(model_path)
    except FileNotFoundError:
        app.state.like_model = None
    yield


app = FastAPI(title="InstaKill", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(Path(__file__).resolve().parent / "static" / "index.html")


@app.post("/users", status_code=201)
def create_user(body: UserCreate):
    return _social_write(lambda: SQLiteSocialRepository().create_user(body.username.strip()))


@app.get("/users")
def list_users():
    return SQLiteSocialRepository().list_users()


@app.get("/users/{user_id}")
def get_user(user_id: int):
    row = SQLiteSocialRepository().get_user(user_id)
    if row is None:
        raise HTTPException(404, "user not found")
    return row


@app.patch("/users/{user_id}")
def patch_user(user_id: int, body: UserPatch):
    row = _social_write(lambda: SQLiteSocialRepository().patch_user(user_id, body.username.strip()))
    if row is None:
        raise HTTPException(404, "user not found")
    return row


@app.delete("/users/{user_id}", status_code=204)
def delete_user(user_id: int):
    if not SQLiteSocialRepository().delete_user(user_id):
        raise HTTPException(404, "user not found")
    return Response(status_code=204)


@app.post("/posts", status_code=201)
def create_post(body: PostCreate):
    return _social_write(lambda: SQLiteSocialRepository().create_post(body.user_id, body.content))


@app.get("/posts")
def list_posts():
    return SQLiteSocialRepository().list_posts()


@app.get("/posts/{post_id}")
def get_post(post_id: int):
    row = SQLiteSocialRepository().get_post(post_id)
    if row is None:
        raise HTTPException(404, "post not found")
    return row


@app.patch("/posts/{post_id}")
def patch_post(post_id: int, body: PostPatch):
    row = SQLiteSocialRepository().patch_post(post_id, body.content)
    if row is None:
        raise HTTPException(404, "post not found")
    return row


@app.delete("/posts/{post_id}", status_code=204)
def delete_post(post_id: int):
    if not SQLiteSocialRepository().delete_post(post_id):
        raise HTTPException(404, "post not found")
    return Response(status_code=204)


@app.post("/follows", status_code=201)
def create_follow(body: FollowCreate):
    return _social_write(lambda: SQLiteSocialRepository().follow(body.follower_id, body.followed_id))


@app.delete("/follows/{follower_id}/{followed_id}", status_code=204)
def delete_follow(follower_id: int, followed_id: int):
    if not SQLiteSocialRepository().unfollow(follower_id, followed_id):
        raise HTTPException(404, "follow not found")
    return Response(status_code=204)


@app.get("/users/{user_id}/following")
def following(user_id: int):
    repository = SQLiteSocialRepository()
    if repository.get_user(user_id) is None:
        raise HTTPException(404, "user not found")
    return repository.follows(user_id, "following")


@app.get("/users/{user_id}/followers")
def followers(user_id: int):
    repository = SQLiteSocialRepository()
    if repository.get_user(user_id) is None:
        raise HTTPException(404, "user not found")
    return repository.follows(user_id, "followers")


@app.post("/likes", status_code=201)
def create_like(body: LikeCreate):
    return _social_write(lambda: SQLiteSocialRepository().like(body.user_id, body.post_id))


@app.delete("/likes/{user_id}/{post_id}", status_code=204)
def delete_like(user_id: int, post_id: int):
    if not SQLiteSocialRepository().unlike(user_id, post_id):
        raise HTTPException(404, "like not found")
    return Response(status_code=204)


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


@app.get("/feed")
def feed(
    user_id: int = Query(..., ge=1),
    limit: int = Query(10, ge=1, le=50),
    strategy: str = Query("weighted", pattern="^(weighted|for_you|chronological|learned)$"),
    freshness: float = Query(1.0, ge=0),
    affinity: float = Query(1.0, ge=0),
    popularity: float = Query(1.0, ge=0),
) -> dict:
    """Personalized feed using freshness, topic affinity, and recent popularity."""
    repository = SQLitePostRepository()
    if not repository.user_exists(user_id):
        raise HTTPException(status_code=404, detail=f"user {user_id} not found")
    posts = repository.list_feed_posts()
    history = repository.list_likes()
    now = datetime.now(timezone.utc).timestamp()
    if strategy == "chronological":
        results = rank_feed(user_id, posts, history, now, limit, "chronological")
        return {"user_id": user_id, "strategy": strategy, "results": results}
    if strategy == "for_you":
        model = getattr(app.state, "like_model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="for_you feed signal constants are unavailable")
        weights = {"freshness": freshness, "affinity": affinity, "popularity": popularity}
        if not any(weights.values()):
            raise HTTPException(status_code=422, detail="at least one ranking weight must be positive")
        results = rank_feed(user_id, posts, history, now, limit, "for_you", model, weights)
        return {"user_id": user_id, "strategy": strategy, "weights": weights, "results": results}
    if strategy == "learned":
        model = getattr(app.state, "like_model", None)
        if model is None:
            raise HTTPException(status_code=503, detail="learned feed model is unavailable")
        results = rank_feed(user_id, posts, history, now, limit, "learned", model)
        return {"user_id": user_id, "strategy": strategy, "results": results}
    weights = {"freshness": freshness, "affinity": affinity, "popularity": popularity}
    if not any(weights.values()):
        raise HTTPException(status_code=422, detail="at least one ranking weight must be positive")
    results = rank_feed(user_id, posts, history, now, limit, "weighted", weights=weights)
    return {"user_id": user_id, "strategy": strategy, "weights": weights, "results": results}
