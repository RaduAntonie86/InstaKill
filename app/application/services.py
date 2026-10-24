"""Application-level operations built on the existing domain models."""

from app.domain.models import Post, User


def prepare_post(author: User, content: str) -> Post:
    """Build a new, not-yet-persisted post for an existing author."""
    if author.id is None:
        raise ValueError("author must have an assigned id")
    return Post(user_id=author.id, content=content)
