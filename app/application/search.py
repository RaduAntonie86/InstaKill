"""Application service that coordinates repository, encoder, and retrieval."""

from app.domain.retrieval import top_k_by_cosine


class SearchIndex:
    def __init__(self, repository, encoder):
        self._repository = repository
        self._encoder = encoder
        self._posts: list[dict] = []
        self._vectors = None

    def refresh(self) -> None:
        self._posts = self._repository.list_search_documents()
        self._vectors = self._encoder.encode([post["content"] for post in self._posts])

    def search(self, query: str, limit: int) -> list[dict]:
        query_vector = self._encoder.encode([query])[0]
        return top_k_by_cosine(query_vector, self._vectors, self._posts, limit)
