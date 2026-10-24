"""Sentence Transformers adapter; model loading stays outside retrieval logic."""

from collections.abc import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class SentenceTransformerEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self._model_name = model_name
        self._model: SentenceTransformer | None = None

    def _load(self) -> SentenceTransformer:
        if self._model is None:
            self._model = SentenceTransformer(self._model_name, device="cpu")
        return self._model

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.empty((0, 0), dtype=np.float32)
        return np.asarray(
            self._load().encode(list(texts), convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )
