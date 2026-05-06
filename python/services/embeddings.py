from __future__ import annotations

import hashlib
import math
from typing import Iterable


class EmbeddingService:
    """Local embedding wrapper with a deterministic fallback for tests/offline demos."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        dimension: int = 384,
        force_fallback: bool = False,
    ):
        self.model_name = model_name
        self.dimension = dimension
        self.force_fallback = force_fallback
        self._model = None
        self.using_fallback = force_fallback

    def embed_query(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        text_list = list(texts)
        if not text_list:
            return []
        if self.force_fallback:
            return [self._hash_embedding(text) for text in text_list]
        try:
            model = self._load_model()
            vectors = model.encode(text_list, normalize_embeddings=True)
            return [self._coerce_vector(vector) for vector in vectors]
        except Exception:
            self.using_fallback = True
            return [self._hash_embedding(text) for text in text_list]

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _coerce_vector(self, vector) -> list[float]:
        values = [float(item) for item in vector]
        if len(values) == self.dimension:
            return values
        if len(values) > self.dimension:
            return values[: self.dimension]
        return values + [0.0] * (self.dimension - len(values))

    def _hash_embedding(self, text: str) -> list[float]:
        vector = [0.0] * self.dimension
        tokens = text.lower().split()
        if not tokens:
            tokens = [text.lower()]
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            for index, byte in enumerate(digest):
                slot = (byte + index * 31) % self.dimension
                vector[slot] += 1.0 if byte % 2 == 0 else -1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]
