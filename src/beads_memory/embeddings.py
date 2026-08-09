"""Embedding providers. Demo embeds synchronously."""

from __future__ import annotations

import hashlib
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class OllamaEmbedder:  # pragma: no cover
    """nomic-embed-text via langchain-ollama; dim must match schema vector(768)."""

    dim = 768

    def __init__(self, model: str = "nomic-embed-text"):
        from langchain_ollama import OllamaEmbeddings

        self._emb = OllamaEmbeddings(model=model)

    def embed(self, text: str) -> list[float]:
        return self._emb.embed_query(text)


class FakeEmbedder:
    """Deterministic embeddings for tests: same text -> same vector; token overlap
    -> higher cosine similarity. Not semantically meaningful, but ranking-testable."""

    dim = 768

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            digest = hashlib.blake2b(tok.encode(), digest_size=4).digest()
            vec[int.from_bytes(digest, "big") % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]
