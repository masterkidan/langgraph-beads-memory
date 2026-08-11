"""Embedding providers. Demo embeds synchronously."""

from __future__ import annotations

import hashlib
from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class OllamaEmbedder:  # pragma: no cover
    """nomic-embed-text via langchain-ollama; dim must match schema vector(768).

    `timeout` is not optional in practice. Embedding runs inside passive
    capture, which runs inside the model-call hot path — an embedding request
    that Ollama accepts and never answers blocks that thread forever, and
    because sub-agents run on a thread pool it takes the rest of the run with
    it. Observed twice: the harness parked with threads in socket reads and
    lock waits while Ollama sat idle with no model loaded. Chat calls were
    already bounded; these were not.
    """

    dim = 768

    def __init__(self, model: str = "nomic-embed-text", timeout: float = 120.0):
        from langchain_ollama import OllamaEmbeddings

        self._emb = OllamaEmbeddings(model=model, client_kwargs={"timeout": timeout})

    def embed(self, text: str) -> list[float]:
        return self._emb.embed_query(text)


class FakeEmbedder:
    """Deterministic embeddings for tests: same text -> same vector; token overlap
    -> higher cosine similarity.

    It has NO notion of meaning, and deliberately so. It verifies plumbing —
    namespace scope, the exclusion list, the descendant rank penalty, dedup —
    fast and offline. It cannot verify that retrieval ranks sensibly, and it must
    not be made to look as if it can: a hand-built topic table would encode the
    expected answer into the fixture, so tests would pass by construction and
    stay green if the real embedder disagreed. Semantic ranking is covered by
    tests/test_ranking_real_embeddings.py against OllamaEmbedder."""

    dim = 768

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            digest = hashlib.blake2b(tok.encode(), digest_size=4).digest()
            vec[int.from_bytes(digest, "big") % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]


class FakeRanker:
    """Embeddings with test-declared cosine similarity.

    `FakeEmbedder` gives token overlap, which is fine for plumbing but useless
    for ranking: a test cannot say "suppose the child's fact is a *better*
    semantic match than the parent's" — the thing the descendant penalty exists
    to override.

    This separates the two concerns. A test declares the similarity it wants and
    gets vectors that have exactly that cosine to the query, so ranking logic is
    tested against a stated premise rather than an accidental one. Nothing here
    guesses at meaning, which is what makes it honest: the relationship is an
    input to the test, not a property of the fixture.

        r = FakeRanker()
        q = r.query("what was the memory optimization?")
        child  = r.at(0.90)   # deliberately the better match
        parent = r.at(0.50)
        # ...assert the parent still ranks first once demoted

    Construction: the query is the first basis vector; a vector at similarity c
    is `c*e0 + sqrt(1-c^2)*e_k` with a fresh orthogonal `e_k` each time, so
    cosine to the query is exactly c and distinct texts stay distinguishable.
    """

    dim = 768

    def __init__(self) -> None:
        self._next_axis = 1

    def query(self, _text: str = "") -> list[float]:
        vec = [0.0] * self.dim
        vec[0] = 1.0
        return vec

    def at(self, similarity: float) -> list[float]:
        """A vector whose cosine to `query()` is exactly `similarity`."""
        if not -1.0 <= similarity <= 1.0:
            raise ValueError(f"similarity must be in [-1, 1], got {similarity}")
        axis = self._next_axis
        self._next_axis += 1
        if axis >= self.dim:
            raise ValueError("FakeRanker ran out of orthogonal axes")
        vec = [0.0] * self.dim
        vec[0] = similarity
        vec[axis] = (1.0 - similarity**2) ** 0.5
        return vec

    def embed(self, text: str) -> list[float]:
        """Embedder-compatible fallback so it can stand in for FakeEmbedder."""
        return FakeEmbedder().embed(text)
