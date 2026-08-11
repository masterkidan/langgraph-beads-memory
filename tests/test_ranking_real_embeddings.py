"""Ranking behaviour against the REAL embedder.

`FakeEmbedder` is deterministic token-overlap, not semantics. That is the right
default for the unit suite — fast, offline, no model — and it genuinely verifies
the plumbing: namespace scope, the exclusion list, the descendant penalty
arithmetic, dedup.

What it cannot verify is that retrieval ranks *sensibly*, because it has no
notion of meaning. Making it pretend to would be worse than leaving it honest: a
hand-built topic table would encode the expected answer into the fixture, so the
tests would pass by construction and stay green if the real embedder disagreed.

These tests therefore use `OllamaEmbedder`. They are skipped unless Ollama is
reachable, so they never block the unit suite.

    uv run pytest tests/test_ranking_real_embeddings.py -v
"""

from __future__ import annotations

import pytest

from beads_memory.store import BeadsStore


@pytest.fixture(scope="module")
def real_embedder():
    """Skip cleanly when Ollama is absent or busy, rather than failing the suite."""
    import urllib.error
    import urllib.request

    try:
        urllib.request.urlopen("http://localhost:11434/api/version", timeout=3)
    except (urllib.error.URLError, OSError) as e:
        pytest.skip(f"Ollama not reachable: {e}")
    from beads_memory.embeddings import OllamaEmbedder

    emb = OllamaEmbedder()
    try:
        emb.embed("warmup")
    except Exception as e:  # noqa: BLE001 - any failure here means "cannot test"
        pytest.skip(f"embedding call failed: {e}")
    return emb


def _store(conn):
    s = BeadsStore(conn)
    s.init_schema()
    return s


def _w(store, ns, body, emb, kind="conclusion"):
    return store.write_fact(
        ns,
        kind=kind,
        body=body,
        source="remember_tool",
        source_key=body,
        agent_id="a",
        acting_on_behalf_of="user",
        embedding=emb.embed(body),
    )


def test_pointed_question_reaches_a_buried_child_fact(conn, real_embedder):
    """The exact failure this design change targets: a detail a sub-agent found,
    which its rollup omitted, must be reachable when asked for directly."""
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    child = store.fork_namespace(root)
    _w(
        store,
        child,
        "Binary quantization reduces RAM usage up to 32x with a modest recall hit",
        real_embedder,
    )
    _w(store, root, "Qdrant is self-hostable with deployment via a single binary", real_embedder)
    _w(store, root, "the budget is $50,000 per year", real_embedder)
    _w(store, root, "it must be self-hostable", real_embedder)

    hits = store.search(
        root.id,
        real_embedder.embed("what was that big memory optimization the Qdrant researcher found?"),
        k=4,
    )
    assert any("quantization" in h.body for h in hits), [h.body for h in hits]


def test_child_noise_does_not_displace_a_parent_constraint(conn, real_embedder):
    """The property isolation was protecting, verified semantically rather than
    by token overlap."""
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    child = store.fork_namespace(root)
    for i in range(10):
        _w(
            store,
            child,
            f"Weaviate module system note {i}: rerankers add upgrade complexity",
            real_embedder,
        )
    _w(store, root, "the budget is $50,000 per year", real_embedder)

    hits = store.search(root.id, real_embedder.embed("what is our annual budget?"), k=3)
    assert any("50,000" in h.body for h in hits), [h.body for h in hits]


def test_an_identical_fact_ranks_higher_in_the_parent(conn, real_embedder):
    """Demotion is real under real embeddings, not an artifact of the fake."""
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    child = store.fork_namespace(root)
    text = "the annual budget is $50,000"
    _w(store, child, text, real_embedder)
    _w(store, root, text, real_embedder)

    hits = store.search(root.id, real_embedder.embed("annual budget"), k=2)
    assert hits[0].namespace_id == root.id


def test_directives_stay_out_of_a_semantically_matching_query(conn, real_embedder):
    """A directive is *most* similar to a query that restates it — the case the
    fake embedder cannot exercise, and the one the exclusion exists for."""
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    q = "which vector database should we pick and why?"
    _w(store, root, q, real_embedder, kind="directive")
    _w(store, root, "the budget is $50,000 per year", real_embedder, kind="user_input")

    hits = store.search(root.id, real_embedder.embed(q), k=2)
    assert not any(h.kind == "directive" for h in hits), [(h.kind, h.body) for h in hits]
    assert any("50,000" in h.body for h in hits)
