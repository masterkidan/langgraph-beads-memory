"""Parents can reach a sub-agent's raw findings, ranked below their own facts.

The original design made the parent/child boundary binary: only a sub-agent's
`conclude_task` summary crossed, and its raw exploration was unreachable. That
cost real information. Measured over N=3: a researcher read "Binary quantization
reduces RAM usage up to 32x", summarised deployment instead, and the detail was
gone — the parent hallucinated a number when later asked for it (0/3, against a
flat-store baseline's 3/3).

But the reason for isolation was never secrecy, it was noise: raw exploration
must not crowd the parent's working context. That is a *ranking* concern, so it
gets a ranking answer. Descendant facts are retrievable and demoted, so a
specific question can still reach a specific detail.

Sibling isolation is a separate guarantee and is unchanged: a child still reads
only itself plus its ancestors.
"""

from __future__ import annotations

from beads_memory.store import BeadsStore


def _store(conn):
    s = BeadsStore(conn)
    s.init_schema()
    return s


def _w(store, ns, body, embedder, kind="conclusion"):
    return store.write_fact(
        ns, kind=kind, body=body, source="remember_tool", source_key=body,
        agent_id="a", acting_on_behalf_of="user", embedding=embedder.embed(body),
    )


def test_parent_can_now_reach_a_childs_raw_finding(conn, embedder):
    """The measured failure: the detail exists in a child and the parent needs it."""
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    child = store.fork_namespace(root)
    _w(store, child, "Binary quantization reduces RAM usage up to 32x", embedder)

    hits = store.search(root.id, embedder.embed("what was the memory optimization?"), k=8)
    assert any("quantization" in h.body for h in hits), [h.body for h in hits]


def test_child_facts_rank_below_the_parents_own(conn, embedder):
    """Demotion, not exclusion: an equally relevant parent fact wins."""
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    child = store.fork_namespace(root)
    text = "the budget is 50k per year"
    _w(store, child, text, embedder)
    _w(store, root, text, embedder)

    hits = store.search(root.id, embedder.embed("budget"), k=8)
    assert hits[0].namespace_id == root.id, "parent fact must outrank an identical child fact"


def test_siblings_still_cannot_see_each_other(conn, embedder):
    """Unchanged guarantee. Descendant visibility is a parent privilege only."""
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    a = store.fork_namespace(root)
    b = store.fork_namespace(root)
    _w(store, a, "sibling A secret finding about quantization", embedder)

    hits = store.search(b.id, embedder.embed("quantization finding"), k=8)
    assert not any("sibling A" in h.body for h in hits), [h.body for h in hits]


def test_a_grandchilds_fact_is_reachable_from_the_root(conn, embedder):
    """Demotion applies to the whole subtree, not just direct children."""
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    child = store.fork_namespace(root)
    grand = store.fork_namespace(child)
    _w(store, grand, "deep finding: HNSW rebuild takes 50 minutes", embedder)

    hits = store.search(root.id, embedder.embed("how long does index rebuild take?"), k=8)
    assert any("HNSW rebuild" in h.body for h in hits), [h.body for h in hits]


def test_demotion_does_not_let_child_noise_crowd_out_constraints(conn, embedder):
    """The property isolation was protecting. Many loosely-related child facts
    must not displace a directly relevant parent constraint."""
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    child = store.fork_namespace(root)
    for i in range(12):
        _w(store, child, f"exploration note {i} about vector database internals", embedder)
    _w(store, root, "the budget is 50k per year", embedder)

    hits = store.search(root.id, embedder.embed("what is our budget?"), k=4)
    assert any("budget is 50k" in h.body for h in hits), [h.body for h in hits]
