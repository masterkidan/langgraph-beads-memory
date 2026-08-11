"""Explicit traversal: an orchestrator looking at what its sub-agents found.

Demoted descendant *search* is similarity-driven — a child fact surfaces only if
the query happens to match it. That is the right default, but it is a guess.

An orchestrator often knows something stronger: it delegated Qdrant to a
researcher, so when asked "what did the Qdrant researcher find?" it should be
able to *look* rather than hope cosine similarity cooperates. The measured
failure was exactly this shape — the detail existed in a child namespace and the
parent never reached it.
"""

from __future__ import annotations

from beads_memory.store import BeadsStore


def _store(conn):
    s = BeadsStore(conn)
    s.init_schema()
    return s


def _w(store, ns, body, embedder, agent_id="a", kind="conclusion"):
    return store.write_fact(
        ns,
        kind=kind,
        body=body,
        source="remember_tool",
        source_key=body,
        agent_id=agent_id,
        acting_on_behalf_of="root",
        embedding=embedder.embed(body),
    )


def test_children_lists_direct_sub_namespaces(conn):
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    a = store.fork_namespace(root)
    b = store.fork_namespace(root)
    ids = {n.id for n in store.children(root.id)}
    assert ids == {a.id, b.id}


def test_children_is_direct_only_not_the_whole_subtree(conn):
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    child = store.fork_namespace(root)
    grand = store.fork_namespace(child)
    assert {n.id for n in store.children(root.id)} == {child.id}
    assert {n.id for n in store.children(child.id)} == {grand.id}


def test_subtree_facts_returns_everything_beneath(conn, embedder):
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    q = store.fork_namespace(root)
    w = store.fork_namespace(root)
    _w(store, q, "qdrant: binary quantization reduces RAM up to 32x", embedder, "researcher_qdrant")
    _w(store, w, "weaviate: hybrid search is the standout feature", embedder, "researcher_weaviate")
    _w(store, root, "a parent fact", embedder, "root")

    bodies = [f.body for f in store.subtree_facts(root.id)]
    assert any("quantization" in b for b in bodies)
    assert any("hybrid search" in b for b in bodies)
    assert not any("a parent fact" in b for b in bodies), "subtree means below, not self"


def test_subtree_facts_filters_by_agent(conn, embedder):
    """The orchestrator knows which researcher it delegated to."""
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    q = store.fork_namespace(root)
    w = store.fork_namespace(root)
    _w(store, q, "qdrant: binary quantization 32x", embedder, "researcher_qdrant")
    _w(store, w, "weaviate: hybrid search", embedder, "researcher_weaviate")

    bodies = [f.body for f in store.subtree_facts(root.id, agent_id="researcher_qdrant")]
    assert bodies == ["qdrant: binary quantization 32x"]


def test_subtree_facts_excludes_retired_facts(conn, embedder):
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    child = store.fork_namespace(root)
    old = _w(store, child, "the budget is 100k per year", embedder)
    new = _w(store, child, "the budget is 100k per year, revised to 50k", embedder)
    store.add_edge(new.id, old.id, "supersedes")

    bodies = [f.body for f in store.subtree_facts(root.id)]
    assert "the budget is 100k per year" not in bodies


def test_a_child_cannot_traverse_into_a_sibling(conn, embedder):
    """Traversal is a parent privilege, exactly as descendant search is."""
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    a = store.fork_namespace(root)
    b = store.fork_namespace(root)
    _w(store, a, "sibling A private finding", embedder, "researcher_a")

    assert store.children(b.id) == []
    assert store.subtree_facts(b.id) == []


# --------------------------------------------------------------------------
# The agent-facing tool
# --------------------------------------------------------------------------


def test_recall_from_subagents_tool_finds_a_buried_detail(conn, embedder):
    """End-to-end shape of the measured failure: the researcher recorded the
    detail, its rollup omitted it, and the orchestrator is later asked for it."""
    from beads_memory.tools import make_recall_from_subagents

    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    q = store.fork_namespace(root)
    _w(store, q, "Binary quantization reduces RAM usage up to 32x", embedder, "researcher_qdrant")
    store.write_fact(
        root,
        kind="summary",
        body="Qdrant is self-hostable via a single binary",
        source="conclude_task",
        source_key="roll",
        agent_id="researcher_qdrant",
        acting_on_behalf_of="root",
        embedding=embedder.embed("qdrant deployment"),
    )

    tool = make_recall_from_subagents(store, root)
    out = tool.invoke({"agent_id": "researcher_qdrant"})
    assert "quantization" in out, out


def test_recall_tool_lists_available_subagents_when_asked_for_an_unknown_one(conn, embedder):
    """A wrong agent name should teach the model the right ones, not dead-end."""
    from beads_memory.tools import make_recall_from_subagents

    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    q = store.fork_namespace(root)
    _w(store, q, "some finding", embedder, "researcher_qdrant")

    out = make_recall_from_subagents(store, root).invoke({"agent_id": "nope"})
    assert "researcher_qdrant" in out, out


def test_recall_tool_reports_plainly_when_there_are_no_subagents(conn, embedder):
    from beads_memory.tools import make_recall_from_subagents

    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    out = make_recall_from_subagents(store, root).invoke({"agent_id": None})
    assert "no sub-agent" in out.lower(), out
