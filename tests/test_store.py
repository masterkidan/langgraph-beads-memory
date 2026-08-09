import pytest

from beads_memory.store import BeadsStore


def test_init_schema_idempotent(conn):
    store = BeadsStore(conn)
    store.init_schema()
    store.init_schema()  # must not raise


def test_get_or_create_root_namespace(conn):
    store = BeadsStore(conn)
    store.init_schema()
    ns1 = store.get_or_create_namespace("sess-1")
    ns2 = store.get_or_create_namespace("sess-1")
    assert ns1.id == ns2.id
    assert ns1.session_id == "sess-1"
    assert ns1.extra_path == []
    assert ns1.parent_id is None


def test_fork_namespace_creates_child(conn):
    store = BeadsStore(conn)
    store.init_schema()
    root = store.get_or_create_namespace("sess-1")
    child = store.fork_namespace(root)
    assert child.parent_id == root.id
    assert child.session_id == "sess-1"
    assert child.extra_path[0] == "task" and child.extra_path[1].startswith("sub-")


def test_ancestor_chain(conn):
    store = BeadsStore(conn)
    store.init_schema()
    root = store.get_or_create_namespace("sess-1")
    child = store.fork_namespace(root)
    grand = store.fork_namespace(child)
    chain = store.ancestor_chain(grand.id)
    assert chain == [grand.id, child.id, root.id]


def _mkstore(conn):
    s = BeadsStore(conn)
    s.init_schema()
    return s


def test_write_fact_idempotent(conn):
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("sess-1")
    f1 = store.write_fact(
        ns,
        kind="user_input",
        body="budget is 100k",
        source="passive_capture",
        source_key="msg-1",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    f2 = store.write_fact(
        ns,
        kind="user_input",
        body="budget is 100k",
        source="passive_capture",
        source_key="msg-1",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    assert f1.id == f2.id
    count = conn.execute("SELECT count(*) FROM facts").fetchone()[0]
    assert count == 1


def test_supersedes_edge_flips_status(conn):
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("sess-1")
    old = store.write_fact(
        ns,
        kind="user_input",
        body="budget is 100k",
        source="passive_capture",
        source_key="m1",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    new = store.write_fact(
        ns,
        kind="conclusion",
        body="budget is 50k",
        source="remember_tool",
        source_key="tc-1",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    store.add_edge(new.id, old.id, "supersedes")
    status = conn.execute("SELECT status FROM facts WHERE id=%s", (old.id,)).fetchone()[0]
    assert status == "superseded"


def test_relates_to_edge_does_not_touch_status(conn):
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("sess-1")
    a = store.write_fact(
        ns,
        kind="conclusion",
        body="A",
        source="remember_tool",
        source_key="t1",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    b = store.write_fact(
        ns,
        kind="conclusion",
        body="B",
        source="remember_tool",
        source_key="t2",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    store.add_edge(a.id, b.id, "relates_to")
    status = conn.execute("SELECT status FROM facts WHERE id=%s", (b.id,)).fetchone()[0]
    assert status == "active"


def test_facts_in_namespace(conn):
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("sess-1")
    other = store.fork_namespace(ns)
    store.write_fact(
        ns,
        kind="user_input",
        body="root fact",
        source="passive_capture",
        source_key="m1",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    store.write_fact(
        other,
        kind="conclusion",
        body="child fact",
        source="remember_tool",
        source_key="t1",
        agent_id="sub",
        acting_on_behalf_of="root",
    )
    bodies = [f.body for f in store.facts_in_namespace(other.id)]
    assert bodies == ["child fact"]


def test_resolve_short_id_unique_match(conn):
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("sess-1")
    fact = store.write_fact(
        ns,
        kind="user_input",
        body="the only fact",
        source="passive_capture",
        source_key="m1",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    from beads_memory.ids import short_id

    resolved = store.resolve_short_id(short_id(fact.id), [ns.id])
    assert resolved.id == fact.id


def test_resolve_short_id_no_match_raises(conn):
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("sess-1")
    with pytest.raises(LookupError):
        store.resolve_short_id("fact-deadbeef", [ns.id])


def test_resolve_short_id_ambiguous_raises(conn, monkeypatch):
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("sess-1")

    # Force two distinct facts to collide on the same 8-hex-char short id prefix
    # by monkeypatching derive_fact_id used inside write_fact.
    import uuid as uuid_mod

    shared_prefix = "aaaaaaaa"
    fixed_uuid_1 = uuid_mod.UUID(shared_prefix + "-0000-4000-8000-000000000001")
    fixed_uuid_2 = uuid_mod.UUID(shared_prefix + "-0000-4000-8000-000000000002")
    calls = iter([fixed_uuid_1, fixed_uuid_2])

    def fake_derive(session_id, namespace_id, source, source_key, body):
        return next(calls)

    monkeypatch.setattr("beads_memory.store.derive_fact_id", fake_derive)

    store.write_fact(
        ns,
        kind="user_input",
        body="one",
        source="passive_capture",
        source_key="m1",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    store.write_fact(
        ns,
        kind="user_input",
        body="two",
        source="passive_capture",
        source_key="m2",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    with pytest.raises(LookupError):
        store.resolve_short_id(f"fact-{shared_prefix}", [ns.id])


def _f(store, ns, body, embedder, kind="conclusion", key=None):
    return store.write_fact(
        ns,
        kind=kind,
        body=body,
        source="remember_tool",
        source_key=key or body,
        agent_id="a",
        acting_on_behalf_of="user",
        embedding=embedder.embed(body),
    )


def test_supersedes_rejected_when_facts_are_unrelated(conn, embedder):
    """A supersede must be *about* its target.

    Measured in a real run: "The investigation into Weaviate has been completed"
    was allowed to retire the user's stated constraints, silently removing them
    from retrieval. Nothing validated the relationship.
    """
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("s1")
    target = _f(store, ns, "the budget is 100k per year", embedder)
    unrelated = _f(store, ns, "zebras migrate across the savannah in herds", embedder)
    created = store.add_edge(unrelated.id, target.id, "supersedes")
    assert created is False
    status = conn.execute("SELECT status FROM facts WHERE id=%s", (target.id,)).fetchone()[0]
    assert status == "active", "an unrelated fact must not retire this one"
    n = conn.execute("SELECT count(*) FROM fact_edges WHERE relation='supersedes'").fetchone()[0]
    assert n == 0


def test_supersedes_allowed_when_facts_are_about_the_same_thing(conn, embedder):
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("s1")
    target = _f(store, ns, "the budget is 100k per year", embedder)
    revision = _f(store, ns, "the budget is 100k per year, revised to 50k", embedder)
    created = store.add_edge(revision.id, target.id, "supersedes")
    assert created is True
    status = conn.execute("SELECT status FROM facts WHERE id=%s", (target.id,)).fetchone()[0]
    assert status == "superseded"


def test_unguarded_relations_are_unaffected(conn, embedder):
    """Only `supersedes` retires a fact, so only it needs the guard."""
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("s1")
    a = _f(store, ns, "the budget is 100k per year", embedder)
    b = _f(store, ns, "zebras migrate across the savannah in herds", embedder)
    assert store.add_edge(b.id, a.id, "relates_to") is True


def test_supersedes_allowed_when_an_embedding_is_missing(conn, embedder):
    """Fail open: a fact written before embeddings existed must not become
    permanently un-supersedable."""
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("s1")
    target = store.write_fact(
        ns,
        kind="user_input",
        body="the budget is 100k",
        source="passive_capture",
        source_key="m1",
        agent_id="a",
        acting_on_behalf_of="user",
    )  # no embedding
    src = _f(store, ns, "the budget is now 50k", embedder)
    assert store.add_edge(src.id, target.id, "supersedes") is True
