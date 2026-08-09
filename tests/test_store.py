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
