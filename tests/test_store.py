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
