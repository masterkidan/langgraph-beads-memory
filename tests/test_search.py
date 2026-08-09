from beads_memory.store import BeadsStore


def _store(conn):
    s = BeadsStore(conn)
    s.init_schema()
    return s


def _write(store, ns, body, embedder, *, kind="conclusion", source="remember_tool", key=None):
    return store.write_fact(
        ns,
        kind=kind,
        body=body,
        source=source,
        source_key=key or body,
        agent_id="root",
        acting_on_behalf_of="user",
        embedding=embedder.embed(body),
    )


def test_search_ranks_relevant_first(conn, embedder):
    store = _store(conn)
    ns = store.get_or_create_namespace("s1")
    _write(store, ns, "the budget for the project is 100k dollars", embedder)
    _write(store, ns, "the team mascot is a heron", embedder)
    hits = store.search(ns.id, embedder.embed("what is the project budget"), k=2)
    assert hits[0].body.startswith("the budget")


def test_search_excludes_superseded(conn, embedder):
    store = _store(conn)
    ns = store.get_or_create_namespace("s1")
    old = _write(store, ns, "budget is 100k", embedder)
    new = _write(store, ns, "budget is 50k", embedder)
    store.add_edge(new.id, old.id, "supersedes")
    hits = store.search(ns.id, embedder.embed("budget"), k=10)
    bodies = [h.body for h in hits]
    assert "budget is 50k" in bodies and "budget is 100k" not in bodies


def test_search_reads_ancestors_not_siblings(conn, embedder):
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    child_a = store.fork_namespace(root)
    child_b = store.fork_namespace(root)
    _write(store, root, "root knows the goal", embedder)
    _write(store, child_a, "sibling A secret finding", embedder)
    hits = store.search(child_b.id, embedder.embed("secret finding goal"), k=10)
    bodies = [h.body for h in hits]
    assert "root knows the goal" in bodies
    assert "sibling A secret finding" not in bodies


def test_search_excludes_explicit_ids(conn, embedder):
    store = _store(conn)
    ns = store.get_or_create_namespace("s1")
    f = _write(store, ns, "budget is 100k", embedder)
    hits = store.search(ns.id, embedder.embed("budget"), k=10, exclude_ids=[f.id])
    assert all(h.id != f.id for h in hits)


def test_search_skips_unembedded_facts(conn, embedder):
    store = _store(conn)
    ns = store.get_or_create_namespace("s1")
    store.write_fact(
        ns,
        kind="user_input",
        body="unembedded budget note",
        source="passive_capture",
        source_key="m9",
        agent_id="root",
        acting_on_behalf_of="user",
    )  # no embedding
    hits = store.search(ns.id, embedder.embed("budget"), k=10)
    assert hits == []
