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


def test_search_resolves_a_stale_match_to_the_current_value(conn, embedder):
    """A query matching the OLD wording returns the NEW value.

    This is the whole point of the DAG, and it is what neither excluding nor
    demoting superseded facts can do. Both of those merely REMOVE the stale row
    and leave the slot empty — measured as a no-op: on 78 LongMemEval
    knowledge-update questions, demoting produced byte-identical answers to
    excluding. The failure being fixed is answering "you currently have three
    bikes" when the user owns 4, and an empty slot does not fix it. Resolution
    replaces the match with the head of its chain.
    """
    store = _store(conn)
    ns = store.get_or_create_namespace("s1")
    old = _write(store, ns, "the budget is 100k per year", embedder)
    new = _write(store, ns, "the budget is 50k per year", embedder)
    store.add_edge(new.id, old.id, "supersedes")
    # deliberately phrased to match the SUPERSEDED fact
    bodies = [h.body for h in store.search(ns.id, embedder.embed("100k budget"), k=10)]
    assert "the budget is 50k per year" in bodies
    assert "the budget is 100k per year" not in bodies
    assert len(bodies) == len(set(bodies)), "versions must collapse, not duplicate"


def test_supersede_chain_collapses_to_one_row(conn, embedder):
    """Three versions of one claim occupy ONE slot, not three.

    Compactness is the property this library actually delivers, so a chain that
    spent three of eight slots restating one value would be a regression even if
    the newest ranked first.
    """
    store = _store(conn)
    ns = store.get_or_create_namespace("s1")
    v1 = _write(store, ns, "the budget is 100k per year", embedder)
    v2 = _write(store, ns, "the budget is 75k per year", embedder)
    v3 = _write(store, ns, "the budget is 50k per year", embedder)
    store.add_edge(v2.id, v1.id, "supersedes")
    store.add_edge(v3.id, v2.id, "supersedes")
    bodies = [h.body for h in store.search(ns.id, embedder.embed("budget"), k=10)]
    budget = [b for b in bodies if "budget is" in b]
    assert budget == ["the budget is 50k per year"], budget


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
