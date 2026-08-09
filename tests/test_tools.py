from beads_memory.ids import short_id
from beads_memory.store import BeadsStore
from beads_memory.tools import _normalize_reference, make_conclude_task, make_remember_fact


def _setup(conn, embedder):
    store = BeadsStore(conn)
    store.init_schema()
    root = store.get_or_create_namespace("s1")
    return store, root


def test_remember_fact_writes_conclusion(conn, embedder):
    store, root = _setup(conn, embedder)
    tool = make_remember_fact(store, root, embedder, agent_id="root", acting_on_behalf_of="user")
    out = tool.invoke({"body": "pgvector fits the 50k budget"})
    assert out.startswith("Remembered [fact-")
    facts = store.facts_in_namespace(root.id)
    assert facts[0].kind == "conclusion" and facts[0].source == "remember_tool"


def test_remember_fact_supersedes_by_short_id(conn, embedder):
    store, root = _setup(conn, embedder)
    old = store.write_fact(
        root,
        kind="user_input",
        body="budget is 100k",
        source="passive_capture",
        source_key="m1",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    tool = make_remember_fact(store, root, embedder, agent_id="root", acting_on_behalf_of="user")
    tool.invoke(
        {
            "body": "budget is 50k",
            "relates_to": short_id(old.id),
            "relation": "supersedes",
        }
    )
    status = conn.execute("SELECT status FROM facts WHERE id=%s", (old.id,)).fetchone()[0]
    assert status == "superseded"


def test_remember_fact_bad_short_id_returns_error_string(conn, embedder):
    store, root = _setup(conn, embedder)
    tool = make_remember_fact(store, root, embedder, agent_id="root", acting_on_behalf_of="user")
    out = tool.invoke({"body": "x", "relates_to": "fact-zzzzzzzz", "relation": "relates_to"})
    assert "Error" in out
    assert store.facts_in_namespace(root.id) == []  # no fact written on bad ref


def test_conclude_task_writes_summary_to_parent_with_rollups(conn, embedder):
    store, root = _setup(conn, embedder)
    child = store.fork_namespace(root)
    exploration = store.write_fact(
        child,
        kind="conclusion",
        body="checked qdrant docs",
        source="remember_tool",
        source_key="t1",
        agent_id="sub-1",
        acting_on_behalf_of="root",
    )
    holder = {}
    tool = make_conclude_task(
        store,
        child,
        root,
        embedder,
        agent_id="sub-1",
        acting_on_behalf_of="root",
        concluded=holder,
    )
    tool.invoke({"summary": "qdrant needs 32GB RAM minimum"})
    assert "fact_id" in holder
    parent_facts = store.facts_in_namespace(root.id)
    assert parent_facts[0].kind == "summary" and parent_facts[0].source == "conclude_task"
    edge = conn.execute(
        "SELECT relation FROM fact_edges WHERE to_fact_id=%s", (exploration.id,)
    ).fetchone()
    assert edge[0] == "rollup_of"


def test_normalize_reference_none():
    assert _normalize_reference(None) == (None, None)


def test_normalize_reference_plain_string():
    assert _normalize_reference("fact-a3f8b2c1") == ("fact-a3f8b2c1", None)


def test_normalize_reference_single_element_list():
    assert _normalize_reference(["fact-a3f8b2c1"]) == ("fact-a3f8b2c1", None)


def test_normalize_reference_empty_list():
    assert _normalize_reference([]) == (None, None)


def test_normalize_reference_tuple():
    assert _normalize_reference(("fact-a3f8b2c1",)) == ("fact-a3f8b2c1", None)


def test_normalize_reference_dict_with_relation():
    assert _normalize_reference({"id": "fact-aaaa1111", "relation": "supersedes"}) == (
        "fact-aaaa1111",
        "supersedes",
    )


def test_normalize_reference_dict_alternate_keys():
    assert _normalize_reference({"fact_id": "fact-aaaa1111"}) == ("fact-aaaa1111", None)
    assert _normalize_reference({"short_id": "fact-aaaa1111"}) == ("fact-aaaa1111", None)
    assert _normalize_reference({"fact": "fact-aaaa1111"}) == ("fact-aaaa1111", None)


def test_normalize_reference_dict_unrecognized_key():
    assert _normalize_reference({"foo": "bar"}) == (None, None)


def test_remember_fact_relates_to_dict_with_nested_relation_measured_case(conn, embedder):
    """The exact real-world malformed shape qwen3:8b emitted three times at
    temperature 0: {'id': ..., 'relation': 'supersedes'} with NO separate
    relation arg."""
    store, root = _setup(conn, embedder)
    old = store.write_fact(
        root,
        kind="user_input",
        body="budget is 100k",
        source="passive_capture",
        source_key="m1",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    tool = make_remember_fact(store, root, embedder, agent_id="root", acting_on_behalf_of="user")
    out = tool.invoke(
        {
            "body": "the budget is 50k per year",
            "relates_to": {"id": short_id(old.id), "relation": "supersedes"},
        }
    )
    assert "Error" not in out
    status = conn.execute("SELECT status FROM facts WHERE id=%s", (old.id,)).fetchone()[0]
    assert status == "superseded"


def test_remember_fact_relates_to_single_element_list_with_explicit_relation(conn, embedder):
    store, root = _setup(conn, embedder)
    old = store.write_fact(
        root,
        kind="user_input",
        body="budget is 100k",
        source="passive_capture",
        source_key="m1",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    tool = make_remember_fact(store, root, embedder, agent_id="root", acting_on_behalf_of="user")
    out = tool.invoke(
        {
            "body": "budget is 50k",
            "relates_to": [short_id(old.id)],
            "relation": "supersedes",
        }
    )
    assert "Error" not in out
    status = conn.execute("SELECT status FROM facts WHERE id=%s", (old.id,)).fetchone()[0]
    assert status == "superseded"


def test_remember_fact_relates_to_unrecognized_dict_key_treated_as_none(conn, embedder):
    store, root = _setup(conn, embedder)
    tool = make_remember_fact(store, root, embedder, agent_id="root", acting_on_behalf_of="user")
    out = tool.invoke({"body": "a plain conclusion", "relates_to": {"foo": "bar"}})
    assert out.startswith("Remembered [fact-")
    facts = store.facts_in_namespace(root.id)
    assert len(facts) == 1 and facts[0].kind == "conclusion"
    edges = conn.execute("SELECT count(*) FROM fact_edges").fetchone()[0]
    assert edges == 0


def test_conclude_task_supersedes_as_dict(conn, embedder):
    store, root = _setup(conn, embedder)
    old = store.write_fact(
        root,
        kind="user_input",
        body="budget is 100k",
        source="passive_capture",
        source_key="m1",
        agent_id="root",
        acting_on_behalf_of="user",
    )
    child = store.fork_namespace(root)
    holder = {}
    tool = make_conclude_task(
        store,
        child,
        root,
        embedder,
        agent_id="sub-1",
        acting_on_behalf_of="root",
        concluded=holder,
    )
    out = tool.invoke(
        {
            "summary": "budget is now 50k",
            "supersedes": {"id": short_id(old.id), "relation": "supersedes"},
        }
    )
    assert "Error" not in out
    assert "fact_id" in holder
    status = conn.execute("SELECT status FROM facts WHERE id=%s", (old.id,)).fetchone()[0]
    assert status == "superseded"
    parent_facts = store.facts_in_namespace(root.id)
    summary_facts = [f for f in parent_facts if f.kind == "summary"]
    assert len(summary_facts) == 1 and summary_facts[0].body == "budget is now 50k"


def test_conclude_task_bad_supersedes_short_id_writes_no_summary(conn, embedder):
    store, root = _setup(conn, embedder)
    child = store.fork_namespace(root)
    store.write_fact(
        child,
        kind="conclusion",
        body="checked qdrant docs",
        source="remember_tool",
        source_key="t1",
        agent_id="sub-1",
        acting_on_behalf_of="root",
    )
    holder = {}
    tool = make_conclude_task(
        store,
        child,
        root,
        embedder,
        agent_id="sub-1",
        acting_on_behalf_of="root",
        concluded=holder,
    )
    out = tool.invoke({"summary": "qdrant needs 32GB RAM minimum", "supersedes": "fact-zzzzzzzz"})
    assert "Error" in out
    assert "fact_id" not in holder
    assert store.facts_in_namespace(root.id) == []


def test_remember_fact_same_body_twice_is_one_fact(conn, embedder):
    """Regression: remember_fact used a random uuid as source_key, so identical
    bodies produced different content-derived ids and duplicated rows. Observed
    in a real run: the model called remember_fact three times with byte-identical
    text and the store kept all three."""
    store, root = _setup(conn, embedder)
    tool = make_remember_fact(store, root, embedder, agent_id="root", acting_on_behalf_of="user")
    body = "The annual budget for the vector database is $50,000, not $100,000."
    first = tool.invoke({"body": body})
    second = tool.invoke({"body": body})
    third = tool.invoke({"body": body})
    assert first == second == third  # same short id returned each time
    facts = store.facts_in_namespace(root.id)
    assert len(facts) == 1, [f.body for f in facts]


def test_remember_fact_different_bodies_are_distinct_facts(conn, embedder):
    """Dedup must not collapse genuinely different conclusions."""
    store, root = _setup(conn, embedder)
    tool = make_remember_fact(store, root, embedder, agent_id="root", acting_on_behalf_of="user")
    tool.invoke({"body": "pgvector fits the budget."})
    tool.invoke({"body": "Qdrant fits the budget."})
    assert len(store.facts_in_namespace(root.id)) == 2


def test_remember_fact_same_body_different_namespaces_not_deduped(conn, embedder):
    """Dedup is scoped to a namespace: a sub-agent recording the same sentence
    in its own namespace is a separate fact from the parent's."""
    store, root = _setup(conn, embedder)
    child = store.fork_namespace(root)
    body = "Qdrant needs 32GB RAM."
    make_remember_fact(store, root, embedder, agent_id="root", acting_on_behalf_of="user").invoke(
        {"body": body}
    )
    make_remember_fact(store, child, embedder, agent_id="sub", acting_on_behalf_of="root").invoke(
        {"body": body}
    )
    assert len(store.facts_in_namespace(root.id)) == 1
    assert len(store.facts_in_namespace(child.id)) == 1
