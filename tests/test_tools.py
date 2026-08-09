from beads_memory.ids import short_id
from beads_memory.store import BeadsStore
from beads_memory.tools import make_conclude_task, make_remember_fact


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
