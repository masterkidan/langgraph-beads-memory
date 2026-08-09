from beads_memory.store import BeadsStore
from beads_memory.subagent import make_subagent_tool


def _setup(conn, embedder):
    store = BeadsStore(conn)
    store.init_schema()
    root = store.get_or_create_namespace("s1")
    return store, root


def test_subagent_conclusion_lands_in_parent(conn, embedder):
    store, root = _setup(conn, embedder)

    def build_agent(middleware, tools):
        def run(task: str) -> str:
            conclude = next(t for t in tools if t.name == "conclude_task")
            conclude.invoke({"summary": f"finished: {task}"})
            return "done"

        return run

    tool = make_subagent_tool(
        "researcher",
        "desc",
        store=store,
        parent_namespace=root,
        embedder=embedder,
        build_agent=build_agent,
    )
    out = tool.invoke({"task": "check qdrant"})
    assert "done" in out
    parent = store.facts_in_namespace(root.id)
    assert len(parent) == 1 and parent[0].source == "conclude_task"
    assert "finished: check qdrant" in parent[0].body


def test_lazy_subagent_gets_fallback_summary(conn, embedder):
    store, root = _setup(conn, embedder)

    def build_agent(middleware, tools):
        return lambda task: "I looked at things but forgot to conclude"

    tool = make_subagent_tool(
        "lazy",
        "desc",
        store=store,
        parent_namespace=root,
        embedder=embedder,
        build_agent=build_agent,
    )
    tool.invoke({"task": "t"})
    parent = store.facts_in_namespace(root.id)
    assert len(parent) == 1
    assert parent[0].source == "fallback_conclude"
    assert "forgot to conclude" in parent[0].body  # last output used as fallback


def test_crashed_subagent_leaves_unresolved_fact(conn, embedder):
    store, root = _setup(conn, embedder)

    def build_agent(middleware, tools):
        def run(task):
            raise RuntimeError("boom")

        return run

    tool = make_subagent_tool(
        "crashy",
        "desc",
        store=store,
        parent_namespace=root,
        embedder=embedder,
        build_agent=build_agent,
    )
    out = tool.invoke({"task": "t"})
    assert "did not complete" in out
    parent = store.facts_in_namespace(root.id)
    assert len(parent) == 1
    assert parent[0].source == "fallback_conclude"
    assert "did not complete" in parent[0].body


def test_parallel_forks_get_distinct_namespaces(conn, embedder):
    store, root = _setup(conn, embedder)
    seen = []

    def build_agent(middleware, tools):
        def run(task):
            seen.append(middleware.namespace.id)
            conclude = next(t for t in tools if t.name == "conclude_task")
            conclude.invoke({"summary": "ok"})
            return "ok"

        return run

    tool = make_subagent_tool(
        "r", "d", store=store, parent_namespace=root, embedder=embedder, build_agent=build_agent
    )
    tool.invoke({"task": "a"})
    tool.invoke({"task": "b"})
    assert len(set(seen)) == 2


def test_rollup_edges_point_at_child_facts_and_raw_facts_stay_in_child(conn, embedder):
    """The core audit-trail claim: a sub-agent's raw exploration facts stay in
    its own child namespace (never copied into the parent), and the parent's
    summary fact carries rollup_of edges pointing at each of them."""
    store, root = _setup(conn, embedder)

    def build_agent(middleware, tools):
        def run(task):
            remember = next(t for t in tools if t.name == "remember_fact")
            remember.invoke({"body": "qdrant supports hybrid search"})
            remember.invoke({"body": "qdrant has a rust core"})
            conclude = next(t for t in tools if t.name == "conclude_task")
            conclude.invoke({"summary": "qdrant looks like a good fit"})
            return "done"

        return run

    tool = make_subagent_tool(
        "researcher",
        "desc",
        store=store,
        parent_namespace=root,
        embedder=embedder,
        build_agent=build_agent,
    )
    tool.invoke({"task": "evaluate qdrant"})

    parent_facts = store.facts_in_namespace(root.id)
    assert len(parent_facts) == 1
    summary_fact = parent_facts[0]
    assert summary_fact.source == "conclude_task"

    # Find the child namespace via the fork: it's the only namespace besides
    # root that has facts, and its facts must NOT appear in the parent.
    rows = store._conn.execute(
        "SELECT DISTINCT namespace_id FROM facts WHERE namespace_id != %s", (root.id,)
    ).fetchall()
    assert len(rows) == 1
    child_ns_id = rows[0][0]
    child_facts = store.facts_in_namespace(child_ns_id)
    assert len(child_facts) == 2
    child_fact_ids = {f.id for f in child_facts}
    assert child_fact_ids.isdisjoint({f.id for f in parent_facts})

    # rollup_of edges from the summary fact must point at exactly the child facts.
    edge_rows = store._conn.execute(
        "SELECT to_fact_id FROM fact_edges WHERE from_fact_id = %s AND relation = 'rollup_of'",
        (summary_fact.id,),
    ).fetchall()
    edge_targets = {r[0] for r in edge_rows}
    assert edge_targets == child_fact_ids
