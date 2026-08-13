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
        ns,
        kind=kind,
        body=body,
        source="remember_tool",
        source_key=body,
        agent_id="a",
        acting_on_behalf_of="user",
        embedding=embedder.embed(body),
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


def test_one_sub_agent_cannot_monopolise_the_parents_slots(conn, embedder):
    """Delegation buys breadth, and similarity ranking alone does not protect it.

    This used to hold by accident — a sub-agent crossed into its parent as
    exactly one summary fact, so it could occupy exactly one slot. Splitting
    summaries per claim removed the accident, and the loss was immediate: in the
    final synthesis turn of an incident run the strongest-scoring researcher
    took two descendant slots in all four model calls while the other two
    researchers appeared in none, so the parent never saw their findings.
    Breadth went from 3 subsystems named to 2.
    """
    store = _store(conn)
    root = store.get_or_create_namespace("s-cap")
    loud = store.fork_namespace(root)
    quiet = store.fork_namespace(root)

    # The loud child records many near-identical claims about the query topic;
    # the quiet one records a single, less similar finding.
    def w(ns, body, agent):
        return store.write_fact(
            ns,
            kind="conclusion",
            body=body,
            source="remember_tool",
            source_key=body,
            agent_id=agent,
            acting_on_behalf_of="root",
            embedding=embedder.embed(body),
        )

    for i in range(6):
        w(loud, f"The database connection pool was exhausted, detail {i}.", "researcher_db")
    w(quiet, "The network path showed no packet loss.", "researcher_net")

    hits = store.search(root.id, embedder.embed("what did we find about the database pool?"), k=8)
    by_agent = {}
    for f in hits:
        if f.namespace_id != root.id:
            by_agent[f.namespace_id] = by_agent.get(f.namespace_id, 0) + 1

    assert by_agent.get(loud.id, 0) <= 2, "one child must not take the whole descendant budget"
    assert quiet.id in by_agent, "the quieter child must still reach the parent"


def test_a_split_summary_cannot_monopolise_the_parents_slots(conn, embedder):
    """The case a namespace-keyed cap misses entirely.

    `conclude_task` writes a sub-agent's summary into the PARENT namespace,
    tagged with the child's agent_id — so summary fragments are not descendant
    rows and carry no penalty. Once summaries were split per claim, one
    researcher's fragments could take several of the parent's slots while other
    researchers took none, which is how an incident run lost two of three
    subsystems from its final synthesis.
    """
    store = _store(conn)
    root = store.get_or_create_namespace("s-summary-cap")

    def summary(agent, body):
        return store.write_fact(
            root,
            kind="summary",
            body=body,
            source="conclude_task",
            source_key=body,
            agent_id=agent,
            acting_on_behalf_of="root",
            embedding=embedder.embed(body),
        )

    for i in range(6):
        summary("researcher_db", f"The database connection pool was exhausted, detail {i}.")
    summary("researcher_net", "The network path showed no packet loss.")

    hits = store.search(root.id, embedder.embed("what did we find about the database pool?"), k=8)
    agents = {}
    for f in hits:
        agents[f.agent_id] = agents.get(f.agent_id, 0) + 1
    assert agents.get("researcher_db", 0) <= 2, "one researcher took the whole floor"
    assert "researcher_net" in agents, "the quieter researcher must still be heard"
