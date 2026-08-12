from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from beads_memory.middleware import BeadsMemoryMiddleware
from beads_memory.store import BeadsStore


def _mw(conn, embedder, window=3):
    store = BeadsStore(conn)
    store.init_schema()
    ns = store.get_or_create_namespace("s1")
    mw = BeadsMemoryMiddleware(
        store=store,
        namespace=ns,
        embedder=embedder,
        agent_id="root",
        acting_on_behalf_of="user",
        window=window,
    )
    return store, ns, mw


def test_before_model_captures_new_human_messages_idempotently(conn, embedder):
    store, ns, mw = _mw(conn, embedder)
    state = {"messages": [HumanMessage("budget is 100k", id="m1")]}
    mw.before_model(state, None)
    mw.before_model(state, None)  # replay — must not duplicate
    facts = store.facts_in_namespace(ns.id)
    assert len(facts) == 1
    assert facts[0].kind == "user_input" and facts[0].source == "passive_capture"


def test_after_model_captures_final_answer_not_toolcalls(conn, embedder):
    store, ns, mw = _mw(conn, embedder)
    toolcall_msg = AIMessage("", tool_calls=[{"name": "x", "args": {}, "id": "tc1"}], id="a1")
    mw.after_model({"messages": [toolcall_msg]}, None)
    assert store.facts_in_namespace(ns.id) == []  # tool-call turns are not conclusions
    final = AIMessage("I recommend pgvector.", id="a2")
    mw.after_model({"messages": [final]}, None)
    facts = store.facts_in_namespace(ns.id)
    assert len(facts) == 1 and facts[0].kind == "conclusion"


def test_after_model_disabled_for_non_root(conn, embedder):
    store, ns, mw = _mw(conn, embedder)
    mw.capture_final = False
    mw.after_model({"messages": [AIMessage("done", id="a1")]}, None)
    assert store.facts_in_namespace(ns.id) == []


def test_wrap_model_call_trims_window_and_injects_facts(conn, embedder):
    store, ns, mw = _mw(conn, embedder, window=3)
    store.write_fact(
        ns,
        kind="user_input",
        body="the budget is 100k dollars",
        source="passive_capture",
        source_key="old-1",
        agent_id="root",
        acting_on_behalf_of="user",
        embedding=embedder.embed("the budget is 100k dollars"),
    )
    msgs = [HumanMessage(f"filler {i}", id=f"m{i}") for i in range(5)]
    msgs.append(HumanMessage("what is the budget?", id="m-q"))
    captured = {}

    def handler(req):
        captured["messages"] = req.messages
        captured["system"] = req.system_message
        return "RESPONSE"

    req = SimpleNamespace(
        messages=msgs,
        system_message=SystemMessage("You are helpful."),
        override=lambda **kw: SimpleNamespace(
            messages=kw.get("messages", msgs),
            system_message=kw.get("system_message", SystemMessage("You are helpful.")),
        ),
    )
    result = mw.wrap_model_call(req, handler)
    assert result == "RESPONSE"
    assert len(captured["messages"]) == 3  # window trim, view-only
    sys_text = str(captured["system"].content)
    assert "fact-" in sys_text and "budget is 100k" in sys_text


def test_wrap_model_call_dedups_facts_still_in_window(conn, embedder):
    store, ns, mw = _mw(conn, embedder, window=5)
    msg = HumanMessage("the budget is 100k dollars", id="m1")
    mw.before_model({"messages": [msg]}, None)  # captured AND still in window
    captured = {}

    def handler(req):
        captured["system"] = req.system_message
        return "R"

    req = SimpleNamespace(
        messages=[msg],
        system_message=SystemMessage("sys"),
        override=lambda **kw: SimpleNamespace(
            messages=kw.get("messages", [msg]),
            system_message=kw.get("system_message", SystemMessage("sys")),
        ),
    )
    mw.wrap_model_call(req, handler)
    assert "budget is 100k" not in str(captured["system"].content)


def test_facts_still_injected_when_question_falls_outside_the_window(conn, embedder):
    """Regression: a long tool-calling turn must not silently disable memory.

    The retrieval query was taken from the *windowed* messages. On a turn with
    enough tool calls to push the user's question past the window, no
    HumanMessage remained in view, the query came back None, and NOT A SINGLE
    fact was injected — memory switched itself off on exactly the long,
    roundabout turns where it matters most. Observed at the boundary in a real
    run: one turn reached exactly the window size.
    """
    store, ns, mw = _mw(conn, embedder, window=3)
    store.write_fact(
        ns,
        kind="user_input",
        body="the budget is 100k dollars per year",
        source="passive_capture",
        source_key="old-1",
        agent_id="root",
        acting_on_behalf_of="user",
        embedding=embedder.embed("the budget is 100k dollars per year"),
    )
    # question first, then enough tool traffic to push it out of a 3-window
    msgs = [HumanMessage("what is the budget?", id="q1")]
    for i in range(5):
        msgs.append(
            AIMessage("", tool_calls=[{"name": "x", "args": {}, "id": f"tc{i}"}], id=f"a{i}")
        )
    captured = {}

    def handler(req):
        captured["system"] = req.system_message
        captured["messages"] = req.messages
        return "R"

    req = SimpleNamespace(
        messages=msgs,
        system_message=SystemMessage("sys"),
        override=lambda **kw: SimpleNamespace(
            messages=kw.get("messages", msgs),
            system_message=kw.get("system_message", SystemMessage("sys")),
        ),
    )
    mw.wrap_model_call(req, handler)

    assert len(captured["messages"]) == 3, "window trim still applies"
    assert not any(
        isinstance(m, HumanMessage) for m in captured["messages"]
    ), "precondition: the question must be outside the window for this test to mean anything"
    assert "budget is 100k" in str(
        captured["system"].content
    ), "facts must still be injected when the question is outside the window"


def test_split_facts_still_dedup_against_the_raw_window(conn, embedder):
    """Regression: the dedup exclusion must mirror how capture keys fragments.

    Capture splits a multi-constraint message into one fact per clause, keyed
    `<base>#<i>`. The exclusion list was still deriving a single id from the
    whole message, so none of the fragments matched and every one of them was
    re-injected into the system prompt while the message was still visible raw —
    the model reading the same content twice.
    """
    store, ns, mw = _mw(conn, embedder, window=5)
    msg = HumanMessage(
        "Constraints: the budget is $100k per year, it must be self-hostable, "
        "and I only trust primary benchmark data we measured ourselves.",
        id="u1",
    )
    mw.before_model({"messages": [msg]}, None)
    assert len(store.facts_in_namespace(ns.id)) >= 3, "precondition: message was split"

    captured = {}

    def handler(req):
        captured["system"] = req.system_message
        return "R"

    req = SimpleNamespace(
        messages=[msg],
        system_message=SystemMessage("sys"),
        override=lambda **kw: SimpleNamespace(
            messages=kw.get("messages", [msg]),
            system_message=kw.get("system_message", SystemMessage("sys")),
        ),
    )
    mw.wrap_model_call(req, handler)
    injected = str(captured["system"].content)
    assert "self-hostable" not in injected, injected
    assert "primary benchmark" not in injected, injected


def test_directives_are_captured_but_not_injected(conn, embedder):
    """Questions and instructions are provenance, so they must be stored and
    stay queryable — but they must not consume retrieval slots. They rank high
    against a query precisely because they resemble it; a measured run spent
    four of eight injected slots on question fragments, displacing the
    constraint the answer needed."""
    store, ns, mw = _mw(conn, embedder, window=2)
    mw.before_model(
        {
            "messages": [
                HumanMessage(
                    "We need to pick a vector database. Constraints: the budget is "
                    "$100k per year, and it must be self-hostable.",
                    id="u1",
                )
            ]
        },
        None,
    )
    facts = store.facts_in_namespace(ns.id)
    kinds = {f.kind for f in facts}
    assert "directive" in kinds, [(f.kind, f.body) for f in facts]
    assert "user_input" in kinds, [(f.kind, f.body) for f in facts]

    # the goal fragment is stored...
    assert any("We need to pick" in f.body for f in facts)
    # ...but default retrieval never returns it
    hits = store.search(ns.id, embedder.embed("which database should we pick?"), k=8)
    assert not any(h.kind == "directive" for h in hits), [h.body for h in hits]
    # ...and it is still reachable when explicitly asked for
    hits = store.search(
        ns.id,
        embedder.embed("which database should we pick?"),
        k=8,
        include_directives=True,
    )
    assert any(h.kind == "directive" for h in hits)


# --------------------------------------------------------------------------
# The agent's own answers were 58% of everything stored — 7 facts, 5,944
# chars, largest 1,925 — against 2,457 chars of actual sub-agent findings.
# They were captured WHOLE while user messages were split per claim, which is
# an inconsistency in our own capture logic, not a property of the domain.
# --------------------------------------------------------------------------


def test_final_answer_is_split_per_claim_like_a_user_message(conn, embedder):
    store, ns, mw = _mw(conn, embedder)
    answer = (
        "Connection pool exhaustion is ruled out. "
        "The fraud-scoring call introduced in 2.14 is the cause. "
        "We recommend disabling the feature flag."
    )
    mw.after_model({"messages": [AIMessage(answer, id="a1")]}, None)
    facts = store.facts_in_namespace(ns.id)
    assert len(facts) == 3, [f.body for f in facts]
    assert all(f.kind == "conclusion" for f in facts)
    # No fact is the whole blob: that is what averaged one embedding across
    # every topic and consumed most of a top-K injection budget.
    assert max(len(f.body) for f in facts) < len(answer)


def test_restating_a_conclusion_does_not_accumulate_copies(conn, embedder):
    """Keyed by content, not message id.

    An agent restating a conclusion it already reached is not new information.
    Keying on the message id meant every turn added another copy, which is how
    the store filled with the agent re-reading its own prose.
    """
    store, ns, mw = _mw(conn, embedder)
    claim = "The fraud-scoring call introduced in 2.14 is the cause."
    mw.after_model({"messages": [AIMessage(claim, id="a1")]}, None)
    mw.after_model({"messages": [AIMessage(claim, id="a2")]}, None)  # different message
    assert len(store.facts_in_namespace(ns.id)) == 1


def test_framing_in_an_answer_is_not_stored(conn, embedder):
    store, ns, mw = _mw(conn, embedder)
    mw.after_model(
        {"messages": [AIMessage("Here is the summary. New shift taking over.", id="a")]}, None
    )
    bodies = [f.body for f in store.facts_in_namespace(ns.id)]
    assert "New shift taking over." not in bodies
