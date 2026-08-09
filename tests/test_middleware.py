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
