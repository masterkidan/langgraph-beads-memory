"""Fact-id collision tests across sessions, namespaces, and write paths.

Fact ids are content-derived so that a LangGraph checkpoint replay re-running a
capture hook is a no-op instead of a duplicate. That property is only safe if
two *genuinely different* facts can never derive the same id. These tests pin
down the boundary: what must collapse (true repeats) and what must never
collide (different sessions, namespaces, write paths, or adversarial text).
"""

from __future__ import annotations

import uuid

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from beads_memory.ids import derive_fact_id
from beads_memory.middleware import BeadsMemoryMiddleware
from beads_memory.store import BeadsStore
from beads_memory.tools import make_remember_fact

# --------------------------------------------------------------------------
# Pure derivation-level collision checks
# --------------------------------------------------------------------------

NS_A = uuid.UUID("00000000-0000-0000-0000-0000000000aa")
NS_B = uuid.UUID("00000000-0000-0000-0000-0000000000bb")


def test_same_inputs_are_stable():
    """The whole point: identical inputs must give an identical id."""
    a = derive_fact_id("sess-1", NS_A, "passive_capture", "msg-1", "hello")
    b = derive_fact_id("sess-1", NS_A, "passive_capture", "msg-1", "hello")
    assert a == b


def test_different_namespace_never_collides():
    a = derive_fact_id("sess-1", NS_A, "remember_tool", "remember", "same text")
    b = derive_fact_id("sess-1", NS_B, "remember_tool", "remember", "same text")
    assert a != b


def test_different_write_path_never_collides():
    """A user typing 'remember' and an agent calling remember_fact('remember')
    are different facts. Before `source` was part of the derivation these
    produced the same id, because passive capture falls back to using the body
    as its source_key when a message carries no id."""
    passive = derive_fact_id("sess-1", NS_A, "passive_capture", "remember", "remember")
    tool = derive_fact_id("sess-1", NS_A, "remember_tool", "remember", "remember")
    assert passive != tool


@pytest.mark.parametrize(
    ("key_a", "body_a", "key_b", "body_b"),
    [
        # Delimiter ambiguity: a naive f"{key}:{body}" join let a colon in the
        # key be re-parsed as part of the body.
        ("remember", "x:y", "remember:x", "y"),
        ("a", "b:c", "a:b", "c"),
        ("conclude", "abc:def", "conclude:abc", "def"),
        # Empty components must not blur into their neighbours.
        ("", "ab", "a", "b"),
        ("ab", "", "a", "b"),
    ],
)
def test_adversarial_separator_text_never_collides(key_a, body_a, key_b, body_b):
    a = derive_fact_id("sess-1", NS_A, "remember_tool", key_a, body_a)
    b = derive_fact_id("sess-1", NS_A, "remember_tool", key_b, body_b)
    assert a != b, f"({key_a!r},{body_a!r}) collided with ({key_b!r},{body_b!r})"


def test_no_collisions_across_a_large_generated_space():
    """Brute-force sweep over the component space, including separator-heavy
    strings, asserting the derivation is injective over it."""
    seen: dict[uuid.UUID, tuple] = {}
    sources = ("passive_capture", "remember_tool", "conclude_task", "fallback_conclude")
    keys = ("remember", "conclude", "m1", "m:1", "", ":", "a:b")
    bodies = ("", ":", "x", "x:y", "remember", "a:b:c", "budget is $50k")
    for ns in (NS_A, NS_B):
        for source in sources:
            for key in keys:
                for body in bodies:
                    combo = ("sess-1", ns, source, key, body)
                    fid = derive_fact_id(*combo)
                    assert fid not in seen, f"{combo} collided with {seen[fid]}"
                    seen[fid] = combo
    assert len(seen) == 2 * len(sources) * len(keys) * len(bodies)


# --------------------------------------------------------------------------
# Multi-session, end-to-end checks against real Postgres
# --------------------------------------------------------------------------


def _store(conn):
    s = BeadsStore(conn)
    s.init_schema()
    return s


def test_same_text_across_sessions_stays_separate(conn, embedder):
    """Two users (or two engagements) saying the identical sentence must be two
    facts. Sessions are the top-level memory scope; bleed here would be a
    cross-tenant data leak, not just noise."""
    store = _store(conn)
    body = "The annual budget is $50,000."
    ids = []
    for session in ("session-alpha", "session-beta", "session-gamma"):
        ns = store.get_or_create_namespace(session)
        tool = make_remember_fact(store, ns, embedder, agent_id="root", acting_on_behalf_of="user")
        tool.invoke({"body": body})
        facts = store.facts_in_namespace(ns.id)
        assert len(facts) == 1
        ids.append(facts[0].id)
    assert len(set(ids)) == 3, "identical text in different sessions must not merge"


def test_repeat_within_a_session_collapses_but_forks_stay_distinct(conn, embedder):
    """The intended dedup and the required isolation, together: a repeat inside
    one namespace collapses; the same sentence in a forked child does not."""
    store = _store(conn)
    root = store.get_or_create_namespace("session-1")
    child_a = store.fork_namespace(root)
    child_b = store.fork_namespace(root)
    body = "Qdrant uses binary quantization."

    for ns in (root, root, child_a, child_b):  # root twice on purpose
        make_remember_fact(store, ns, embedder, agent_id="a", acting_on_behalf_of="user").invoke(
            {"body": body}
        )

    assert len(store.facts_in_namespace(root.id)) == 1  # repeat collapsed
    assert len(store.facts_in_namespace(child_a.id)) == 1
    assert len(store.facts_in_namespace(child_b.id)) == 1
    all_ids = {store.facts_in_namespace(ns.id)[0].id for ns in (root, child_a, child_b)}
    assert len(all_ids) == 3, "sibling forks must not share a fact id"


def test_passive_capture_and_remember_tool_do_not_collide(conn, embedder):
    """End-to-end version of the cross-path case: a user message whose text is
    exactly 'remember' plus an agent conclusion with the same text."""
    store = _store(conn)
    ns = store.get_or_create_namespace("session-1")
    mw = BeadsMemoryMiddleware(
        store=store,
        namespace=ns,
        embedder=embedder,
        agent_id="root",
        acting_on_behalf_of="user",
    )
    # HumanMessage constructed without an explicit id -> source_key falls back
    # to the body, which is what made this collide.
    mw.before_model({"messages": [HumanMessage("remember")]}, None)
    make_remember_fact(store, ns, embedder, agent_id="root", acting_on_behalf_of="user").invoke(
        {"body": "remember"}
    )

    facts = store.facts_in_namespace(ns.id)
    assert len(facts) == 2, [(f.kind, f.source, f.body) for f in facts]
    assert {f.source for f in facts} == {"passive_capture", "remember_tool"}


def test_user_input_and_final_answer_with_identical_text_do_not_collide(conn, embedder):
    """An agent that echoes the user verbatim must not overwrite the user's
    fact: both are passive_capture, so only the message id separates them."""
    store = _store(conn)
    ns = store.get_or_create_namespace("session-1")
    mw = BeadsMemoryMiddleware(
        store=store,
        namespace=ns,
        embedder=embedder,
        agent_id="root",
        acting_on_behalf_of="user",
    )
    echoed = "Budget is $50,000."
    mw.before_model({"messages": [HumanMessage(echoed, id="u1")]}, None)
    mw.after_model({"messages": [AIMessage(echoed, id="a1")]}, None)

    facts = store.facts_in_namespace(ns.id)
    assert len(facts) == 2
    assert {f.kind for f in facts} == {"user_input", "conclusion"}


def test_replay_across_many_turns_is_idempotent(conn, embedder):
    """Simulated checkpoint replay: re-running the hook over a growing message
    list many times must never duplicate, across several conversations sharing
    one session."""
    store = _store(conn)
    ns = store.get_or_create_namespace("session-1")
    mw = BeadsMemoryMiddleware(
        store=store,
        namespace=ns,
        embedder=embedder,
        agent_id="root",
        acting_on_behalf_of="user",
    )
    messages: list = []
    for conversation in range(3):
        for turn in range(3):
            messages.append(HumanMessage(f"turn {turn}", id=f"c{conversation}-t{turn}"))
            for _ in range(3):  # replay the same state repeatedly
                mw.before_model({"messages": messages}, None)

    facts = store.facts_in_namespace(ns.id)
    assert len(facts) == 9, [f.body for f in facts]


# --------------------------------------------------------------------------
# Content-addressing: reproducibility and fingerprinting
# --------------------------------------------------------------------------


def test_namespace_id_is_derived_not_random(conn):
    """Namespace ids must be content-derived from (session_id, extra_path).

    Fact ids are derived from the namespace id, so a random namespace id makes
    the whole chain irreproducible: tear the database down, replay the identical
    conversation, and every fact gets a new id. Deriving the namespace id keeps
    the content-addressing property true end to end.
    """
    from beads_memory.ids import derive_namespace_id

    store = _store(conn)
    ns = store.get_or_create_namespace("session-repro")
    assert ns.id == derive_namespace_id("session-repro", [])


def test_facts_survive_a_full_rebuild_with_identical_ids(conn, embedder):
    """The reproducibility guarantee, end to end: same session, same input,
    same ids — even after the schema is dropped and recreated."""
    store = _store(conn)
    body = "The annual budget is $50,000."

    def write_once():
        ns = store.get_or_create_namespace("session-repro")
        make_remember_fact(store, ns, embedder, agent_id="root", acting_on_behalf_of="user").invoke(
            {"body": body}
        )
        return store.facts_in_namespace(ns.id)[0].id

    first = write_once()
    conn.execute("DROP TABLE fact_edges, facts, namespaces CASCADE")
    store.init_schema()
    assert write_once() == first


def test_fingerprint_is_sha256_and_stable():
    import hashlib

    from beads_memory.ids import fingerprint

    assert fingerprint("hello") == hashlib.sha256(b"hello").hexdigest()
    assert fingerprint("hello") == fingerprint("hello")
    assert fingerprint("hello") != fingerprint("hellO")


def test_passive_capture_never_uses_raw_body_as_key(conn, embedder):
    """A message without an id must key on a labelled content fingerprint, not
    on the raw body. Raw bodies are arbitrary text that collide with real keys."""
    store = _store(conn)
    ns = store.get_or_create_namespace("session-1")
    mw = BeadsMemoryMiddleware(
        store=store,
        namespace=ns,
        embedder=embedder,
        agent_id="root",
        acting_on_behalf_of="user",
    )
    # A body that looks exactly like another path's source key.
    mw.before_model({"messages": [HumanMessage("remember")]}, None)
    mw.before_model({"messages": [HumanMessage("remember")]}, None)  # replay
    facts = store.facts_in_namespace(ns.id)
    assert len(facts) == 1, "replay must still collapse"

    make_remember_fact(store, ns, embedder, agent_id="root", acting_on_behalf_of="user").invoke(
        {"body": "remember"}
    )
    assert len(store.facts_in_namespace(ns.id)) == 2


def test_long_body_does_not_bloat_the_derivation_input(conn):
    """Bodies are fingerprinted before derivation, so a megabyte of text costs
    the same as a sentence."""
    from beads_memory.ids import _canonical, fingerprint

    huge = "x" * 1_000_000
    assert len(fingerprint(huge)) == 64
    assert len(_canonical("s", "p", "k", fingerprint(huge))) < 200
