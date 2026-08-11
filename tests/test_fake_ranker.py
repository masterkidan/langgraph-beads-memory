"""The descendant penalty, tested against declared similarity.

FakeEmbedder cannot express "the child's fact is a better semantic match than
the parent's" — which is the exact situation the penalty exists to override. A
test that cannot state its own premise cannot verify the behaviour.
"""

from __future__ import annotations

import math

import pytest

from beads_memory.embeddings import FakeRanker
from beads_memory.store import DESCENDANT_RANK_PENALTY, BeadsStore


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb)


@pytest.mark.parametrize("target", [0.95, 0.7, 0.5, 0.0, -0.5])
def test_declared_similarity_is_exact(target):
    r = FakeRanker()
    assert _cos(r.query(), r.at(target)) == pytest.approx(target, abs=1e-9)


def test_two_facts_at_the_same_similarity_are_still_distinct():
    r = FakeRanker()
    a, b = r.at(0.8), r.at(0.8)
    assert a != b
    assert _cos(a, b) < 0.99


def test_rejects_impossible_similarity():
    with pytest.raises(ValueError):
        FakeRanker().at(1.5)


def _w(store, ns, body, vec):
    return store.write_fact(
        ns,
        kind="conclusion",
        body=body,
        source="remember_tool",
        source_key=body,
        agent_id="a",
        acting_on_behalf_of="user",
        embedding=vec,
    )


def test_penalty_demotes_a_child_that_is_the_better_match(conn):
    """The premise FakeEmbedder cannot state: the child fact matches the query
    *better*, and the penalty must still put the parent first."""
    store = BeadsStore(conn)
    store.init_schema()
    root = store.get_or_create_namespace("s1")
    child = store.fork_namespace(root)

    r = FakeRanker()
    q = r.query()
    _w(store, child, "child, closer match", r.at(0.90))
    _w(store, root, "parent, weaker match", r.at(0.80))

    # distances are 0.10 and 0.20; the 0.15 penalty must flip the order
    hits = store.search(root.id, q, k=2)
    assert hits[0].body.startswith("parent"), [h.body for h in hits]


def test_penalty_does_not_hide_a_decisively_better_child(conn):
    """Demotion, not exclusion. A child far closer than anything the parent has
    must still surface — that is what makes a buried detail recoverable."""
    store = BeadsStore(conn)
    store.init_schema()
    root = store.get_or_create_namespace("s1")
    child = store.fork_namespace(root)

    r = FakeRanker()
    q = r.query()
    _w(store, child, "child, decisively closer", r.at(0.99))
    _w(store, root, "parent, unrelated", r.at(0.10))

    hits = store.search(root.id, q, k=2)
    assert hits[0].body.startswith("child"), [h.body for h in hits]


def test_the_penalty_boundary_is_where_the_constant_says_it_is(conn):
    """Pins the documented trade-off: a child must beat the parent by more than
    DESCENDANT_RANK_PENALTY in cosine distance to outrank it."""
    store = BeadsStore(conn)
    store.init_schema()
    root = store.get_or_create_namespace("s1")
    child = store.fork_namespace(root)

    r = FakeRanker()
    q = r.query()
    parent_sim = 0.50  # distance 0.50
    just_under = parent_sim + DESCENDANT_RANK_PENALTY - 0.02
    _w(store, child, "child just under the threshold", r.at(just_under))
    _w(store, root, "parent", r.at(parent_sim))
    assert store.search(root.id, q, k=2)[0].body == "parent"
