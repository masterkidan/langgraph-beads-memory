"""Classifying a captured fragment as a claim or a directive.

Questions and instructions are provenance — they explain why work happened —
so they must be captured and stay queryable. But they are not claims about the
world, and injecting them when answering a later question wastes a retrieval
slot on text that is nearly the query itself. Measured in a real run: four of
eight injected slots were question and instruction fragments, and the
constraint they displaced never reached the model.

Retention and retrievability are separate concerns; this is the classifier that
separates them.
"""

from __future__ import annotations

import pytest

from beads_memory.segment import DIRECTIVE, STATEMENT, classify_fragment

DIRECTIVES = [
    "We need to pick a vector database for our product.",
    "Please investigate our vector database options in depth.",
    "which vector database should we pick, why?",
    "Be specific about how it fits our constraints.",
    "And remind me — what was that big memory optimization the Qdrant researcher found?",
    "Delegate pgvector, Qdrant, and Weaviate to your researchers.",
    "Tell me what you found.",
    "Can you compare them on cost?",
]

STATEMENTS = [
    "the budget is $100k per year",
    "it must be self-hostable",
    "I only trust primary benchmark data we measured ourselves, not vendor marketing numbers.",
    "One correction before we wrap up: the budget is $50k per year, not $100k.",
    "Qdrant costs approximately $30,000/year.",
    "We deploy on AWS.",
]


@pytest.mark.parametrize("text", DIRECTIVES)
def test_directives(text):
    assert classify_fragment(text) == DIRECTIVE, text


@pytest.mark.parametrize("text", STATEMENTS)
def test_statements(text):
    assert classify_fragment(text) == STATEMENT, text


def test_the_measured_message_splits_into_a_goal_plus_three_claims():
    from beads_memory.segment import split_into_facts

    msg = (
        "We need to pick a vector database for our product. Constraints: the budget "
        "is $100k per year, it must be self-hostable, and I only trust primary "
        "benchmark data we measured ourselves, not vendor marketing numbers."
    )
    kinds = [classify_fragment(f) for f in split_into_facts(msg)]
    assert kinds == [DIRECTIVE, STATEMENT, STATEMENT, STATEMENT], kinds


def test_empty_is_a_directive_not_a_claim():
    """Degenerate input must never be treated as an assertable fact."""
    assert classify_fragment("") == DIRECTIVE
    assert classify_fragment("   ") == DIRECTIVE
