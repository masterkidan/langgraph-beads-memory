"""Splitting a message into individually-retrievable facts.

Motivated by measurement, not taste. Capturing a whole message as one fact gave
one embedding averaged across every topic in it, which had two consequences in a
real N=3 run:

1. A constraint stated once inside a longer sentence never reached the top-8 for
   a related question — its embedding was diluted by the other clauses.
2. A single bad `supersedes` edge retired three constraints at once, because
   they shared a row.

A similarity guard on `supersedes` was measured and rejected: against a bundled
fact, spurious edges scored HIGHER (0.63-0.74) than legitimate revisions
(0.45-0.68), so a threshold would block the correct edges and permit the wrong
ones. Splitting is what makes that guard viable later.
"""

from __future__ import annotations

import pytest

from beads_memory.segment import split_into_facts

CONSTRAINTS = (
    "We need to pick a vector database for our product. Constraints: the budget "
    "is $100k per year, it must be self-hostable, and I only trust primary "
    "benchmark data we measured ourselves, not vendor marketing numbers."
)


def test_the_measured_case_splits_into_separate_constraints():
    parts = split_into_facts(CONSTRAINTS)
    joined = " || ".join(parts)
    assert any("budget is $100k" in p for p in parts), joined
    assert any("self-hostable" in p for p in parts), joined
    assert any("primary benchmark data" in p for p in parts), joined
    # the three constraints must not share a fact
    budget = next(p for p in parts if "budget is $100k" in p)
    assert "self-hostable" not in budget and "primary benchmark" not in budget


def test_trailing_qualifier_stays_with_its_clause():
    """', not vendor marketing numbers' is a qualifier, not a constraint. It has
    no verb and must not become a standalone fact."""
    parts = split_into_facts(CONSTRAINTS)
    assert not any(p.strip().startswith("not vendor") for p in parts), parts
    primary = next(p for p in parts if "primary benchmark data" in p)
    assert "vendor marketing" in primary


def test_single_sentence_is_left_alone():
    text = "The budget is $50k per year, not $100k."
    assert split_into_facts(text) == [text]


def test_short_or_empty_input_is_not_shredded():
    assert split_into_facts("ok") == ["ok"]
    assert split_into_facts("") == []
    assert split_into_facts("   ") == []


def test_plain_prose_is_not_over_split():
    """Ordinary multi-clause prose without an enumeration should stay whole
    enough to remain meaningful."""
    text = "I think Qdrant looks promising, though I want to see the numbers first."
    parts = split_into_facts(text)
    assert len(parts) == 1, parts


def test_multi_sentence_splits_per_sentence():
    text = "The budget is $50k. It must be self-hostable. We deploy on AWS."
    parts = split_into_facts(text)
    assert len(parts) == 3
    assert parts[0].startswith("The budget")


def test_every_part_is_verbatim_substring_of_the_input():
    """Splitting must never paraphrase — the verbatim-capture guarantee holds
    per fragment, so an audit can still find the words the user actually used."""
    for part in split_into_facts(CONSTRAINTS):
        core = part.rstrip(".").strip()
        assert core in CONSTRAINTS, core


@pytest.mark.parametrize("sep", ["; ", ", and ", " and "])
def test_common_enumeration_separators(sep):
    text = f"The budget is $50k per year{sep}it must be self-hostable."
    parts = split_into_facts(text)
    assert len(parts) == 2, parts
