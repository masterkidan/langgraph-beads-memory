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

from beads_memory.segment import (
    DIRECTIVE,
    STATEMENT,
    classify_fragment,
    split_into_facts,
)

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


class TestOpenerWordBoundaries:
    """A statement misfiled as a directive is held out of retrieval entirely,
    so this is the expensive direction to get wrong.

    Found on real data: "Checkout p99 latency went from 180ms to 4.2s" — the
    central fact of an incident — was classified DIRECTIVE because "checkout"
    starts with the imperative opener "check", making it unretrievable.
    """

    @pytest.mark.parametrize(
        "text",
        [
            "Checkout p99 latency went from 180ms to 4.2s",
            "Listing prices rose 12% last quarter",
            "Belgium is our primary market",  # "be"
            "Letters were sent to customers",  # "let"
            "Canary promotion happened at 13:50",  # "can"
            "Reviewers approved the change",  # "review"
            "Helpdesk tickets doubled",  # "help"
        ],
    )
    def test_word_that_merely_starts_with_an_opener_is_a_statement(self, text):
        assert classify_fragment(text) == STATEMENT

    @pytest.mark.parametrize(
        "text",
        [
            "Check the connection pool",
            "List the subsystems we ruled out",
            "Review the deploy timeline",
            "Help me understand the spike",
        ],
    )
    def test_the_actual_imperative_is_still_a_directive(self, text):
        assert classify_fragment(text) == DIRECTIVE

    @pytest.mark.parametrize(
        "text",
        [
            "Is the budget fixed",
            "Are we self-hosting",
            "Did the canary promote cleanly",
            "Be specific about the constraints",
            "Let's pick pgvector",
            "And remind me what we ruled out",
        ],
    )
    def test_openers_written_with_trailing_space_still_match(self, text):
        """Entries like "is ", "be " and the phrase "and remind" must survive
        the switch to whole-word matching — unpunctuated verb-initial questions
        are exactly why they were added."""
        assert classify_fragment(text) == DIRECTIVE
