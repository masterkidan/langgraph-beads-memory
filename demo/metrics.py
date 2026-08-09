"""Objective metrics: no judge involved."""

from __future__ import annotations

from demo.scenario import PLANTED


def token_usage(messages: list) -> dict:
    """Sum usage_metadata across AI messages in a transcript."""
    tin = tout = 0
    for m in messages:
        usage = getattr(m, "usage_metadata", None)
        if usage:
            tin += usage.get("input_tokens", 0)
            tout += usage.get("output_tokens", 0)
    return {"input_tokens": tin, "output_tokens": tout}


def _any_variant(text: str, variants: list[str]) -> bool:
    """True if any surface form of a planted term appears."""
    return any(v.lower() in text for v in variants)


def constraint_carry(final_answer: str, buried_answer: str) -> dict:
    """Did planted constraints survive to conversation 3, with the REVISED
    budget (not the stale one) and the buried detail?

    Two corrections after inspecting the first real run (2026-08-08):

    1. `buried_detail_recalled` required the literal substring "32x", but the
       model wrote "32 times" — semantically correct, scored wrong. Now matches
       any surface form. This correction makes BOTH conditions score higher; it
       was found in a baseline run and it helps the baseline, so it is not a
       tilt toward the treatment.
    2. `pick_is_feasible` was renamed `mentions_feasible_option` because it
       overclaimed: a run that named pgvector and Qdrant while refusing to
       recommend anything ("if you have additional constraints, let me know")
       scored True. Substring matching cannot tell a recommendation from a
       mention. Whether the answer actually commits to a defensible pick is a
       judgment call, and it belongs to the LLM judge's `final` dimension —
       objective metrics stay literal and cheap.
    """
    fa = final_answer.lower()
    ba = buried_answer.lower()
    return {
        "uses_revised_budget": PLANTED["revised_budget"] in fa,
        "avoids_stale_budget_as_current": not (
            PLANTED["stale_budget"] in fa and PLANTED["revised_budget"] not in fa
        ),
        "mentions_selfhost": _any_variant(fa, PLANTED["selfhost_variants"]),
        "mentions_primary_sources": PLANTED["constraint_primary_sources"] in fa,
        "mentions_feasible_option": any(p in fa for p in PLANTED["expected_pick_one_of"]),
        "buried_detail_recalled": all(
            _any_variant(ba, variants) for variants in PLANTED["buried_detail_variants"]
        ),
    }
