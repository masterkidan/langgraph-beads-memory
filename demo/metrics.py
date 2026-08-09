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


def constraint_carry(final_answer: str, buried_answer: str) -> dict:
    """Did planted constraints survive to conversation 3, with the REVISED
    budget (not the stale one) and the buried detail?"""
    fa = final_answer.lower()
    ba = buried_answer.lower()
    return {
        "uses_revised_budget": PLANTED["revised_budget"] in fa,
        "avoids_stale_budget_as_current": not (
            PLANTED["stale_budget"] in fa and PLANTED["revised_budget"] not in fa
        ),
        "mentions_selfhost": PLANTED["constraint_selfhost"] in fa,
        "mentions_primary_sources": PLANTED["constraint_primary_sources"] in fa,
        "pick_is_feasible": any(p in fa for p in PLANTED["expected_pick_one_of"]),
        "buried_detail_recalled": all(
            t in ba for t in [x.lower() for x in PLANTED["buried_detail_terms"]]
        ),
    }
