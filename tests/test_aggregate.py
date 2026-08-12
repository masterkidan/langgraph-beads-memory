from demo.aggregate import format_table, rescore, summarize


def _run(condition, run, final, buried, tokens=None, errors=0):
    return {
        "condition": condition,
        "run": run,
        "transcript": [
            {"conversation": "conv-1", "user": "u", "final": "x"},
            {"conversation": "conv-3", "user": "q1", "final": final},
            {"conversation": "conv-3", "user": "q2", "final": buried},
        ],
        "tokens": tokens or {"input_tokens": 100, "output_tokens": 10},
        "errors": [{"e": "boom"}] * errors,
    }


GOOD = "Qdrant fits the revised $50,000 budget, is self-hosted, and uses primary data."
BURIED = "Binary quantization cut RAM up to 32 times."


def test_rescore_uses_current_metric_code_not_stored_snapshot():
    record = _run("treatment", 0, GOOD, BURIED)
    # a stale, wrong snapshot must be ignored in favour of recomputation
    record["constraint_carry"] = {"uses_revised_budget": False}
    assert rescore(record)["uses_revised_budget"] is True


def test_rescore_handles_missing_conv3():
    record = {"condition": "baseline", "run": 0, "transcript": [], "tokens": {}, "errors": []}
    scored = rescore(record)
    assert scored["uses_revised_budget"] is False


def test_summarize_counts_passes_and_averages_tokens():
    records = [
        _run("treatment", 0, GOOD, BURIED, {"input_tokens": 100, "output_tokens": 10}),
        _run(
            "treatment", 1, "no budget here", "nothing", {"input_tokens": 200, "output_tokens": 20}
        ),
        _run(
            "baseline",
            0,
            "no budget here",
            "nothing",
            {"input_tokens": 50, "output_tokens": 5},
            errors=2,
        ),
    ]
    s = summarize(records)
    assert s["treatment"]["n"] == 2
    # `metrics` holds the per-run values, not a pre-summed count: a numeric
    # metric (breadth_subsystems_named) has to average rather than count, so the
    # raw values are kept and the formatter decides how to read them.
    assert sum(s["treatment"]["metrics"]["uses_revised_budget"]) == 1  # 1 of 2 runs
    assert s["treatment"]["tokens"]["input_tokens"] == 150
    assert sum(s["baseline"]["metrics"]["uses_revised_budget"]) == 0
    assert s["baseline"]["errors"] == 2


def test_format_table_marks_differing_metrics():
    records = [_run("treatment", 0, GOOD, BURIED), _run("baseline", 0, "nothing", "nothing")]
    table = format_table(summarize(records))
    assert "| uses_revised_budget | 0/1 | 1/1 ←" in table
    assert "mean input tokens" in table


def test_format_table_flags_a_single_arm_rather_than_implying_a_comparison():
    """One arm still prints its numbers — useful when validating a new scenario
    at N=1 — but must say so, so a one-sided table is never read as a result."""
    table = format_table(summarize([_run("treatment", 0, GOOD, BURIED)]))
    assert "Only one arm present: treatment" in table
    assert "baseline" not in table


def test_format_table_handles_no_runs():
    assert format_table(summarize([])) == "No runs found."
