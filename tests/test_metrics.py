from demo.metrics import constraint_carry, token_usage


class _StubMsg:
    def __init__(self, usage_metadata=None):
        self.usage_metadata = usage_metadata


def test_constraint_carry_happy_path():
    final = (
        "I recommend pgvector: it fits the revised $50k budget, is "
        "self-hostable, and our primary benchmark data supports it."
    )
    buried = "Qdrant's binary quantization cut RAM up to 32x."
    c = constraint_carry(final, buried)
    assert all(c.values())


def test_constraint_carry_stale_budget_detected():
    final = "I recommend Weaviate; it fits the $100k budget."
    c = constraint_carry(final, "no idea")
    assert not c["uses_revised_budget"]
    assert not c["avoids_stale_budget_as_current"]
    assert not c["buried_detail_recalled"]


def test_constraint_carry_stale_and_revised_both_present_is_ok():
    # If the revised budget IS present alongside the stale one (e.g. explaining
    # the correction), that's not "presenting stale as current" - only flag
    # runs that mention $100k WITHOUT also mentioning the $50k revision.
    final = "The budget was revised from $100k to $50k; pgvector fits and is self-host."
    c = constraint_carry(final, "")
    assert c["uses_revised_budget"]
    assert c["avoids_stale_budget_as_current"]


def test_token_usage_sums_across_messages():
    messages = [
        _StubMsg(usage_metadata={"input_tokens": 10, "output_tokens": 5}),
        _StubMsg(usage_metadata=None),
        _StubMsg(usage_metadata={"input_tokens": 3, "output_tokens": 7}),
        object(),  # no usage_metadata attribute at all
    ]
    result = token_usage(messages)
    assert result == {"input_tokens": 13, "output_tokens": 12}


def test_token_usage_empty_list():
    assert token_usage([]) == {"input_tokens": 0, "output_tokens": 0}
