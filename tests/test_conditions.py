"""Tests for the demo's condition wiring.

These are fairness tests. They cover repairs made to the BASELINE so that a
comparison measures memory architecture rather than a small model's inability
to satisfy one library's tool schema.
"""

import pytest


class TestPermissiveLangmemSchema:
    """Coercions that keep the BASELINE working, so the comparison stays fair.

    Measured cause: one baseline investigation issued 70 manage_memory calls
    and persisted 3 memories. Every rejection came from pydantic at the
    args_schema boundary, and the model re-invented the same bad argument each
    retry until the turn deadline fired.
    """

    def _schema(self):
        from langmem import create_manage_memory_tool

        from demo.conditions import _permissive_args_schema

        return _permissive_args_schema(create_manage_memory_tool(namespace=("m", "t")).args_schema)

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"action": "create", "content": "x", "id": "incident_1"},  # invented id
            {"action": "update", "content": "x", "id": "incident_summary"},
            {"action": "save", "content": "x"},  # invented action
            {"action": "create", "content": "x", "id": ""},
            {"action": "create", "content": {"p99": "4.2s"}},  # structured content
            {"action": "create", "content": ["a", "b"]},
        ],
    )
    def test_malformed_model_output_still_validates(self, kwargs):
        self._schema().model_validate(kwargs)

    def test_an_unusable_id_downgrades_update_to_create(self):
        m = self._schema().model_validate({"action": "update", "content": "x", "id": "not-a-uuid"})
        assert m.action == "create"
        assert m.id is None

    def test_a_real_uuid_update_is_left_alone(self):
        real = "c8b0a158-d10d-4951-a87f-0e07a7aff001"
        m = self._schema().model_validate({"action": "update", "content": "x", "id": real})
        assert m.action == "update"
        assert str(m.id) == real

    def test_structured_content_is_flattened_not_dropped(self):
        m = self._schema().model_validate(
            {"action": "create", "content": {"p99": "4.2s", "err": "7%"}}
        )
        assert "4.2s" in m.content and "7%" in m.content
