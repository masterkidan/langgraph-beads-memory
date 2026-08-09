"""Agent-facing tools, bound per-namespace as closures."""

from __future__ import annotations

import uuid

from langchain_core.tools import tool
from pydantic import BaseModel, model_validator

from .embeddings import Embedder
from .ids import short_id
from .store import BeadsStore, Namespace

_RELATIONS = ("supersedes", "contradicts", "relates_to")

_REFERENCE_ID_KEYS = ("id", "fact_id", "short_id", "fact")


def _normalize_reference(value):
    """Coerce a reference argument into (short_id, relation_override).

    Small local models routinely nest a reference into a list or a dict
    (e.g. {"id": "fact-aaaa1111", "relation": "supersedes"}) instead of
    passing a plain short-id string, even when the schema advertises
    `str | None`. Be liberal in what we accept here; the public tool
    signatures stay strict so the JSON schema the model sees is unchanged.
    """
    if value is None:
        return None, None
    if isinstance(value, (list, tuple)):
        if not value:
            return None, None
        return _normalize_reference(value[0])
    if isinstance(value, dict):
        for key in _REFERENCE_ID_KEYS:
            if key in value:
                return value[key], value.get("relation")
        return None, None
    return value, None


def _coerce_reference_field(data, field: str, *, merge_relation: bool = False):
    """Shared `model_validator(mode="before")` body for both args schemas.

    Pydantic validates raw tool-call args against the args_schema BEFORE our
    function body ever runs, so a dict/list-shaped reference must be coerced
    here or `tool.invoke(...)` raises a ValidationError before `_normalize_
    reference` in the function body gets a chance to run. This runs at the
    whole-input level (not a single-field validator) because a nested
    relation override needs to flow into the sibling `relation` field. The
    JSON schema advertised to the model is derived from the field
    annotations (`str | None`), which this leaves untouched.
    """
    if not isinstance(data, dict):
        return data
    data = dict(data)
    ref_id, relation_override = _normalize_reference(data.get(field))
    data[field] = ref_id
    if merge_relation and data.get("relation") is None and relation_override is not None:
        data["relation"] = relation_override
    return data


class _RememberFactArgs(BaseModel):
    body: str
    relates_to: str | None = None
    relation: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        return _coerce_reference_field(data, "relates_to", merge_relation=True)


class _ConcludeTaskArgs(BaseModel):
    summary: str
    supersedes: str | None = None

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, data):
        return _coerce_reference_field(data, "supersedes")


def make_remember_fact(
    store: BeadsStore,
    namespace: Namespace,
    embedder: Embedder,
    *,
    agent_id: str,
    acting_on_behalf_of: str,
):
    @tool(args_schema=_RememberFactArgs)
    def remember_fact(body: str, relates_to: str | None = None, relation: str | None = None) -> str:
        """Durably remember a conclusion you have reached. Use relates_to (a short
        fact id like 'fact-a3f8b2c1' from your Memory context) with relation
        'supersedes' when this conclusion replaces an earlier fact, 'contradicts'
        or 'relates_to' otherwise."""
        ref_id, relation_override = _normalize_reference(relates_to)
        if relation is None:
            relation = relation_override
        target = None
        if ref_id is not None:
            if relation not in _RELATIONS:
                return f"Error: relation must be one of {_RELATIONS}"
            try:
                readable = store.ancestor_chain(namespace.id)
                target = store.resolve_short_id(ref_id, readable)
            except LookupError as e:
                return f"Error: {e}"
        fact = store.write_fact(
            namespace,
            kind="conclusion",
            body=body,
            source="remember_tool",
            source_key=f"remember:{uuid.uuid4()}",
            agent_id=agent_id,
            acting_on_behalf_of=acting_on_behalf_of,
            embedding=embedder.embed(body),
        )
        if target is not None:
            store.add_edge(fact.id, target.id, relation)
        return f"Remembered [{short_id(fact.id)}]"

    return remember_fact


def make_conclude_task(
    store: BeadsStore,
    child_namespace: Namespace,
    parent_namespace: Namespace,
    embedder: Embedder,
    *,
    agent_id: str,
    acting_on_behalf_of: str,
    concluded: dict,
):
    @tool(args_schema=_ConcludeTaskArgs)
    def conclude_task(summary: str, supersedes: str | None = None) -> str:
        """REQUIRED before you finish: report your task's conclusion. The summary
        is written to your parent's memory; your raw exploration stays in your own.
        Optionally pass supersedes=<short fact id> if your conclusion replaces an
        earlier fact."""
        ref_id, _relation_override = _normalize_reference(supersedes)
        target = None
        if ref_id is not None:
            try:
                readable = store.ancestor_chain(child_namespace.id)
                target = store.resolve_short_id(ref_id, readable)
            except LookupError as e:
                return f"Error: {e}"
        fact = store.write_fact(
            parent_namespace,
            kind="summary",
            body=summary,
            source="conclude_task",
            source_key=f"conclude:{child_namespace.id}",
            agent_id=agent_id,
            acting_on_behalf_of=acting_on_behalf_of,
            embedding=embedder.embed(summary),
        )
        for child_fact in store.facts_in_namespace(child_namespace.id):
            store.add_edge(fact.id, child_fact.id, "rollup_of")
        if target is not None:
            store.add_edge(fact.id, target.id, "supersedes")
        concluded["fact_id"] = fact.id
        return f"Task concluded [{short_id(fact.id)}]"

    return conclude_task
