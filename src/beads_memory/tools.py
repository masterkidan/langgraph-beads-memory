"""Agent-facing tools, bound per-namespace as closures."""

from __future__ import annotations

import uuid

from langchain_core.tools import tool

from .embeddings import Embedder
from .ids import short_id
from .store import BeadsStore, Namespace

_RELATIONS = ("supersedes", "contradicts", "relates_to")


def make_remember_fact(
    store: BeadsStore,
    namespace: Namespace,
    embedder: Embedder,
    *,
    agent_id: str,
    acting_on_behalf_of: str,
):
    @tool
    def remember_fact(body: str, relates_to: str | None = None, relation: str | None = None) -> str:
        """Durably remember a conclusion you have reached. Use relates_to (a short
        fact id like 'fact-a3f8b2c1' from your Memory context) with relation
        'supersedes' when this conclusion replaces an earlier fact, 'contradicts'
        or 'relates_to' otherwise."""
        target = None
        if relates_to is not None:
            if relation not in _RELATIONS:
                return f"Error: relation must be one of {_RELATIONS}"
            try:
                readable = store.ancestor_chain(namespace.id)
                target = store.resolve_short_id(relates_to, readable)
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
    @tool
    def conclude_task(summary: str, supersedes: str | None = None) -> str:
        """REQUIRED before you finish: report your task's conclusion. The summary
        is written to your parent's memory; your raw exploration stays in your own.
        Optionally pass supersedes=<short fact id> if your conclusion replaces an
        earlier fact."""
        target = None
        if supersedes is not None:
            try:
                readable = store.ancestor_chain(child_namespace.id)
                target = store.resolve_short_id(supersedes, readable)
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
