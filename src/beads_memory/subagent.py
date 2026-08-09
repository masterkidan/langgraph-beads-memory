"""Sub-agent fork/rollup wrapper: fork namespace, run, enforce a conclusion."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import tool as tool_decorator

from .embeddings import Embedder
from .middleware import BeadsMemoryMiddleware
from .store import BeadsStore, Namespace
from .tools import make_conclude_task


def make_subagent_tool(
    name: str,
    description: str,
    *,
    store: BeadsStore,
    parent_namespace: Namespace,
    embedder: Embedder,
    build_agent: Callable,  # (middleware, tools) -> Callable[[str], str]
    parent_agent_id: str = "root",
):
    def _run(task: str) -> str:
        child = store.fork_namespace(parent_namespace)
        concluded: dict = {}
        middleware = BeadsMemoryMiddleware(
            store=store,
            namespace=child,
            embedder=embedder,
            agent_id=name,
            acting_on_behalf_of=parent_agent_id,
            capture_final=False,  # sub-agents conclude via conclude_task, not passively
        )
        conclude = make_conclude_task(
            store,
            child,
            parent_namespace,
            embedder,
            agent_id=name,
            acting_on_behalf_of=parent_agent_id,
            concluded=concluded,
        )
        tools = list(middleware.tools) + [conclude]
        agent = build_agent(middleware, tools)
        try:
            output = agent(task)
        except Exception as e:  # noqa: BLE001 - a crashing sub-agent must never propagate
            output = None
            error = str(e)
        else:
            error = None

        if "fact_id" not in concluded:
            # Enforced rollup: synthesize the summary the sub-agent failed to write.
            if error is not None:
                body = f"Task did not complete: sub-agent '{name}' failed with: {error}"
            else:
                body = f"(auto-summary, agent did not conclude) {output}"
            fact = store.write_fact(
                parent_namespace,
                kind="summary",
                body=body,
                source="fallback_conclude",
                source_key=f"fallback:{child.id}",
                agent_id=name,
                acting_on_behalf_of=parent_agent_id,
                embedding=embedder.embed(body),
            )
            for child_fact in store.facts_in_namespace(child.id):
                store.add_edge(fact.id, child_fact.id, "rollup_of")
            if error is not None:
                return f"Sub-agent '{name}' did not complete: {error}"
        return output if output is not None else f"Sub-agent '{name}' did not complete."

    _run.__name__ = name
    _run.__doc__ = description
    return tool_decorator(_run)


# v1 scope: tool-invoked sub-agents only. Handoff-style delegation
# (Command(goto=...)) is documented future work.
