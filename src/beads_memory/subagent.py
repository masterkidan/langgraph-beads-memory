"""Sub-agent fork/rollup wrapper: fork namespace, run, enforce a conclusion."""

from __future__ import annotations

from collections.abc import Callable

from langchain_core.tools import tool as tool_decorator

from .embeddings import Embedder
from .middleware import BeadsMemoryMiddleware
from .store import BeadsStore, Namespace
from .tools import make_conclude_task

# How many of a silent sub-agent's own facts to fold into a reconstructed
# summary. Enough to carry its findings, bounded so one chatty sub-agent
# cannot flood the parent's namespace with a single enormous fact.
_FALLBACK_FACT_LIMIT = 6


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
            elif str(output or "").strip():
                body = f"(auto-summary, agent did not conclude) {output}"
            else:
                # The sub-agent returned nothing AND never concluded. Reconstruct
                # from what it recorded in its own namespace — the reason that
                # namespace exists.
                #
                # MEASURED FAILURE this prevents: an investigator returned an
                # empty string, so the fallback wrote the literal body
                # "(auto-summary, agent did not conclude) " and the parent
                # received "" as the tool result. Knowing nothing, the parent
                # INVENTED a finding — "the application tier has been thoroughly
                # investigated and is not the cause" — which passive capture then
                # stored as a durable fact, and every later turn retrieved and
                # repeated it. A silent empty rollup is worse than a loud
                # failure: it is indistinguishable from a sub-agent that found
                # nothing worth reporting.
                recorded = [f.body for f in store.facts_in_namespace(child.id) if f.body.strip()]
                if recorded:
                    joined = " ".join(recorded[-_FALLBACK_FACT_LIMIT:])
                    body = (
                        f"(auto-summary, agent did not conclude) Recorded by " f"'{name}': {joined}"
                    )
                else:
                    body = (
                        f"Sub-agent '{name}' did NOT complete its task and recorded no "
                        f"findings. Its conclusions are MISSING — do not assume this "
                        f"subsystem was investigated or cleared."
                    )
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
        # An empty string is not a report. Returning it left the parent with
        # no information and no way to say so — which is exactly when it began
        # inventing findings. Hand back the synthesized body instead, so
        # "this sub-agent produced nothing" is something the parent can read.
        if str(output or "").strip():
            return output
        return body if "fact_id" not in concluded else f"Sub-agent '{name}' completed."

    _run.__name__ = name
    _run.__doc__ = description
    return tool_decorator(_run)


# v1 scope: tool-invoked sub-agents only. Handoff-style delegation
# (Command(goto=...)) is documented future work.
