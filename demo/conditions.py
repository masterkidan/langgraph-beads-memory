"""Builds the two conditions. Only the memory layer differs."""

from __future__ import annotations

import pathlib
import uuid

import psycopg
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool, tool
from pgvector.psycopg import register_vector

from demo.llm import make_llm
from demo.scenario import RESEARCH_SYSTEM_PROMPT, SUBAGENT_SYSTEM_PROMPT

CORPUS = pathlib.Path(__file__).parent / "corpus"
DSN = "postgresql://beads:beads@localhost:5433/beads"


@tool
def read_document(name: str) -> str:
    """Read an evaluation document. Available: pgvector, qdrant, weaviate."""
    path = CORPUS / f"{name.lower().strip()}.md"
    if not path.exists():
        return f"No document named {name}. Available: pgvector, qdrant, weaviate"
    return path.read_text()


SUBTOPICS = ["pgvector", "qdrant", "weaviate"]


def _crash_safe(t: StructuredTool) -> StructuredTool:
    """Wrap a langmem tool so a bad call degrades to a ToolMessage instead of
    crashing the whole graph run.

    Discovered while wiring the baseline: langmem's manage_memory raises a bare
    ValueError from its own business logic (not a pydantic validation error) when
    qwen3:8b supplies id="" or id=None while action="create" - a mistake it makes
    routinely, since small local models tend to fill every optional schema field
    rather than omitting them. LangGraph's ToolNode.handle_tool_errors (the default
    create_agent wires up) only catches its own ToolInvocationError, and BaseTool's
    own handle_tool_error only catches langchain_core's ToolException - neither
    catches a plain ValueError raised inside a tool's function body, so it
    propagates and kills the run. beads_memory's remember_fact/conclude_task never
    raise for equivalent bad input (they catch LookupError internally and return an
    error string), so leaving this unguarded would crash the baseline on mistakes
    the treatment merely shrugs off and retries from - a robustness asymmetry with
    nothing to do with the memory architecture under test. This wrapper restores
    parity: any exception becomes ToolMessage content the agent can read and retry
    from, exactly like the treatment's tools already behave.
    """

    def _safe(**kwargs):
        try:
            return t.invoke(kwargs)
        except Exception as e:  # noqa: BLE001 - deliberately broad, see docstring
            return f"Error invoking tool '{t.name}' with kwargs {kwargs}: {e}"

    return StructuredTool.from_function(
        _safe,
        name=t.name,
        description=t.description,
        args_schema=t.args_schema,
    )


# ---------------------------------------------------------------- treatment
def build_treatment(session_id: str, run_schema: str):
    """beads-memory condition. Returns (invoke_fn, cleanup_fn). invoke_fn(thread_id,
    user_text) -> result. A fresh agent is built per call (new thread), but the SAME
    session_id namespace is reused - that is the cross-conversation memory claim."""
    from beads_memory import (
        BeadsMemoryMiddleware,
        BeadsStore,
        OllamaEmbedder,
        make_subagent_tool,
    )

    conn = psycopg.connect(DSN, autocommit=True)
    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{run_schema}"')
    conn.execute(f'SET search_path TO "{run_schema}", public')
    register_vector(conn)
    store = BeadsStore(conn)
    store.init_schema()
    embedder = OllamaEmbedder()
    root_ns = store.get_or_create_namespace(session_id)

    def make_researcher(topic: str):
        def build_agent(middleware, tools):
            # `tools` already contains middleware.tools (remember_fact) + conclude_task.
            # `middleware=[middleware]` below re-registers middleware.tools automatically,
            # so only pass the tools NOT already owned by the middleware to avoid
            # double-registering remember_fact.
            extra_tools = [t for t in tools if t not in middleware.tools]
            agent = create_agent(
                model=make_llm(),
                tools=[read_document] + extra_tools,
                system_prompt=SUBAGENT_SYSTEM_PROMPT + f" Your assigned topic is: {topic}.",
                middleware=[middleware],
            )

            def run(task: str) -> str:
                result = agent.invoke(
                    {"messages": [("user", task)]},
                    {"configurable": {"thread_id": f"sub-{uuid.uuid4()}"}},
                )
                return str(result["messages"][-1].content)

            return run

        return make_subagent_tool(
            f"researcher_{topic}",
            f"Delegate in-depth research on {topic} to a focused researcher.",
            store=store,
            parent_namespace=root_ns,
            embedder=embedder,
            build_agent=build_agent,
        )

    def invoke(thread_id: str, user_text: str) -> dict:
        middleware = BeadsMemoryMiddleware(
            store=store,
            namespace=root_ns,
            embedder=embedder,
            agent_id="root",
            acting_on_behalf_of="user",
        )
        agent = create_agent(
            model=make_llm(),
            tools=[read_document] + [make_researcher(t) for t in SUBTOPICS],
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            middleware=[middleware],
        )
        return agent.invoke(
            {"messages": [("user", user_text)]},
            {"configurable": {"thread_id": thread_id}},
        )

    return invoke, conn.close


# ----------------------------------------------------------------- baseline
def build_baseline(session_id: str, run_schema: str):
    """LangMem + PostgresStore condition: idiomatic supervisor, sub-agent results
    return as tool messages, LangMem store shared by all agents.

    `run_schema` is accepted for signature parity with build_treatment but is not
    used for storage isolation here: LangGraph's PostgresStore partitions data by
    namespace tuple (("memories", session_id)), not by Postgres schema, so a
    unique session_id is what isolates one demo run's memories from another's.
    """
    from langchain_ollama import OllamaEmbeddings
    from langgraph.store.postgres import PostgresStore
    from langmem import create_manage_memory_tool, create_search_memory_tool

    store_cm = PostgresStore.from_conn_string(
        DSN,
        index={"dims": 768, "embed": OllamaEmbeddings(model="nomic-embed-text")},
    )
    store = store_cm.__enter__()
    store.setup()
    ns = ("memories", session_id)
    mem_tools = [
        _crash_safe(create_manage_memory_tool(namespace=ns, store=store)),
        _crash_safe(create_search_memory_tool(namespace=ns, store=store)),
    ]

    def make_researcher(topic: str):
        researcher = create_agent(
            model=make_llm(),
            tools=[read_document] + mem_tools,
            system_prompt=SUBAGENT_SYSTEM_PROMPT
            + f" Your assigned topic is: {topic}."
            + " Save important findings with the memory tool.",
        )

        def _run(task: str) -> str:
            result = researcher.invoke({"messages": [("user", task)]})
            return str(result["messages"][-1].content)

        return StructuredTool.from_function(
            _run,
            name=f"researcher_{topic}",
            description=f"Delegate in-depth research on {topic} to a focused researcher.",
        )

    def invoke(thread_id: str, user_text: str) -> dict:
        agent = create_agent(
            model=make_llm(),
            tools=[read_document] + mem_tools + [make_researcher(t) for t in SUBTOPICS],
            system_prompt=RESEARCH_SYSTEM_PROMPT.replace(
                "remember_fact", "the manage_memory tool"
            ).replace(
                "relation='supersedes' and the old fact's short id from your " "Memory context",
                "an update to the existing memory",
            )
            + " Before answering, search your memory for relevant context.",
        )
        return agent.invoke(
            {"messages": [("user", user_text)]},
            {"configurable": {"thread_id": thread_id}},
        )

    def cleanup():
        store_cm.__exit__(None, None, None)

    return invoke, cleanup
