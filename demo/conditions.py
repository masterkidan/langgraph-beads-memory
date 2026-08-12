"""Builds the two conditions. Only the memory layer differs."""

from __future__ import annotations

import contextlib
import os
import threading
import uuid

import psycopg
from langchain.agents import create_agent
from langchain_core.tools import StructuredTool, tool
from pgvector.psycopg import register_vector
from pydantic import model_validator

from demo.llm import make_llm
from demo.scenarios import Arm, Scenario, get_arm, get_scenario

DSN = "postgresql://beads:beads@localhost:5433/beads"


def make_read_document(scenario: Scenario):
    """Corpus reader bound to one scenario's document set.

    Scenario-scoped rather than global so demo 2's investigator cannot read
    demo 1's corpus, and so the tool's own description lists the right
    documents — a wrong list here silently costs a sub-agent its source
    material.
    """
    available = ", ".join(scenario.subtopics)

    @tool
    def read_document(name: str) -> str:
        """Read an investigation document."""
        path = scenario.corpus_dir / f"{name.lower().strip()}.md"
        if not path.exists():
            return f"No document named {name}. Available: {available}"
        return path.read_text()

    read_document.description = f"Read an investigation document. Available: {available}."
    return read_document


# LangGraph's ToolNode dispatches a message's tool calls across a thread pool,
# so the three researchers hit Ollama simultaneously. That reproducibly wedges
# the server: two separate N=3 attempts stalled with the client blocked, Ollama
# idle, and five or six connections open — no error, no progress, and it stayed
# wedged until a restart (a fresh request got no response in 45s, versus ~6s on
# a healthy server). Client-side timeouts do not rescue this; see demo/llm.py.
#
# get_executor_for_config honours max_concurrency, so 1 serialises tool calls
# and avoids the condition entirely. The cost is real — profiling measured
# ~2.4x effective concurrency — but a run that finishes slowly beats a run that
# hangs, and the earlier microbenchmark put the throughput gain at only ~1.05x
# because the GPU is already saturated by a single stream.
MAX_CONCURRENCY = int(os.environ.get("BEADS_DEMO_MAX_CONCURRENCY", "1"))


def _config(thread_id: str, callbacks: list | None = None) -> dict:
    """Run config. `callbacks` is how the profiler observes a run; it is None in
    normal operation, so this adds no overhead to the benchmark itself."""
    config: dict = {
        "configurable": {"thread_id": thread_id},
        "max_concurrency": MAX_CONCURRENCY,
    }
    if callbacks:
        config["callbacks"] = callbacks
    return config


class _ThreadLocalConnection:
    """A psycopg connection per thread, sharing one schema.

    The demo previously handed a single connection to the root agent and every
    sub-agent. psycopg3 guards a connection with an internal lock, so concurrent
    use does not corrupt anything — it serialises. But LangGraph's ToolNode runs
    tool calls on a thread pool, and a worker holding that lock while blocked on
    a slow operation leaves the main thread parked in
    `lock_PyThread_acquire_lock` indefinitely. Observed repeatedly: the run hung
    on the delegation turn with the main thread and a worker both waiting on
    locks, and no in-process deadline could break it — Python only runs signal
    handlers between bytecodes, which a blocking lock acquire never yields.

    Giving each thread its own connection removes the contention entirely.
    Each new connection re-applies the search_path and pgvector registration,
    so every thread sees the same schema.
    """

    def __init__(self, dsn: str, schema: str):
        self._dsn = dsn
        self._schema = schema
        self._local = threading.local()

    def _get(self):
        conn = getattr(self._local, "conn", None)
        if conn is None or conn.closed:
            conn = psycopg.connect(self._dsn, autocommit=True)
            conn.execute(f'SET search_path TO "{self._schema}", public')
            register_vector(conn)
            self._local.conn = conn
            with self._lock_all:
                self._all.append(conn)
        return conn

    _all: list = []
    _lock_all = threading.Lock()

    def execute(self, *args, **kwargs):
        return self._get().execute(*args, **kwargs)

    def close(self):
        with self._lock_all:
            for c in self._all:
                with contextlib.suppress(Exception):  # a dead connection is fine
                    c.close()
            self._all.clear()

    def __getattr__(self, name):
        return getattr(self._get(), name)


def _permissive_args_schema(base):
    """Coerce malformed `manage_memory` args before pydantic rejects them.

    A FAIRNESS REPAIR, stated plainly because it helps the baseline.

    Measured on a real conv-1: three `manage_memory` calls, three failures,
    **zero** memories saved. A baseline that stores nothing has nothing to
    recall, which would hand the treatment a win that says nothing about memory
    architecture — precisely the strawman this project already had to correct
    once in its own diagrams. The two failure shapes, both from qwen3:8b:

        content={"p99": "4.2s", ...}   -> "Input should be a valid string"
        action="create", id=""         -> "Input should be a valid UUID"

    Both are rejected by pydantic at the args_schema boundary, so a wrapper
    around the tool *function* (see `_crash_safe`) never runs and cannot help.
    The coercion has to happen in a `model_validator(mode="before")`, which is
    exactly how `beads_memory.tools` handles the identical class of malformed
    input for the treatment's own tools — a model nesting a value in a dict
    where a string was expected. Repairing one side and not the other is what
    would be unfair; the schema mismatch belongs to the model, not to either
    memory architecture.

    Structured `content` is flattened to "key: value" lines rather than JSON,
    because the value is embedded for semantic search and prose embeds better
    than braces. Nothing is discarded.
    """

    class Permissive(base):
        @model_validator(mode="before")
        @classmethod
        def _coerce(cls, data):
            if not isinstance(data, dict):
                return data
            data = dict(data)
            # An absent id is valid; an empty-string id is not a UUID. Dropping
            # it lets `create` succeed, and leaves `update`/`delete` to fail in
            # langmem's own logic, where _crash_safe turns it into a readable
            # ToolMessage the agent can retry from.
            if not data.get("id"):
                data.pop("id", None)
            content = data.get("content")
            if isinstance(content, dict):
                data["content"] = "\n".join(f"{k}: {v}" for k, v in content.items())
            elif isinstance(content, (list, tuple)):
                data["content"] = "\n".join(str(item) for item in content)
            return data

    Permissive.__name__ = base.__name__
    Permissive.__doc__ = base.__doc__
    return Permissive


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
        args_schema=_permissive_args_schema(t.args_schema),
    )


# ---------------------------------------------------------------- treatment
def build_treatment(session_id: str, run_schema: str, scenario: Scenario, arm: Arm):
    """beads-memory condition. Returns (invoke_fn, cleanup_fn). invoke_fn(thread_id,
    user_text) -> result. A fresh agent is built per call (new thread), but the SAME
    session_id namespace is reused - that is the cross-conversation memory claim.

    `arm` selects the ablation: `retire_superseded` controls whether a supersedes
    edge actually retires its target, and `subrecall` appends the instruction that
    makes `recall_from_subagents` reachable."""
    from beads_memory import (
        BeadsMemoryMiddleware,
        BeadsStore,
        OllamaEmbedder,
        make_subagent_tool,
    )

    read_document = make_read_document(scenario)
    root_prompt = scenario.root_prompt
    if arm.subrecall:
        from demo.scenario_incident import SUBRECALL_PROMPT_SUFFIX

        root_prompt += SUBRECALL_PROMPT_SUFFIX

    bootstrap = psycopg.connect(DSN, autocommit=True)
    bootstrap.execute(f'CREATE SCHEMA IF NOT EXISTS "{run_schema}"')
    bootstrap.close()
    # Per-thread connections: sub-agents run on ToolNode's thread pool and a
    # shared connection deadlocks them on psycopg's internal lock (see above).
    conn = _ThreadLocalConnection(DSN, run_schema)
    store = BeadsStore(conn, retire_superseded=arm.retire_superseded)
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
                system_prompt=scenario.subagent_prompt + f" Your assigned topic is: {topic}.",
                middleware=[middleware],
            )

            def run(task: str) -> str:
                result = agent.invoke(
                    {"messages": [("user", task)]},
                    _config(f"sub-{uuid.uuid4()}", None),
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

    def invoke(thread_id: str, user_text: str, callbacks: list | None = None) -> dict:
        middleware = BeadsMemoryMiddleware(
            store=store,
            namespace=root_ns,
            embedder=embedder,
            agent_id="root",
            acting_on_behalf_of="user",
        )
        agent = create_agent(
            model=make_llm(),
            tools=[read_document] + [make_researcher(t) for t in scenario.subtopics],
            system_prompt=root_prompt,
            middleware=[middleware],
        )
        return agent.invoke(
            {"messages": [("user", user_text)]},
            _config(thread_id, callbacks),
        )

    return invoke, conn.close


# ----------------------------------------------------------------- baseline
def build_baseline(session_id: str, run_schema: str, scenario: Scenario, arm: Arm):
    """LangMem + PostgresStore condition: idiomatic supervisor, sub-agent results
    return as tool messages, LangMem store shared by all agents.

    `run_schema` is accepted for signature parity with build_treatment but is not
    used for storage isolation here: LangGraph's PostgresStore partitions data by
    namespace tuple (("memories", session_id)), not by Postgres schema, so a
    unique session_id is what isolates one demo run's memories from another's.
    """
    read_document = make_read_document(scenario)

    from langchain_ollama import OllamaEmbeddings
    from langgraph.store.postgres import PostgresStore
    from langmem import create_manage_memory_tool, create_search_memory_tool

    # A connection pool, for the same reason the treatment uses per-thread
    # connections: sub-agents run on ToolNode's thread pool, and a single shared
    # psycopg connection serialises on an internal lock. A worker holding it
    # while blocked leaves the main thread parked in `lock_PyThread_acquire_lock`
    # forever — observed hanging this condition on the delegation turn.
    #
    # Fixing only the treatment would also have been a fairness bug: one
    # condition protected from a deadlock the other still suffers is not a
    # comparison of memory architectures.
    store_cm = PostgresStore.from_conn_string(
        DSN,
        # Same bounded timeout the treatment's embedder uses. Untimed embedding
        # calls hung both conditions: Ollama accepts the request, never answers,
        # and the blocked thread takes the run with it.
        index={
            "dims": 768,
            "embed": OllamaEmbeddings(model="nomic-embed-text", client_kwargs={"timeout": 120.0}),
        },
        pool_config={"min_size": 1, "max_size": 8},
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
            system_prompt=scenario.subagent_prompt
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

    def invoke(thread_id: str, user_text: str, callbacks: list | None = None) -> dict:
        agent = create_agent(
            model=make_llm(),
            tools=[read_document] + mem_tools + [make_researcher(t) for t in scenario.subtopics],
            system_prompt=scenario.root_prompt.replace(
                "remember_fact", "the manage_memory tool"
            ).replace(
                "relation='supersedes' and the old fact's short id from your " "Memory context",
                "an update to the existing memory",
            )
            + " Before answering, search your memory for relevant context.",
        )
        return agent.invoke(
            {"messages": [("user", user_text)]},
            _config(thread_id, callbacks),
        )

    def cleanup():
        store_cm.__exit__(None, None, None)

    return invoke, cleanup


def build(arm_name: str, scenario_name: str, session_id: str, run_schema: str):
    """Resolve an arm + scenario to (invoke_fn, cleanup_fn)."""
    scenario = get_scenario(scenario_name)
    arm = get_arm(arm_name)
    builder = build_treatment if arm.kind == "treatment" else build_baseline
    return builder(session_id, run_schema, scenario, arm), scenario
