"""LangGraph agent middleware: the integration surface of beads-memory."""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .embeddings import Embedder
from .ids import derive_fact_id, short_id
from .store import BeadsStore, Namespace
from .tools import make_remember_fact


class BeadsMemoryMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        store: BeadsStore,
        namespace: Namespace,
        embedder: Embedder,
        agent_id: str,
        acting_on_behalf_of: str,
        window: int = 10,
        k: int = 8,
        capture_final: bool = True,
        extra_tools: list | None = None,
    ):
        super().__init__()
        self.store = store
        self.namespace = namespace
        self.embedder = embedder
        self.agent_id = agent_id
        self.acting_on_behalf_of = acting_on_behalf_of
        self.window = window
        self.k = k
        self.capture_final = capture_final
        self.tools = [
            make_remember_fact(
                store,
                namespace,
                embedder,
                agent_id=agent_id,
                acting_on_behalf_of=acting_on_behalf_of,
            )
        ] + (extra_tools or [])

    # -- write path 1: passive user-input capture ---------------------------
    def before_model(self, state, runtime):
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                body = str(msg.content)
                self.store.write_fact(
                    self.namespace,
                    kind="user_input",
                    body=body,
                    source="passive_capture",
                    source_key=msg.id or body,
                    agent_id=self.agent_id,
                    acting_on_behalf_of=self.acting_on_behalf_of,
                    embedding=self.embedder.embed(body),
                )
        return None

    # -- write path 4: passive final-answer capture (root only) -------------
    def after_model(self, state, runtime):
        if not self.capture_final:
            return None
        messages = state["messages"]
        last = messages[-1] if messages else None
        if isinstance(last, AIMessage) and not last.tool_calls and str(last.content).strip():
            body = str(last.content)
            self.store.write_fact(
                self.namespace,
                kind="conclusion",
                body=body,
                source="passive_capture",
                source_key=last.id or body,
                agent_id=self.agent_id,
                acting_on_behalf_of=self.acting_on_behalf_of,
                embedding=self.embedder.embed(body),
            )
        return None

    # -- view-only window trim + fact injection -----------------------------
    def wrap_model_call(self, request, handler):
        msgs = list(request.messages)
        windowed = msgs[-self.window :] if self.window else msgs

        # Dedup: facts derived from messages still visible raw must not re-inject.
        exclude = [
            derive_fact_id(self.namespace.id, m.id or str(m.content), str(m.content))
            for m in windowed
            if isinstance(m, HumanMessage)
        ]
        query_text = next(
            (str(m.content) for m in reversed(windowed) if isinstance(m, HumanMessage)),
            None,
        )
        facts = []
        if query_text:
            facts = self.store.search(
                self.namespace.id,
                self.embedder.embed(query_text),
                k=self.k,
                exclude_ids=exclude,
            )
        system = request.system_message or SystemMessage("")
        if facts:
            lines = [f"- [{short_id(f.id)}] ({f.kind}) {f.body}" for f in facts]
            memory_block = (
                "\n\n## Memory (beads)\n"
                "Durable facts from this session. Cite short ids in remember_fact"
                " when a new conclusion supersedes/contradicts/relates to one.\n" + "\n".join(lines)
            )
            system = SystemMessage(str(system.content) + memory_block)
        return handler(request.override(messages=windowed, system_message=system))
