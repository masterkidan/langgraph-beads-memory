"""LangGraph agent middleware: the integration surface of beads-memory."""

from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .embeddings import Embedder
from .ids import content_key, derive_fact_id, short_id
from .segment import DIRECTIVE, classify_fragment, is_substantive, split_into_facts
from .store import BeadsStore, Namespace
from .tools import make_recall_from_subagents, make_remember_fact


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
        recorder: list | None = None,
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
        # Optional append-only log of what was actually injected, and why.
        # Without it, "what did the agent see?" can only be reconstructed by
        # re-running the query later against a store that has since changed —
        # which produced a confidently wrong ranking analysis once already.
        self.recorder = recorder
        self.tools = [
            make_remember_fact(
                store,
                namespace,
                embedder,
                agent_id=agent_id,
                acting_on_behalf_of=acting_on_behalf_of,
            )
        ] + (extra_tools or [])
        if capture_final:
            # Orchestrator-only. Demoted descendant search is similarity-driven
            # and may or may not surface a child's finding; this lets an agent
            # that knows it delegated a topic go and read what the researcher
            # actually recorded. Bound where capture_final is set — the same
            # flag that distinguishes a root agent from a forked sub-agent —
            # so sub-agents cannot use it to reach across at their siblings.
            self.tools.append(make_recall_from_subagents(store, namespace))

    # -- write path 1: passive user-input capture ---------------------------
    def before_model(self, state, runtime):
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                self._capture_user_message(msg)
        return None

    def _user_fact_specs(self, msg) -> list[tuple[str, str]]:
        """(source_key, body) for each fact a user message produces.

        Single source of truth shared by capture and by the dedup exclusion.
        They were previously derived independently, and splitting broke them
        apart: capture wrote `<base>#<i>` fragment keys while the exclusion
        still derived one id from the whole message, so nothing matched and
        every fragment was re-injected while the message was still on screen.
        """
        whole = str(msg.content)
        fragments = split_into_facts(whole)
        base_key = msg.id or content_key(whole)
        specs = (
            [(base_key, fragments[0])]
            if len(fragments) == 1
            else [(f"{base_key}#{i}", body) for i, body in enumerate(fragments)]
        )
        # Drop conversational framing before it becomes memory. A directive is
        # kept regardless: it is already held out of retrieval and is the
        # provenance of downstream choices. See segment.is_substantive for the
        # measured reason — "New shift taking over." outranked every real fact
        # for the query "what should we try next".
        keep = [(k, b) for k, b in specs if classify_fragment(b) == DIRECTIVE or is_substantive(b)]
        # Never let filtering silence a message entirely; a message made only of
        # framing still gets recorded whole rather than vanishing from the trail.
        return keep or [(base_key, whole)]

    def _capture_user_message(self, msg) -> None:
        """Capture a user message as one fact per distinct claim.

        A message stating several constraints at once used to become a single
        fact, which measurably hurt in two ways: one embedding averaged across
        every topic, so an individual constraint never reached the top-K for a
        related question; and one row, so a single bad `supersedes` edge retired
        every constraint in it. Splitting is verbatim and heuristic — passive
        capture is in the model-call hot path and cannot afford an LLM.

        The `source_key` carries the fragment index so the parts of one message
        stay distinct under the content-derived id scheme, and a replay of the
        same message still collapses onto the same ids.
        """
        for source_key, body in self._user_fact_specs(msg):
            # Questions, instructions and stated goals are kept — they are the
            # provenance of every downstream choice — but marked so retrieval
            # does not spend a top-K slot re-injecting the current query.
            kind = "directive" if classify_fragment(body) == DIRECTIVE else "user_input"
            self.store.write_fact(
                self.namespace,
                kind=kind,
                body=body,
                source="passive_capture",
                source_key=source_key,
                agent_id=self.agent_id,
                acting_on_behalf_of=self.acting_on_behalf_of,
                embedding=self.embedder.embed(body),
            )

    # -- write path 4: passive final-answer capture (root only) -------------
    def after_model(self, state, runtime):
        """Capture the agent's answer the same way a user message is captured.

        This used to store the whole answer as ONE fact, while user messages
        were split per claim. That inconsistency was ours, and it dominated the
        store. Measured on a real run: the root's own previous answers were
        **58% of all stored text** — 7 facts, 5,944 characters, the largest
        1,925 — against 2,457 characters of actual sub-agent findings.

        Three consequences, all observed:

        - One embedding averaged over an entire multi-topic answer, which is the
          exact dilution splitting exists to prevent, applied to only one side.
        - Token bloat. A 1,925-character fact is ~500 tokens; two or three in a
          top-8 consume most of the injection budget, which is why the treatment
          spent MORE input tokens than a flat-blob baseline.
        - A feedback loop. The answer is derived FROM memory, written back INTO
          memory, then retrieved and re-read as evidence. A hallucination ("the
          application tier is not the cause") became a durable fact and was
          repeated in every later turn.

        Splitting plus a content-derived key fixes all three: claims become
        individually retrievable, restating a claim collapses onto the same
        fact id instead of accumulating another copy, and framing is dropped.
        """
        if not self.capture_final:
            return None
        messages = state["messages"]
        last = messages[-1] if messages else None
        if not (isinstance(last, AIMessage) and not last.tool_calls and str(last.content).strip()):
            return None
        whole = str(last.content)
        for body in split_into_facts(whole):
            if not is_substantive(body):
                continue
            self.store.write_fact(
                self.namespace,
                kind="conclusion",
                body=body,
                source="passive_capture",
                # Content-derived, NOT the message id. An agent restating a
                # conclusion it already reached is not a new fact, and keying on
                # the message id meant every turn added another copy.
                source_key=content_key(body),
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
        # This must mirror before_model's write exactly — same source, same key
        # fallback — or the derived ids diverge and dedup silently stops working.
        exclude = [
            derive_fact_id(
                self.namespace.session_id,
                self.namespace.id,
                "passive_capture",
                source_key,
                body,
            )
            for m in windowed
            if isinstance(m, HumanMessage)
            for source_key, body in self._user_fact_specs(m)
        ]
        # Query from the FULL message list, not the window. The window governs
        # what the model re-reads; it must not govern what memory is retrievable.
        # Taking the query from `windowed` meant a turn with enough tool calls to
        # push the user's question out of view produced no query at all, and
        # therefore injected no facts — memory silently switched off on exactly
        # the long, roundabout turns where it is most needed. Falling back to the
        # last non-empty message keeps retrieval working even on a turn that
        # opens with tool traffic and contains no HumanMessage at all.
        query_text = next(
            (str(m.content) for m in reversed(msgs) if isinstance(m, HumanMessage)),
            None,
        ) or next((str(m.content) for m in reversed(msgs) if str(m.content).strip()), None)
        facts, scored = [], []
        if query_text:
            scored = self.store.search(
                self.namespace.id,
                self.embedder.embed(query_text),
                k=self.k,
                exclude_ids=exclude,
                with_scores=True,
            )
            facts = [f for f, _d, _demoted in scored]
        if self.recorder is not None:
            self.recorder.append(
                {
                    "agent_id": self.agent_id,
                    "query": (query_text or "")[:200],
                    "k": self.k,
                    "excluded": len(exclude),
                    "injected": [
                        {
                            "id": short_id(f.id),
                            "kind": f.kind,
                            "source": f.source,
                            "agent_id": f.agent_id,
                            "distance": round(d, 4),
                            "demoted": demoted,
                            "body": f.body[:160],
                        }
                        for f, d, demoted in scored
                    ],
                }
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
