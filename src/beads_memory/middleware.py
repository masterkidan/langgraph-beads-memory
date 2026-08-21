"""LangGraph agent middleware: the integration surface of beads-memory."""

from __future__ import annotations

import os
import uuid

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .embeddings import Embedder
from .ids import content_key, derive_fact_id, short_id
from .segment import DIRECTIVE, classify_fragment, is_substantive, split_into_facts
from .store import BeadsStore, Namespace
from .tools import make_recall_from_subagents, make_remember_fact, make_search_memory

# Ceiling on how much of one tool result becomes memory. Message content is
# bounded by what a person or model typed; tool output is not — this benchmark
# reads ~1.3 KB corpus files, but the playground calls web search, and an
# uncapped path would put whole pages into a store whose premise is that a claim
# is not a document.
TOOL_RESULT_CAPTURE_LIMIT = int(os.environ.get("BEADS_TOOL_RESULT_LIMIT", "4000"))


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
        inject: bool = True,
        search_tool: bool = False,
        capture_tool_results: bool = True,
    ):
        """`inject=False` with `search_tool=True` swaps automatic recall for an
        agent-invoked `search_memory`, matching the baseline's interface exactly
        so that ranking is the only variable left between the arms. Capture is
        unaffected either way — what gets written does not depend on how it is
        read back.

        `capture_tool_results` defaults ON as of 2026-08-20, and the reason it
        used to be off no longer reproduces.

        It buys a capability nothing else reaches: `buried_metric_recalled` goes
        0/3 to 3/3 (incident, gemma4:12b, N=3), because a sub-agent reads a
        figure and drops it when summarising — the fact is never written, so no
        ranking change can touch it. It is the single worst metric in the suite,
        passing 9 of 56 incident runs.

        The recorded price was `breadth_complete` 1/3 to 0/3 and input tokens
        going from 35% below baseline to 13% below. Re-measured on 2026-08-16
        across four cells with the derivation filter in place, the store still
        roughly doubled (94 -> 182 facts) but tokens moved -4.8%, -4.9%, +5.3%,
        -17.6% — flat or better. That is consistent with the price having been a
        crowding artefact of the old ranker rather than a property of capture:
        the derivation filter drops restatements before they take slots, so
        extra candidates no longer inflate the block.

        Captured into the CALLING agent's namespace, which is what makes the
        volume affordable: a sub-agent's raw reads land in its own namespace,
        demoted for the parent and reachable through `recall_from_subagents`. A
        root-initiated read still lands in root, because that is where it
        originated — verified in a real run at 44 child facts to 27 root."""
        super().__init__()
        self.store = store
        self.namespace = namespace
        self.embedder = embedder
        self.agent_id = agent_id
        self.acting_on_behalf_of = acting_on_behalf_of
        self.window = window
        self.k = k
        self.capture_final = capture_final
        self.inject = inject
        self.capture_tool_results = capture_tool_results
        # Optional append-only log of what was actually injected, and why.
        # Without it, "what did the agent see?" can only be reconstructed by
        # re-running the query later against a store that has since changed —
        # which produced a confidently wrong ranking analysis once already.
        self.recorder = recorder
        # The facts injected into the most recent model call. Written by
        # wrap_model_call, consumed by after_model as derivation provenance.
        self._last_injected: list = []
        self.tools = [
            make_remember_fact(
                store,
                namespace,
                embedder,
                agent_id=agent_id,
                acting_on_behalf_of=acting_on_behalf_of,
            )
        ] + (extra_tools or [])
        if search_tool:
            self.tools.append(make_search_memory(store, namespace, embedder))
        if capture_final:
            # Orchestrator-only. Demoted descendant search is similarity-driven
            # and may or may not surface a child's finding; this lets an agent
            # that knows it delegated a topic go and read what the researcher
            # actually recorded. Bound where capture_final is set — the same
            # flag that distinguishes a root agent from a forked sub-agent —
            # so sub-agents cannot use it to reach across at their siblings.
            self.tools.append(make_recall_from_subagents(store, namespace))

        # Every memory tool this middleware binds — remember_fact, conclude_task
        # (arriving via extra_tools), search_memory, recall_from_subagents.
        # DERIVED, not a hardcoded name list: capturing a memory tool's output
        # is a feedback loop, since retrieved facts would be written back as new
        # facts, retrieved again, and written again. Deriving it means a memory
        # tool added later is excluded automatically rather than silently
        # starting to feed itself.
        self._memory_tool_names = {t.name for t in self.tools}

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
        specs = self._user_fact_specs(msg)
        # ONE embed call for the whole message, reused by every claim in it —
        # so a message splitting into six facts costs seven embeds, not twelve.
        # Skipped when the message produced a single fact covering all of it,
        # where the context vector would just duplicate the claim's own.
        whole = str(msg.content)
        ctx = self.embedder.embed(whole) if len(specs) > 1 else None
        for source_key, body in specs:
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
                context_embedding=ctx,
            )

    # -- write path 4: passive tool-result capture --------------------------
    def wrap_tool_call(self, request, handler):
        """Capture what a tool returned, into the namespace of the agent that
        called it.

        Motivating failure: `buried_metric_recalled` scored 0/3 in EVERY arm,
        both code versions, across the whole benchmark — the only metric no
        ranking change could move. The number it asks for (a TLS handshake p99
        of 41ms) is in `demo/corpus_incident/network.md`; the network
        researcher read that file and dropped the figure when summarising. The
        fact was never written, so no retrieval could find it. Capturing tool
        results fixes that at the source rather than at the ranking layer.

        Scoped to `self.namespace`, which is the CALLING agent's — so a
        sub-agent's raw reads land in the sub-agent's namespace, demoted for the
        parent and reachable through `recall_from_subagents`, rather than
        growing the root namespace. That matters: the root already fits 94 facts
        into 8 slots with a rank-8/rank-9 margin of 0.00126, and adding
        candidates there would make selection measurably more fragile.
        """
        result = handler(request)
        if not self.capture_tool_results:
            return result
        try:
            name = (request.tool_call or {}).get("name")
            if name and name not in self._memory_tool_names:
                content = getattr(result, "content", None)
                if isinstance(content, str) and content.strip():
                    self._capture_tool_result(name, content)
        except Exception:  # noqa: BLE001
            # Capture must never break the tool call it observed. A memory
            # write failing is a lost fact; an exception here would instead
            # take out the researcher whose finding we were trying to keep.
            pass
        return result

    def _capture_tool_result(self, tool_name: str, content: str) -> None:
        # Ceiling, because tool output is unbounded in a way message content is
        # not. The benchmark's corpus is ~1.3 KB per read, but the playground
        # calls web search, and an un-capped path would put whole pages into a
        # store whose entire premise is that a claim is not a document.
        text = content[:TOOL_RESULT_CAPTURE_LIMIT]
        bodies = [b for b in split_into_facts(text) if is_substantive(b)]
        if not bodies:
            return
        ctx = self.embedder.embed(text) if len(bodies) > 1 else None
        for i, body in enumerate(bodies):
            self.store.write_fact(
                self.namespace,
                kind="conclusion",
                body=body,
                source="tool_result",
                # Content-derived and tool-scoped: re-reading the same document
                # collapses onto the same ids rather than accumulating copies,
                # the same property `after_model` relies on.
                source_key=f"tool:{tool_name}:{content_key(text)}#{i}",
                agent_id=self.agent_id,
                acting_on_behalf_of=self.acting_on_behalf_of,
                embedding=self.embedder.embed(body),
                context_embedding=ctx,
            )

    # -- write path 5: passive final-answer capture (root only) -------------
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
        bodies = [b for b in split_into_facts(whole) if is_substantive(b)]
        # The answer this claim came from, embedded once. This is the capture
        # that most needs it: an agent's synthesis states its root cause in one
        # sentence and its reasoning in the surrounding ten, and that one
        # sentence alone was measured at rank 30 of ~90 against the very
        # question it answered.
        ctx = self.embedder.embed(whole) if len(bodies) > 1 else None
        for body in bodies:
            fact = self.store.write_fact(
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
                context_embedding=ctx,
            )
            self._record_provenance(fact.id)
        return None

    def _record_provenance(self, fact_id: uuid.UUID) -> None:
        """Link a captured conclusion to the facts that were in context for it.

        Provenance is free here and unavailable anywhere else. The injected set
        is exactly what memory contributed to the answer, so recording the edge
        at capture time needs no extraction call — which matters, because this
        runs in the model-call hot path where an LLM round trip is not an option.

        This is what makes invalidation follow derivation instead of guessing
        from cosine distance. Without it a correction retires the row it points
        at and leaves everything computed from that row standing; the measured
        version of that was an answer still citing a $100,000 budget three turns
        after the user corrected it to $50,000.

        Self-links are skipped: restating a claim collapses onto the same
        content-derived id, so a fact can appear in its own injected set.
        """
        for source_id in self._last_injected:
            if source_id != fact_id:
                self.store.add_edge(fact_id, source_id, "derived_from")

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
        if not self.inject:
            # Recall is the agent's job this run (see `search_tool`). Capture
            # and the window trim below still apply — only automatic injection
            # is off, so the arms differ in interface alone.
            query_text = None
        if query_text:
            scored = self.store.search(
                self.namespace.id,
                self.embedder.embed(query_text),
                k=self.k,
                exclude_ids=exclude,
                with_scores=True,
            )
            facts = [f for f, _d, _demoted in scored]
        # Held for after_model, which turns it into `derived_from` edges.
        self._last_injected = [f.id for f in facts]
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
