"""Agent-facing tools, bound per-namespace as closures."""

from __future__ import annotations

from langchain_core.tools import tool
from pydantic import BaseModel, model_validator

from .embeddings import Embedder
from .ids import short_id
from .segment import is_substantive, split_into_facts
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
            # Stable, content-derived key — NOT a random uuid. Fact ids are
            # derive_fact_id(namespace, source_key, body), so a random key here
            # defeated idempotency on this path entirely: recording the same
            # sentence twice produced two rows. Observed in a real run, where the
            # model called remember_fact three times with byte-identical text and
            # the store kept all three. With a constant key, an identical body in
            # the same namespace resolves to the same id and ON CONFLICT DO
            # NOTHING collapses the repeat. Different namespaces still get
            # distinct facts, because namespace_id is part of the derivation.
            source_key="remember",
            agent_id=agent_id,
            acting_on_behalf_of=acting_on_behalf_of,
            embedding=embedder.embed(body),
        )
        if target is not None and not store.add_edge(fact.id, target.id, relation):
            # Refused: the guard found the two facts unrelated. Say so, so the
            # agent can correct its reference rather than believing it retired
            # something it did not.
            return (
                f"Remembered [{short_id(fact.id)}], but the {relation} link to "
                f"{relates_to} was refused: that fact is about something else. "
                "Cite the short id of the fact this actually replaces."
            )
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
        # One claim per fact, as everywhere else. A summary used to cross into
        # the parent whole, which made it the last un-split write path — and the
        # cost showed up in invalidation rather than in retrieval. Measured on a
        # vecdb run: each researcher's summary paired a finding with a budget
        # judgement ("Internal benchmarks show p95 ~15ms ... fits within the
        # $100k budget"), so correcting the budget retired all three summaries
        # whole and took the benchmark findings with them. There is no
        # granularity at which invalidation can be right about a fact that mixes
        # a stale premise with live findings; the fix belongs here, at capture.
        fragments = [f for f in split_into_facts(summary) if is_substantive(f)] or [summary]
        # The whole rollup, embedded once and shared by its fragments. This is
        # the path where splitting cost the most: a researcher's summary names
        # its subsystem in one sentence and its verdict in another, so neither
        # fragment alone matches "what did we rule out, and what survived" as
        # well as the summary did. Measured — the fraud-scoring root cause was
        # stored nine times over and never once reached a top-8 injection.
        ctx = embedder.embed(summary) if len(fragments) > 1 else None
        facts = [
            store.write_fact(
                parent_namespace,
                kind="summary",
                body=body,
                source="conclude_task",
                # Indexed so the parts of one summary stay distinct under the
                # content-derived id scheme, mirroring _user_fact_specs.
                source_key=(
                    f"conclude:{child_namespace.id}"
                    if len(fragments) == 1
                    else f"conclude:{child_namespace.id}#{i}"
                ),
                agent_id=agent_id,
                acting_on_behalf_of=acting_on_behalf_of,
                embedding=embedder.embed(body),
                context_embedding=ctx,
            )
            for i, body in enumerate(fragments)
        ]
        # Every claim is a rollup of the child's exploration, not just the first
        # — "where did this come from?" has to stay answerable per claim, which
        # is the reason for having the edges at all.
        for child_fact in store.facts_in_namespace(child_namespace.id):
            for fact in facts:
                store.add_edge(fact.id, child_fact.id, "rollup_of")
        # The supersedes link is the sub-agent's single explicit claim about what
        # its conclusion replaces, so it stays on one fact rather than being
        # multiplied across fragments.
        fact = facts[0]
        refused = target is not None and not store.add_edge(fact.id, target.id, "supersedes")
        concluded["fact_id"] = fact.id
        if refused:
            return (
                f"Task concluded [{short_id(fact.id)}], but the supersedes link to "
                f"{supersedes} was refused: that fact is about something else."
            )
        return f"Task concluded [{short_id(fact.id)}]"

    return conclude_task


def make_search_memory(store: BeadsStore, namespace: Namespace, embedder: Embedder):
    """`search_memory`, byte-identical in name and description to langmem's.

    This exists to remove a confound, not to add a feature. The two arms of the
    benchmark differed in TWO ways at once: the baseline recalls through an
    agent-authored `search_memory` call, while the treatment injects
    automatically using the raw user message as its query. So a difference in
    outcome could be attributed to the ranking OR to the interface, and the
    measured trace says the interface mattered — on the incident scenario the
    baseline's agent wrote itself "incident status investigation progress ruled
    out facts" while the treatment embedded "New shift taking over. Given
    everything we've established, what should we try next..." and retrieved six
    restatements of one constraint.

    Holding the name, description and signature fixed makes the model's
    behaviour the constant and the ranking the variable. The description is
    copied verbatim from `langmem.create_search_memory_tool` — if that string
    drifts upstream, this comparison quietly stops being controlled, which is
    why it is pinned here in one place rather than paraphrased.

    The RESPONSE, unlike the signature, is deliberately not flat. Each hit is
    rendered with its distance-1 neighbours, labelled by relation — which is
    the one thing a document store structurally cannot answer. A flat list of
    claims is thin by construction: splitting a message per claim is what makes
    `supersedes` surgical, and it is also what strips each claim of the context
    it was carved from. The edges put that context back at read time instead of
    duplicating it at write time.

    What this buys, stated honestly: measured offline on both scenarios,
    one-hop expansion did NOT improve aspect coverage (incident 89%, vecdb 88%,
    unchanged). Its value is legibility — `supersedes` renders as "replaces",
    so a corrected budget arrives with the correction attached rather than as
    two numbers with nothing marking which is current. That is a claim about
    what the model can reason with, and it is not yet measured; it should be
    treated as untested until an end-to-end run says otherwise.

    Superseded neighbours are shown on purpose. They are labelled, never listed
    as current, and "this replaced $100k" is the most useful thing that can be
    said about a $50k fact.
    """
    # Read as a verb phrase from the seed: "[seed] --replaces--> [neighbour]".
    _OUT = {"supersedes": "replaces", "contradicts": "contradicts",
            "relates_to": "relates to", "derived_from": "derived from",
            "rollup_of": "summarises"}
    _IN = {"supersedes": "was replaced by", "contradicts": "contradicted by",
           "relates_to": "relates to", "derived_from": "supports",
           "rollup_of": "summarised in"}

    # Which edges are worth characters, in order. `derived_from` is EXCLUDED,
    # not merely ranked last, and that is the difference between this response
    # being an advantage and being a regression.
    #
    # Measured: `derived_from` is emitted on every capture — 303 edges in one
    # run against 75 `rollup_of` and 6 `supersedes` — so it dominates any hit's
    # neighbourhood while carrying the least. Including it rendered fragments
    # like "Specifically, it measured" and "We have a production incident" as
    # context, and pushed a k=8 response to 2,662 characters against the flat
    # baseline's 2,051 — spending the payload advantage this library exists to
    # provide, on provenance the model cannot use. It stays queryable through
    # the store; it just does not buy a slot here.
    #
    # `relates_to` is excluded on the same principle, one step further in. It
    # is the catch-all the model reaches for when it wants to link two facts
    # and has no claim about HOW they relate — the tool description offers it
    # as the fallback after `supersedes` and `contradicts`. An edge that means
    # "these are somehow associated" cannot be rendered as anything a reader
    # can act on, so it would spend characters restating the similarity that
    # ranked the hit in the first place.
    #
    # What remains is exactly the set a document store cannot express, and each
    # renders as a claim rather than an association: what replaced what, what
    # contradicts what, and which raw findings a summary stands for.
    _EXPAND = {"supersedes": 0, "contradicts": 1, "rollup_of": 2}

    # Strength is a property of the NEIGHBOUR too, not only of the edge. A link
    # is worth characters when it lands on something someone actually asserted
    # — a claim the user stated, a conclusion the agent reached, or a
    # sub-agent's rollup of the same. `directive` neighbours are excluded for
    # the reason `search` already holds directives out of retrieval: a question
    # resembles the query by construction, so expanding onto one spends a slot
    # echoing what was just asked.
    _EXPAND_KINDS = ("user_input", "conclusion", "summary")
    MAX_NEIGHBOURS = 2

    @tool
    def search_memory(query: str, limit: int = 10) -> str:
        """Search your long-term memories for information relevant to your current context."""
        k = max(1, min(limit, 25))
        scored = store.search(namespace.id, embedder.embed(query), k=k, with_scores=True)
        if not scored:
            return "No memories found."
        facts = [f for f, _d, _dem in scored]
        readable = store.ancestor_chain(namespace.id) + store.descendant_scope(namespace.id)
        nbrs = store.neighbours([f.id for f in facts], readable)

        lines = []
        # Seeded with every hit, and carried ACROSS hits: a fact already shown
        # — as a result or as some earlier hit's neighbour — is not worth its
        # characters twice, and dense stores link the same few summaries from
        # everywhere.
        seen = {f.id for f in facts}
        for fact in facts:
            lines.append(f"- [{short_id(fact.id)}] {fact.body}")
            edges = sorted(
                (e for e in nbrs.get(fact.id, [])
                 if e[1] in _EXPAND and e[0].kind in _EXPAND_KINDS),
                key=lambda e: (_EXPAND[e[1]], len(e[0].body)),
            )
            shown = 0
            for nb, relation, direction in edges:
                if shown >= MAX_NEIGHBOURS:
                    break
                if nb.id in seen:
                    continue
                seen.add(nb.id)
                shown += 1
                label = (_OUT if direction == "out" else _IN).get(relation, relation)
                # Neighbours are context, not answers: truncated so one densely
                # linked hit cannot crowd out the other k-1 results.
                body = nb.body if len(nb.body) <= 120 else nb.body[:117] + "..."
                stale = " (superseded)" if nb.status != "active" else ""
                lines.append(f"    └─ {label}{stale}: [{short_id(nb.id)}] {body}")
        return "\n".join(lines)

    return search_memory


def make_recall_from_subagents(store: BeadsStore, namespace: Namespace):
    """Tool: read what this agent's sub-agents actually recorded.

    Demoted descendant search is similarity-driven, so a child fact surfaces
    only when the query happens to match it. An orchestrator usually knows
    something stronger — it delegated a topic to a named researcher — and should
    be able to look directly.

    This exists because of a measured failure: a researcher read "Binary
    quantization reduces RAM usage up to 32x", its rollup summarised deployment
    instead, and when later asked for the optimization the orchestrator
    hallucinated a number. The detail was in the store the whole time, one
    namespace below, with no way to ask for it.
    """

    @tool
    def recall_from_subagents(agent_id: str | None = None) -> str:
        """Read the raw findings your sub-agents recorded, beyond the one-line
        summary each reported back. Pass agent_id (e.g. 'researcher_qdrant') to
        read one researcher's findings, or omit it to see all of them. Use this
        when asked about a specific detail a researcher investigated."""
        available = sorted({f.agent_id for f in store.subtree_facts(namespace.id, limit=200)})
        if not available:
            return "No sub-agent findings recorded yet."
        if agent_id and agent_id not in available:
            return f"No findings from {agent_id!r}. Available sub-agents: " + ", ".join(available)
        facts = store.subtree_facts(namespace.id, agent_id=agent_id, limit=25)
        if not facts:
            return "No sub-agent findings recorded yet."
        lines = [f"- [{short_id(f.id)}] ({f.agent_id}) {f.body}" for f in facts]
        return "Sub-agent findings:\n" + "\n".join(lines)

    return recall_from_subagents
