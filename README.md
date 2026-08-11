# langgraph-beads-memory

Beads-style durable memory for [LangGraph](https://github.com/langchain-ai/langgraph) agents on Postgres — a typed fact/conclusion graph with explicit capture (not blind auto-extraction) and enforced sub-agent memory forking with rollup summaries, instead of an opaque conversation summary.

> Status: **package implemented, N=3 results measured.** Core library built and tested (87 tests, real Postgres); scored comparison complete — see [results](results/2026-08-09-results.md).

## How it compares to LangGraph's built-in memory

![Side-by-side comparison: stock LangGraph memory (checkpointer plus LangMem extraction store) versus langgraph-beads-memory (typed fact graph), running the same three-conversation scenario in lockstep through capture, a new thread, delegation, a corrected constraint, and the final answer.](docs/assets/comparison.svg)

Both lanes run the **same scenario**, step for step. The structural differences that matter:

| | stock LangGraph memory | langgraph-beads-memory |
|---|---|---|
| capture | agent must call `manage_memory` — if it doesn't, nothing persists | user input + final answers captured automatically, verbatim |
| granularity | one memory per saved document | one fact per claim, each separately embedded and supersedable |
| across threads | new `thread_id` resets history; agent must decide to search the store | one `session_id` spans threads; relevant facts injected automatically |
| revising a fact | old and new documents coexist; nothing marks which is current | typed `supersedes` edge retires the stale fact, keeps it for audit |
| questions vs claims | undifferentiated | `directive` kind kept for provenance but held out of retrieval |
| sub-agents | results return as messages; no link back to what produced them | isolated namespaces, enforced `conclude_task`, `rollup_of` audit edges |
| a crashed sub-agent | silently returns nothing | wrapper synthesizes a "did not complete" fact |

**What the built-in option does well, and what this costs.** The checkpointer gives complete message history within a thread, `BaseStore` has real vector search, and it's first-party with no extra dependency — when the agent does save a memory, cross-thread recall genuinely works. This library adds a dependency and a Postgres schema.

**Measured token cost (N=3): this library used ~45% *fewer* input tokens** (10,083 vs 18,220 mean), consistently across all three runs. An earlier single run had suggested the opposite (~46% more); that run predated a scenario fix and is superseded. Injecting a compact, relevance-ranked fact set turned out cheaper than the baseline's accumulated history plus memory-search payloads. The comparison diagram above still shows the old figure and is pending an update.

## How it works

![How langgraph-beads-memory works: user messages and agent conclusions become durable facts on a session-wide memory string; sub-agents fork isolated namespaces and roll summaries back; a revised budget supersedes the stale one; a later conversation on a new thread recalls only active facts.](docs/assets/mechanism.svg)

One session (`session_id: vecdb-research`) spanning three conversations. Facts are beads on the session string — threads come and go, the string stays.

1. **Capture** — every user message is written verbatim onto the session string, no extraction LLM involved. The turn's final answer is captured as a conclusion automatically, so durable memory never depends on the model remembering a tool call.
2. **Fork** — delegating research gives each sub-agent an isolated child namespace. It reads its ancestors; it never sees its siblings.
3. **Enforced rollup** — each sub-agent must call `conclude_task`. One summary lands on the parent, linked by `rollup_of` edges back to its raw exploration. A crashed sub-agent leaves a "did not complete" fact rather than vanishing.
4. **Supersede** — when the user revises a constraint, the new fact supersedes the old one. The stale value is retired from retrieval but kept for audit.
5. **Recall** — a brand-new thread starts warm, and only *active* facts can reach the model. This is exactly where thread-scoped memory starts cold, and where extraction stores surface both the old and new value with nothing marking which is current.

## The memory model

![Memory hierarchy: a session scope containing a root namespace and isolated child namespaces per sub-agent; the anatomy of a single fact with its kind, status and source; the four fact kinds with directives held out of retrieval; the active/superseded/archived lifecycle; and the typed edges between facts.](docs/assets/memory-hierarchy.svg)

### Scope: sessions, namespaces, and what can read what

```
session_id                          one long-lived memory scope, spanning many thread_ids
 └── root namespace  {}             the supervisor's memory
     ├── {task, sub-3f2a}           a forked sub-agent
     ├── {task, sub-91cc}           another, isolated from its siblings
     └── {task, sub-d04e}
```

A sub-agent reads **its own namespace plus its ancestors, never a sibling's**. That is what
stops three parallel researchers contaminating each other's context.

The relationship is deliberately **asymmetric**: a parent additionally reads its
whole subtree, with a rank penalty. Descendant visibility is a parent privilege,
not a general relaxation — children still read only upward, so siblings remain
mutually invisible.

Namespace ids are *derived* — `uuid5(session_id, extra_path)` — so replaying the same conversation into an empty database reproduces the same ids. The one deliberately random element is the child suffix (`sub-3f2a`), because two concurrently spawned sub-agents must not collide.

### Kinds: what a memory is

| kind | what it holds | retrieved by default? |
|---|---|---|
| `user_input` | a claim the user stated, captured verbatim | yes |
| `directive` | a question, instruction, or stated goal | **no** — stored and queryable, held out of retrieval |
| `conclusion` | something the agent concluded | yes |
| `summary` | a sub-agent's rollup into its parent | yes |

A user message is split into **one fact per claim**: *"the budget is $100k per year, it must be self-hostable, and I only trust primary benchmark data"* becomes three facts, not one. Each then gets its own embedding, and a `supersedes` edge can retire one without touching the others.

`directive` exists because questions rank highly against a query precisely *by resembling it*. Measured: four of eight injected slots were question fragments, displacing the constraint the answer needed. Directives are provenance — they explain why work happened — so they are kept and remain queryable via `search(include_directives=True)`; they just don't compete for the retrieval budget.

### Status: retirement is not deletion

| status | meaning |
|---|---|
| `active` | retrievable |
| `superseded` | replaced by a newer fact — out of retrieval, **kept for audit** |
| `archived` | compacted — out of retrieval, kept for audit |

**Nothing is ever deleted.** A superseded fact stays queryable, so "what did we believe before, and what replaced it?" is always answerable.

### Source: which path wrote it

| source | when |
|---|---|
| `passive_capture` | user messages and final answers — automatic, no tool call, no LLM |
| `remember_tool` | the agent deliberately called `remember_fact` |
| `conclude_task` | a sub-agent's enforced rollup |
| `fallback_conclude` | the sub-agent crashed or forgot; the wrapper synthesised one |
| `compaction` | produced by compaction (designed; not exercised in the demo) |

### Edges: how facts relate

| relation | meaning |
|---|---|
| `supersedes` | replaces. **Guarded** — refused unless the two facts are semantically close (cosine ≥ 0.55) |
| `rollup_of` | a summary points back at every raw exploration fact behind it — the audit trail |
| `contradicts` | asserted to conflict |
| `relates_to` | associated |
| `derived_from` | a compaction summary points at what it replaced |

The `supersedes` guard exists because an agent once retired the user's entire constraints message with *"The investigation into Weaviate has been completed."* Nothing checked the two facts were about the same thing. A rule based on fact *kind* would not have worked — the one legitimate revision had the identical shape — so the guard uses similarity instead.

### Ranking

The retrieval score is deliberately simple, and worth stating plainly rather
than leaving implied by the machinery around it:

```
score = cosine_distance(fact.embedding, query)
      + 0.15   if the fact lives in a descendant namespace
take top 8
```

Everything else is a **filter**, not a ranking signal: `status = 'active'`,
`kind <> 'directive'`, has an embedding, not already visible in the raw message
window, namespace in scope.

**The typed graph barely participates.** `kind` and `status` are consulted only
as binary excludes, and `fact_edges` is not consulted at all — a `supersedes`
edge influences retrieval solely by flipping a status, while `rollup_of` and
`relates_to` have no effect on what gets injected. This is a typed store with a
largely type-blind ranker.

That gap is visible in the measured results. The revised budget survives partly
because it gets *restated* repeatedly, and the ranker has no frequency or
reinforcement term to make that deliberate. Directives had to be excluded
outright rather than down-weighted, because there was no weighting lever. And
with no diversity term, near-duplicate facts can occupy several of the eight
slots.

Signals a fuller ranker would likely carry, none of which exist here: recency,
`kind` weighting (a stated constraint arguably outranking an agent's own
conclusion), edge awareness, and diversity. The contribution of this project is
the typed, auditable *store*; retrieval is currently a thin layer over pgvector
sitting on top of it.

### Identity

A fact's id is derived from `(session_id, namespace_id, source, source_key, sha256(body))`. Content-addressed, so a LangGraph checkpoint replay re-running a capture hook writes nothing new rather than duplicating.

## The problem

LangGraph ships two memory primitives: a checkpointer for thread-scoped state, and a `BaseStore`/`PostgresStore` for cross-thread key-value memory. Frameworks built on top (LangMem, Mem0, Zep) mostly bet on automatic LLM extraction — scan the transcript, pull out "facts," write them somewhere. That's fast to wire up, but it's also imprecise, hard to audit, and gives you no way to say "this conclusion replaced that one" or "this sub-agent's exploration shouldn't pollute the parent's context."

[beads](https://github.com/steveyegge/beads) — Steve Yegge's dependency-aware issue tracker for coding agents — took a different stance for task memory: a typed graph of issues, explicit `bd remember` calls for durable insight, and semantic decay instead of silent deletion. `langgraph-beads-memory` brings that same stance to LangGraph's conversational/multi-agent memory, backed by nothing but Postgres.

## What it does differently

- **Explicit, dual-path capture.** User input is captured verbatim as it arrives — no extraction LLM call needed. Agent conclusions are captured only when the agent deliberately calls a `remember_fact` tool. Nothing gets written to memory the agent (or user) didn't put there.
- **A typed fact graph, not a blob.** Facts relate to each other through typed edges — `supersedes`, `contradicts`, `relates_to`, `derived_from`, `rollup_of` — so "this replaced that" and "this summary was derived from these five facts" are first-class, queryable relationships.
- **Durable, auditable, enforced sub-agent rollups.** LangGraph sub-agents already have isolated context — that's not the claim. The claim: when a supervisor spawns a sub-agent, its memory forks into a child namespace, and its conclusion is *required* — the adapter synthesizes a "task did not complete" fact if the sub-agent crashes or forgets, so no delegated task ever vanishes silently. The rollup summary is a durable fact (not a message that scrolls away), linked by `rollup_of` edges back to every exploration fact that produced it, so any conclusion can be audited by drill-down. Raw exploration stays in the child namespace and is **demoted, not hidden** — a parent can still reach it (see [Ranking](#ranking)), because a summary written by a small model is lossy and the detail it drops is sometimes the one you need later.
- **Postgres-only.** No Neo4j, no separate vector database — namespaces, facts, and edges all live in one Postgres schema (`pgvector` for embeddings).
- **Session-scoped, not identity-coupled.** The core schema is anchored on `session_id` alone — a long-lived memory scope that deliberately spans LangGraph threads, so a new conversation that continues the same work recalls everything from earlier ones. `langgraph-beads-memory` doesn't need to know what a "user" is; if an application wants a user↔session mapping, it owns that table itself.

## How it fits into a LangGraph app

It's an **agent middleware** (LangGraph's `create_agent` pre/post-model hook API) — not a `BaseStore` implementation and not a checkpointer replacement. Thread-level state/replay stays with LangGraph's own `PostgresSaver`; this owns a separate schema for durable, structured memory and wires in purely through hooks, so there are no explicit store calls to write in the common path.

Per turn, the middleware:
1. Passively captures new user messages as facts (no LLM call).
2. Keeps a sliding window of the last ~10 raw messages in context; older messages get distilled into facts as they roll off.
3. Runs semantic search over the current namespace, its ancestors, and (for a parent) its demoted descendants, then injects the top-K relevant facts.

Full schema, hook lifecycle, idempotency guarantees, and error handling are in the design spec (linked below).

## Comparison

No existing tool combines all of this. The closest points of comparison:

| | LangGraph-native | Mem0 | Zep/Graphiti | Cognee | Letta | **langgraph-beads-memory** |
|---|---|---|---|---|---|---|
| Postgres-only | Strong | Weak (graph mode needs Neo4j) | Absent (Neo4j) | Strong | Adequate | **Strong** |
| Typed fact graph | Absent | Weak | Strong | Adequate | Absent | **Strong** |
| Explicit (non-blind) capture | Adequate | Weak | Weak | Weak | Adequate | **Strong** |
| Enforced sub-agent fork + rollup | Absent | Absent | Absent | Absent | Adequate | **Strong** |

Full writeup, positioning, and strategic analysis in the competitive brief (linked below).

## Project status

- [x] Architecture design ([spec](docs/superpowers/specs/2026-07-31-beads-memory-design.md))
- [x] Competitive landscape research ([brief](docs/superpowers/specs/2026-07-31-beads-memory-competitive-brief.md))
- [x] Demo/benchmark design ([spec](docs/superpowers/specs/2026-08-08-beads-memory-demo-design.md))
- [x] **`langgraph-beads-memory` package** — store, middleware, tools, sub-agent fork/rollup. 65 tests against real Postgres.
- [x] **Comparison harness** — scripted scenario, both conditions, objective metrics, blinded LLM judge
- [x] **Explainer animations** — [comparison](docs/assets/comparison.svg), [mechanism](docs/assets/mechanism-full.svg)
- [x] **Scored N=3 results** — [results](results/2026-08-09-results.md); method and disclosed corrections in [results/README.md](results/README.md)
- [ ] Publish write-up

Running the demo needs Docker (Postgres + pgvector) and Ollama; see
[results/README.md](results/README.md) for exact steps.

### Honest status of the evidence

The mechanism is verified working end to end with a real LLM — forked child
namespaces, genuine `conclude_task` rollups, and `rollup_of` audit edges
confirmed against live Postgres, not just in unit tests.

On the comparison: at N=3, one metric separates cleanly — carrying a *revised*
constraint into a later thread, 0/3 for the built-in memory versus 3/3 here —
and the blinded judge favours this library on all three dimensions. Most other
metrics are tied or noisy, and the baseline beat us on one (recalling a specific
buried detail, 3/3 vs 2/3, which inspection showed was model variance rather
than architecture). The scenario was designed to exercise this mechanism, so
treat it as a demonstration on a case built for it. Full numbers, the failure
analysis, and every disclosed correction are in
[results/2026-08-09-results.md](results/2026-08-09-results.md).

## Docs

- [Architecture design](docs/superpowers/specs/2026-07-31-beads-memory-design.md) — namespace model, Postgres schema, capture mechanisms, fork/rollup, compaction, error handling
- [Competitive brief](docs/superpowers/specs/2026-07-31-beads-memory-competitive-brief.md) — landscape, positioning, opportunities/threats
- [Demo design](docs/superpowers/specs/2026-08-08-beads-memory-demo-design.md) — the scenario and methodology used to demonstrate this against plain LangGraph memory

## License

MIT — see [LICENSE](LICENSE).
