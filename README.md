# langgraph-beads-memory

Beads-style durable memory for [LangGraph](https://github.com/langchain-ai/langgraph) agents on Postgres — a typed fact/conclusion graph with explicit capture (not blind auto-extraction) and enforced sub-agent memory forking with rollup summaries, instead of an opaque conversation summary.

> Status: **design phase**. The architecture below is specced and committed; implementation hasn't started yet. See [Project status](#project-status).

## How it works

![How langgraph-beads-memory works: user messages and agent conclusions become durable facts on a session-wide memory string; sub-agents fork isolated namespaces and roll summaries back; a revised budget supersedes the stale one; a later conversation on a new thread recalls only active facts.](docs/assets/mechanism.svg)

One session (`session_id: vecdb-research`) spanning three conversations. Facts are beads on the session string — threads come and go, the string stays.

1. **Capture** — every user message is written verbatim onto the session string, no extraction LLM involved. The turn's final answer is captured as a conclusion automatically, so durable memory never depends on the model remembering a tool call.
2. **Fork** — delegating research gives each sub-agent an isolated child namespace. It reads its ancestors; it never sees its siblings.
3. **Enforced rollup** — each sub-agent must call `conclude_task`. One summary lands on the parent, linked by `rollup_of` edges back to its raw exploration. A crashed sub-agent leaves a "did not complete" fact rather than vanishing.
4. **Supersede** — when the user revises a constraint, the new fact supersedes the old one. The stale value is retired from retrieval but kept for audit.
5. **Recall** — a brand-new thread starts warm, and only *active* facts can reach the model. This is exactly where thread-scoped memory starts cold, and where extraction stores surface both the old and new value with nothing marking which is current.

## The problem

LangGraph ships two memory primitives: a checkpointer for thread-scoped state, and a `BaseStore`/`PostgresStore` for cross-thread key-value memory. Frameworks built on top (LangMem, Mem0, Zep) mostly bet on automatic LLM extraction — scan the transcript, pull out "facts," write them somewhere. That's fast to wire up, but it's also imprecise, hard to audit, and gives you no way to say "this conclusion replaced that one" or "this sub-agent's exploration shouldn't pollute the parent's context."

[beads](https://github.com/steveyegge/beads) — Steve Yegge's dependency-aware issue tracker for coding agents — took a different stance for task memory: a typed graph of issues, explicit `bd remember` calls for durable insight, and semantic decay instead of silent deletion. `langgraph-beads-memory` brings that same stance to LangGraph's conversational/multi-agent memory, backed by nothing but Postgres.

## What it does differently

- **Explicit, dual-path capture.** User input is captured verbatim as it arrives — no extraction LLM call needed. Agent conclusions are captured only when the agent deliberately calls a `remember_fact` tool. Nothing gets written to memory the agent (or user) didn't put there.
- **A typed fact graph, not a blob.** Facts relate to each other through typed edges — `supersedes`, `contradicts`, `relates_to`, `derived_from`, `rollup_of` — so "this replaced that" and "this summary was derived from these five facts" are first-class, queryable relationships.
- **Durable, auditable, enforced sub-agent rollups.** LangGraph sub-agents already have isolated context — that's not the claim. The claim: when a supervisor spawns a sub-agent, its memory forks into a child namespace, and its conclusion is *required* — the adapter synthesizes a "task did not complete" fact if the sub-agent crashes or forgets, so no delegated task ever vanishes silently. The rollup summary is a durable fact (not a message that scrolls away), linked by `rollup_of` edges back to every exploration fact that produced it, so any conclusion can be audited by drill-down. Raw exploration stays in the child namespace; only the rollup reaches the parent.
- **Postgres-only.** No Neo4j, no separate vector database — namespaces, facts, and edges all live in one Postgres schema (`pgvector` for embeddings).
- **Session-scoped, not identity-coupled.** The core schema is anchored on `session_id` alone — a long-lived memory scope that deliberately spans LangGraph threads, so a new conversation that continues the same work recalls everything from earlier ones. `langgraph-beads-memory` doesn't need to know what a "user" is; if an application wants a user↔session mapping, it owns that table itself.

## How it fits into a LangGraph app

It's an **agent middleware** (LangGraph's `create_agent` pre/post-model hook API) — not a `BaseStore` implementation and not a checkpointer replacement. Thread-level state/replay stays with LangGraph's own `PostgresSaver`; this owns a separate schema for durable, structured memory and wires in purely through hooks, so there are no explicit store calls to write in the common path.

Per turn, the middleware:
1. Passively captures new user messages as facts (no LLM call).
2. Keeps a sliding window of the last ~10 raw messages in context; older messages get distilled into facts as they roll off.
3. Runs semantic search over the current namespace (plus ancestor read-through if forked) and injects the top-K relevant facts.

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

This repo currently holds the design work; implementation is next.

- [x] Architecture design ([spec](docs/superpowers/specs/2026-07-31-beads-memory-design.md))
- [x] Competitive landscape research ([brief](docs/superpowers/specs/2026-07-31-beads-memory-competitive-brief.md))
- [x] Demo/benchmark design ([spec](docs/superpowers/specs/2026-08-08-beads-memory-demo-design.md)) — a local, scripted 3-session scenario comparing this adapter against LangGraph's `BaseStore`+LangMem baseline on recall accuracy, delegation quality, and final-answer quality, judged by an LLM rubric
- [ ] `langgraph-beads-memory` package (demo-scoped subset)
- [ ] Comparison harness + results
- [ ] Explainer animation
- [ ] Publish write-up

## Docs

- [Architecture design](docs/superpowers/specs/2026-07-31-beads-memory-design.md) — namespace model, Postgres schema, capture mechanisms, fork/rollup, compaction, error handling
- [Competitive brief](docs/superpowers/specs/2026-07-31-beads-memory-competitive-brief.md) — landscape, positioning, opportunities/threats
- [Demo design](docs/superpowers/specs/2026-08-08-beads-memory-demo-design.md) — the scenario and methodology used to demonstrate this against plain LangGraph memory

## License

MIT — see [LICENSE](LICENSE).
