# langgraph-beads-memory: A memory adapter for LangGraph on Postgres, inspired by beads

Status: Design approved, not yet implemented.
Date: 2026-07-31
Package name finalized 2026-08-08: `langgraph-beads-memory` (PyPI/GitHub),
referred to in prose below as beads-memory.

## 1. Overview & Goals

A Python package (`langgraph-beads-memory`) that plugs into LangGraph's `create_agent`
middleware API and gives agents beads-style durable memory on Postgres: a typed
fact/conclusion graph instead of an opaque conversation summary, explicit capture
(not blind auto-extraction), and namespace-scoped forking for sub-agents that
mirrors beads' epic/sub-task rollup pattern.

Prior art checked (2026-07-31): LangGraph ships `BaseStore`/`PostgresStore`
(cross-thread JSON key-value memory) and checkpointers (thread-level state/replay).
LangMem adds LLM-based extraction and procedural memory. None of these provide a
dependency/fact graph, fork-per-subagent semantics, or a facts-vs-conclusions
distinction the way beads separates issues from `bd remember`. This does not
duplicate existing functionality.

Non-goals:
- Not a checkpointer replacement — thread-level state/replay stays with
  LangGraph's own `PostgresSaver`.
- Not a `BaseStore` implementation — the relational/graph shape doesn't fit
  `BaseStore`'s namespace+key+JSON-blob model. This is a separate Postgres
  schema, owned by this package, wired in purely through middleware hooks.

## 2. Integration point

Agent middleware (LangGraph's `create_agent` pre/post-model hook API), not an
explicit graph node and not a checkpointer extension. Facts are captured and
injected automatically around model calls; the graph author does not write
explicit store calls for the common path.

## 3. Namespace model

*Amended 2026-08-07: dropped `user_id` as a core/required field. See rationale
below.*

A namespace is anchored on one required field plus an optional extra path for
sub-scoping (forks, task-level isolation, etc.):

- `session_id` — required; this is the same value as LangGraph's `thread_id`,
  renamed because "thread" is overloaded in a memory context. Not a broader
  multi-thread concept.
- `extra_path` — array, defaults to empty; used for forked/child namespaces,
  e.g. `('task', 'sub-a1b2')`.

`user_id` is **not** part of the core schema. beads-memory's unit of memory is the
session, not the user — this keeps the adapter usable in contexts where
"user" isn't a meaningful concept (a CI job, a one-off script) and avoids
coupling the memory schema to an app's auth/identity model. If an application
needs to know which user owns which session, it maintains its own
`user_sessions` (or equivalent) mapping table outside beads-memory's schema —
beads-memory never joins against it.

Forked sub-agent namespaces get a **short random hash suffix** (beads-style),
not a sequential counter — e.g. `task/sub-a1b2` — so concurrent spawns from the
same parent can never collide. Do not rely on a DB unique-constraint-and-retry
scheme for this; the hash avoids the round-trip entirely.

## 4. Postgres schema

```sql
namespaces
  id              uuid pk
  session_id      text not null
  extra_path      text[] not null default '{}'
  parent_id       uuid fk -> namespaces.id, nullable   -- set for forked/child namespaces
  created_at      timestamptz
  unique (session_id, extra_path)

facts
  id                    uuid pk   -- deterministically derived, see 5.4 (idempotency)
  namespace_id          uuid fk -> namespaces.id
  session_id            text not null   -- denormalized from namespace, indexed
  kind                  text not null   -- 'user_input' | 'conclusion' | 'summary'
  body                  text not null
  embedding             vector(N)       -- pgvector; null until embedded (async, see 6)
  status                text not null   -- 'active' | 'superseded' | 'archived'
  source                text not null   -- 'passive_capture' | 'remember_tool'
                                         -- | 'conclude_task' | 'fallback_conclude'
                                         -- | 'compaction'
  agent_id              text not null   -- which agent wrote this fact
  acting_on_behalf_of   text not null   -- parent agent_id, or the sentinel
                                         -- 'user' if this fact's agent is root
  created_at            timestamptz

fact_edges
  id              uuid pk
  from_fact_id    uuid fk -> facts.id
  to_fact_id      uuid fk -> facts.id
  relation        text not null   -- 'supersedes' | 'contradicts' | 'relates_to'
                                   -- | 'derived_from' | 'rollup_of'
  created_at      timestamptz
```

`user_sessions` (optional, app-owned, outside beads-memory's schema): if an app
wants to look up "all sessions for user X," it maintains its own table
mapping `user_id -> session_id`. beads-memory's queries never require it.

`rollup_of` points from a sub-agent's `conclude_task` summary fact to every fact
created in its child namespace (audit trail / drill-down). `derived_from` is
what compaction uses when it archives raw facts into a summary.

Default retrieval excludes `status IN ('archived', 'superseded')`. Whenever a
`supersedes` edge is created (via `remember_fact(..., relation='supersedes')`
or `conclude_task(..., supersedes=<fact_id>)`), the target fact's `status` is
flipped to `superseded` in the same write — this is the only path that sets
that status.

## 5. Capture mechanisms

Three write paths, all into `facts`:

1. **Passive user-input capture.** On the pre-model hook, any new `HumanMessage`
   since the last hook invocation is written as `kind='user_input'`,
   `source='passive_capture'`, verbatim — no LLM call.
2. **`remember_fact(body, relates_to=None, relation=None)` tool.** Bound into
   the agent's toolset by the middleware. The agent calls it deliberately for
   conclusions it reaches. `source='remember_tool'`.
3. **`conclude_task(summary, supersedes=None)` tool.** Only bound in a forked
   (sub-agent) namespace. Writes one `kind='summary'` fact into the **parent**
   namespace, plus a `rollup_of` edge from that fact to every fact created in
   the child namespace during the task.

### 5.1 Fork/rollup model (sub-agent spawning)

When a supervisor spawns a sub-agent with a task summary, the adapter provides
a `make_subagent_tool(subgraph, ...)` wrapper that:

- Creates a forked child namespace (hash-suffixed `extra_path`, see §3).
- Gives the sub-agent a read-only view of its full ancestor chain (parent,
  grandparent, ... up to root) — **not** its siblings; parallel sub-agent
  forks never see each other's facts. Isolation holds between concurrent
  children.
- Requires the sub-agent to call `conclude_task` before returning. This is
  **enforced by the adapter**, not left as an optional tool: if the sub-agent's
  run ends without calling it, the wrapper synthesizes a fallback summary fact
  itself (`source='fallback_conclude'`), using the last AI message if one
  exists, or — if the sub-agent errored before producing any output — a
  system-generated "task did not complete" summary. Either way a `rollup_of`
  edge is written back to whatever child facts exist (possibly none). The
  parent namespace is never left silently unaware that a spawned task failed.
- Raw exploration facts stay in the child namespace — queryable for
  audit/drill-down, but excluded from the parent's default retrieval.

### 5.2 Middleware hook lifecycle (per turn)

**Pre-model hook:**
1. Passive-capture new human messages (idempotently, see §5.4).
2. Trim/window raw messages to the last N (default 10, configurable); anything
   rolling off the window is handed to compaction (§6) before being dropped
   from raw context.
3. Semantic search over `facts` for the current namespace + full ancestor
   read-through (§5.1), inject top-K as context (K configurable, default TBD
   at plan time).

**Post-model hook:** none required — `remember_fact`/`conclude_task` already
write synchronously when the agent invokes them as tool calls.

### 5.3 Actor identity

Every fact records `agent_id` (who wrote it) and `acting_on_behalf_of` (a
single-hop pointer: the parent agent_id, or the sentinel `'user'` if this
agent is the root — beads-memory has no `user_id` to point to, see §3). The full
delegation chain is reconstructable by walking namespaces' `parent_id` when
needed — not stored redundantly on every fact.

### 5.4 Idempotency under checkpoint replay

LangGraph can replay a node from a checkpoint (error retry, time-travel
debugging), which could re-run the pre-model hook and, without protection,
re-write facts already captured. `facts.id` is **deterministically derived**
(hash of `namespace_id` + message/tool-call id + body) and writes use
`ON CONFLICT DO NOTHING`. Replays naturally no-op.

## 6. Compaction

Two triggers, one mechanism:
- **Window overflow** (primary, normal operation): messages rolling off the
  ~10-message sliding window are distilled before being dropped from raw
  context.
- **Fact-count threshold** (background): a periodic job checks namespaces with
  many `active` facts and compacts older/superseded ones.

Process: an LLM call summarizes a batch of facts/messages into one
`kind='summary'` fact, `source='compaction'`. Each original fact gets a
`derived_from` edge from the summary and its `status` flips to `archived`.
Archived facts are excluded from default retrieval but remain queryable for
audit — mirrors beads keeping closed issues rather than deleting them.

## 7. Embeddings

Deferred/async: a fact is written immediately with `embedding=null`; a
background worker embeds it shortly after. Retrieval only searches facts that
already have embeddings. A given turn's own just-written facts may not be
searchable yet — acceptable, since they're still present in the raw sliding
window for that turn.

## 8. Error handling

- **Postgres unavailable**: fail open — the agent proceeds without memory
  injection/capture rather than blocking the conversation. Failed writes are
  logged, not retried inline.
- **Embedding worker failure**: fact stays `embedding=null` (invisible to
  semantic search) until a retry succeeds; does not block fact creation or
  the conversation.
- **Tool-call args fail validation** (e.g. bad `relates_to` id): tool returns
  a structured error to the agent via standard LangGraph tool-error handling;
  no fact is written.
- **Compaction LLM call fails**: batch left uncompacted, retried next cycle;
  facts stay `active` (no data loss, delayed compaction only).

## 9. Testing approach

- **Schema/repository layer**: integration tests against a real Postgres
  (testcontainers or similar) — namespace creation/forking, fact CRUD,
  idempotent writes under simulated replay, edge creation, compaction
  archiving.
- **Middleware hooks**: unit tests with a fake store, driving the pre-model
  hook through message-window trimming, passive capture, and retrieval
  injection — assert exact context shape without needing a real LLM.
- **Tools** (`remember_fact`, `conclude_task`): tested via LangGraph's test
  harness invoking the compiled agent with a stub model, asserting facts land
  correctly, including the mandatory-`conclude_task`-enforcement fallback
  path.
- **Fork/rollup end-to-end**: a small supervisor + sub-agent graph, asserting
  the parent namespace only gets the rollup summary (not raw child facts) and
  that `rollup_of` edges are correct.
- **Collision test**: concurrently spawn N sub-agents, assert N distinct
  namespaces with no unique-constraint violations.

## Open items for the implementation plan

- Exact embedding model/provider (pluggable interface, default TBD at plan
  time).
- Background worker mechanism for async embedding + compaction (in-process
  scheduler vs external job runner) — a deployment concern, not a schema
  concern.
