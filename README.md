# langgraph-beads-memory

Beads-style durable memory for [LangGraph](https://github.com/langchain-ai/langgraph) agents on Postgres — a typed fact/conclusion graph with explicit capture (not blind auto-extraction) and enforced sub-agent memory forking with rollup summaries, instead of an opaque conversation summary.

> Status: **implemented and measured.** 238 tests against real Postgres.
> Retrieval cost is constant in the size of the store, the payload is ~5×
> smaller than document-based recall, and ranking is typed rather than
> similarity-only. Measured on `gemma4:12b`, N=1: 16,745 input tokens against
> 23,561 at equal accuracy.
> Method and every disclosed correction: [results/README.md](results/README.md).
> How the benefit differs by model is a separate study:
> [results/model-study.md](results/model-study.md).

## What it gives you

Three properties. Each is a consequence of storing memory as a typed graph of
individual claims rather than as saved documents, and each is measured.

### 1 · Retrieval cost is constant

The injected block is **k facts per call, whatever the store holds.** Measured
across two scenarios while the store grew by an order of magnitude:

| | store grew to | injected per call |
|---|---|---|
| incident | 9,250 chars (12×) | 8 facts · 596–961 chars |
| vecdb | 7,655 chars (10×) | 8 facts · 725–1,065 chars |

Once there are more than `k` facts to choose from, injection stops tracking the
store. A session can accumulate indefinitely without the per-turn bill following
it — recall cost is set by `k` and by the size of one claim, both constants.

### 2 · The payload is small, because a claim is not a document

`search_memory` returns whole saved documents. This returns the claims that
matter. Same turn, same question:

```
stock       3,653 chars  (~913 tokens)   N documents × whatever the agent saved
fact graph    793 chars  (~198 tokens)   k claims    × one claim
```

Per-claim capture is what makes that bound hard. The stock ceiling is soft: save
bigger blobs and retrieval grows with them.

And it does not accumulate. Stock recall arrives as a **tool result in the
message history**, so it is re-sent on every later call in the turn — input
climbs ~710 → ~1,600 → ~2,400 tokens across three calls. Here recall lives in
the **system prompt**, rewritten each call, so it is paid once and replaced.

### 3 · What comes back is relevant, because the graph is typed

Ranking is not similarity alone. Kind, status and provenance all participate:

| | effect |
|---|---|
| `directive` facts | held out — questions rank highly by *resembling* the query; four of eight slots were once question fragments |
| `superseded` facts | retired from retrieval, kept for audit — a corrected value cannot resurface |
| descendant facts | demoted — a sub-agent's raw exploration stays reachable without displacing the parent's constraints |
| framing | never stored — "New shift taking over." was once the top-ranked fact for "what should we try next" |

![One turn, call by call. Stock memory returns whole documents as a search_memory tool result that lands in the message history and is re-sent on every later call, so input grows from ~710 to ~2,400 tokens across the turn. The fact graph injects eight ranked claims into the system prompt, which is rewritten each call rather than accumulated, so input stays roughly flat.](docs/assets/token-mechanism.svg)

### Measured effect

One instrumented run, incident scenario, `gemma4:12b`, N=1:

| | stock memory | this library |
|---|---|---|
| input tokens | 23,561 | 16,745 |
| output tokens | 1,887 | 1,708 |
| objective metrics passed | 7 of 8 | 7 of 8 |
| stored | 1,854 chars | 9,250 chars |

Same accuracy, 29% less input. The store being larger is incidental — it is
recorded because it shows retrieval cost is decoupled from store size, not
because storing more is itself useful.

N=1 on one model. The direction has held across four earlier N=3 rounds on a
different model, but the magnitude has varied (−36%, −45%, −29%), so treat it
as "meaningfully cheaper" rather than as a fixed figure.

You can read the ranking off any run rather than taking it on faith:

```bash
uv run python -m demo.show_memory results/fresh-gemma/incident --turn conv-3
```

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
| sub-agents | results return as messages; no link back to what produced them | own namespace, enforced `conclude_task`, `rollup_of` audit edges |
| a sub-agent's raw findings | mixed into one flat store | demoted in the orchestrator's retrieval, and directly readable via `recall_from_subagents` |
| a crashed sub-agent | silently returns nothing | wrapper synthesizes a "did not complete" fact |

**What the built-in option does well, and what this costs.** The checkpointer gives complete message history within a thread, `BaseStore` has real vector search, and it's first-party with no extra dependency — when the agent does save a memory, cross-thread recall genuinely works. This library adds a dependency and a Postgres schema.

*One caveat on the token figures:* this library also trims the message window to the last 10 messages. In these turns (2–4 calls each) that almost certainly never binds, so it is unlikely to be contributing — but it has not been isolated, and it would matter in longer turns.

## How it works

![How langgraph-beads-memory works: user messages and agent conclusions become durable facts on a session-wide memory string; sub-agents fork isolated namespaces and roll summaries back; a revised budget supersedes the stale one; a later conversation on a new thread recalls only active facts.](docs/assets/mechanism.svg)

One session (`session_id: vecdb-research`) spanning three conversations. Facts are beads on the session string — threads come and go, the string stays.

1. **Capture** — every user message is written verbatim onto the session string, no extraction LLM involved. The turn's final answer is captured as a conclusion automatically, so durable memory never depends on the model remembering a tool call.
2. **Fork** — delegating research gives each sub-agent its own child namespace. It reads upward (its ancestors) and never sideways (a sibling). The orchestrator reads downward too, demoted — see [Ranking](#ranking).
3. **Enforced rollup** — each sub-agent must call `conclude_task`. One summary lands on the parent, linked by `rollup_of` edges back to its raw exploration. A crashed sub-agent leaves a "did not complete" fact rather than vanishing.
4. **Supersede** — when the user revises a constraint, the new fact supersedes the old one. The stale value is retired from retrieval but kept for audit.
5. **Recall** — a brand-new thread starts warm, and only *active* facts can reach the model. This is exactly where thread-scoped memory starts cold, and where extraction stores surface both the old and new value with nothing marking which is current.

## Where memory is written, and how it is ranked

![Four write triggers converge on one path — split per claim, drop conversational framing, classify statement or directive, derive a content-addressed id — then retrieval scopes to the namespace and its ancestors and descendants, applies four filters, ranks by cosine distance plus a penalty for descendant facts, and injects the top eight.](docs/assets/memory-pipeline.svg)

Every filter in that pipeline traces to a measured failure rather than a design preference — the notes under the diagram say which. Two examples: the query is taken from the full message list because taking it from the trimmed window meant a turn with enough tool calls produced no query at all and memory silently switched off; and `directive` facts are held out because questions rank highly against a query precisely by resembling it, and four of eight injected slots were once question fragments.

You can read this off any run rather than taking it on faith:

```bash
uv run python -m demo.show_memory results/fresh-gemma/incident --turn conv-3
```

which prints the ranked facts that actually reached the model, with cosine distances and demotion flags, plus what the run stored broken down by kind and source.

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

## Using it

```python
from beads_memory import BeadsMemoryMiddleware, BeadsStore, OllamaEmbedder, make_subagent_tool

store = BeadsStore(conn)          # any psycopg connection; one per thread
store.init_schema()
ns = store.get_or_create_namespace("vecdb-research")   # the session scope

agent = create_agent(
    model=llm,
    tools=[...],
    middleware=[BeadsMemoryMiddleware(
        store=store, namespace=ns, embedder=OllamaEmbedder(),
        agent_id="root", acting_on_behalf_of="user",
    )],
)
```

Capture and injection then happen automatically; there are no store calls to
write in the common path.

**Tools the middleware binds for the agent**

| tool | who gets it | what it does |
|---|---|---|
| `remember_fact` | every agent | record a conclusion, optionally `supersedes`/`contradicts`/`relates_to` an existing fact by short id |
| `conclude_task` | forked sub-agents | required before returning; writes one summary into the parent with `rollup_of` edges back to the raw work |
| `recall_from_subagents` | orchestrators only | read what a named sub-agent actually recorded, past its one-line summary |

`recall_from_subagents` exists because demoted search is a *guess* — a child's
fact surfaces only if the query happens to match it. An orchestrator usually
knows something stronger: it delegated a topic to a named researcher. This lets
it look rather than hope. It is bound only where `capture_final` is set, the
same flag that distinguishes a root agent from a fork, so sub-agents cannot use
it to read siblings.

**Reading the store directly**

```python
store.search(ns.id, embedder.embed(q), k=8)   # self + ancestors + demoted descendants
store.children(ns.id)                          # direct sub-namespaces
store.subtree_facts(ns.id, agent_id="researcher_qdrant")   # what one sub-agent found
store.facts_in_namespace(ns.id)                # everything here, including retired
```

`search` and `subtree_facts` return only `active` facts. `facts_in_namespace`
does not filter, which is how you audit what was superseded and by what.

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
- [x] **`langgraph-beads-memory` package** — store, middleware, tools, sub-agent fork/rollup. 238 tests against real Postgres.
- [x] **Comparison harness** — two scenarios, four arms, objective metrics, blinded LLM judge with a grounding dimension
- [x] **Instrumentation** — every run records what retrieval injected (with cosine distances) and a snapshot of what it stored, so rankings are read rather than reconstructed: `uv run python -m demo.show_memory <run-dir>`
- [x] **Diagrams** — [comparison](docs/assets/comparison.svg), [mechanism](docs/assets/mechanism-full.svg), [write + ranking pipeline](docs/assets/memory-pipeline.svg), [why it costs less context](docs/assets/token-mechanism.svg)
- [x] **Scored results, four N=3 rounds on `qwen3:8b`** (newest first):
  - [2026-08-11 · demoted descendant recall](results/2026-08-11-descendant-results.md) — read its attribution section first
  - [2026-08-10 · directive fix](results/2026-08-10-directive-results.md)
  - [2026-08-10 · granularity + supersede guard](results/2026-08-10-postfix-results.md)
  - [2026-08-09 · first scored run](results/2026-08-09-results.md)
- [x] **Pre-registered second scenario** — [predictions committed before the first run](results/2026-08-11-demo2-preregistration.md), including the two metrics the baseline was expected to win
- [x] **Instrumented N=1 pairs** on `gemma4:12b` and `qwen3.5:9b` — `results/fresh-gemma/`, `results/fresh-qwen35/`
- [ ] **[Model study](results/model-study.md)** — how the benefit differs by model; 2 of 5 measured, its own write-up
- [ ] N=5 on the configuration that wins
- [ ] A scale scenario large enough to find the token crossover
- [ ] Publish write-up

Method, every disclosed correction, and the operational notes are in
[results/README.md](results/README.md). It is long on purpose: several rounds
ran with bugs that were later found and fixed, and each one is recorded with
what it changed rather than quietly re-run.

Running the demo needs Docker (Postgres + pgvector) and Ollama; see
[results/README.md](results/README.md) for exact steps.

### Honest status of the evidence

The mechanism is verified working end to end with a real LLM — forked child
namespaces, genuine `conclude_task` rollups, and `rollup_of` audit edges
confirmed against live Postgres, not just in unit tests.

On the comparison, from the [latest N=3 round](results/2026-08-11-descendant-results.md)
(the first with zero errored turns): the clearest separation is committing to a
*feasible* option — 3/3 here versus 0/3 for the built-in memory, which
recommended Weaviate in every run at roughly $60k/yr against a revised $50k
budget, citing no budget figure at all. Carrying the revised budget forward is
2/3 here versus 0/3. The blinded judge favours this library on all three
dimensions, though it also scored a run 5/5 on "recall" whose answer contained a
fabricated number, so weight it accordingly.

**Two things worth stating plainly.** The baseline is not a strawman here: it
called `manage_memory` four times per run and `search_memory` when asked,
reliably, in every run — and it beats us on recalling a specific buried detail
(3/3 vs 2/3). And when the targeted metric for that round *did* improve, tracing
the database showed only **one of three runs** was actually attributable to the
feature under test; the rest was variance in what sub-agents chose to summarise.
Single-metric movements of 1/3 at this N are not distinguishable from noise.

> **⚠ Model is an uncontrolled variable in these numbers.**
> Every round above ran on `qwen3:8b`. In a first clean run on `gemma4:12b`
> (2026-08-12) the **baseline carried the revised budget correctly** — the
> metric that is 0/3 for the baseline in every qwen3 round and forms the
> headline separation here. It also committed to a feasible option, which it
> never did on qwen3.
>
> A second run on the same model then went the other way (1 of 2), so this is
> variance, not a finding. What it establishes is narrower and methodological:
> **model is a variable these rounds did not control for.** That is why the
> benchmark now runs the same scenarios across five models from five vendors —
> the question "does structured memory help" is only answerable once the model
> is held constant or varied deliberately.
>
> Read the per-round numbers as measurements of one implementation on one
> model, not as the architecture's ceiling. Several of the rounds above also
> ran with bugs since fixed (see the corrections in
> [results/README.md](results/README.md)), all of which degraded one arm or
> the other.
>
> Several other bugs found on 2026-08-12 also degraded only the baseline
> (empty sub-agent returns, tool calls rejected at the schema boundary, a retry
> loop). All are fixed, and all of them ran during the rounds above. See
> [results/README.md](results/README.md).

The scenario was designed to exercise this mechanism, so treat it as a
demonstration on a case built for it. Full numbers, the attribution analysis and
every disclosed correction are in
[results/2026-08-11-descendant-results.md](results/2026-08-11-descendant-results.md),
with the round-by-round index in [results/README.md](results/README.md).

## Docs

- [Architecture design](docs/superpowers/specs/2026-07-31-beads-memory-design.md) — namespace model, Postgres schema, capture mechanisms, fork/rollup, compaction, error handling
- [Competitive brief](docs/superpowers/specs/2026-07-31-beads-memory-competitive-brief.md) — landscape, positioning, opportunities/threats
- [Demo design](docs/superpowers/specs/2026-08-08-beads-memory-demo-design.md) — the scenario and methodology used to demonstrate this against plain LangGraph memory

## License

MIT — see [LICENSE](LICENSE).
