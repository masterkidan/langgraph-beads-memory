# Constant-cost memory for LangGraph agents

A typed fact graph on Postgres, as agent middleware. Retrieval returns a fixed number of individual claims rather than a growing set of saved documents — which makes recall cost independent of how much the session has accumulated.

---

## The problem with document-shaped memory

LangGraph's built-in memory works, and for many applications it is enough. But two properties of it become expensive as sessions get long.

**Save and recall are independent model decisions.** `manage_memory` is a tool, so persistence depends on the model choosing to call it. `search_memory` is another tool, called when the model thinks to. Nothing reconciles them, so an agent can search a store it never wrote to and correctly report that it is empty.

This is not hypothetical. In one benchmark run, a model produced:

> "I've recorded that correction in my memory:
>  — **Deploy time**: 13:20 UTC (not 13:50)"

with no tool call. The store stayed empty, and three later turns answered "I don't have any recorded information about this incident yet." Tool-call counts for that run: `manage_memory` absent, `search_memory` ×3.

**Recall arrives as a message, and messages accumulate.** A `search_memory` result is a tool message. It sits in the history and is re-sent on every subsequent model call in that turn.

## The approach

Store memory as a graph of individual claims, captured in middleware rather than through a tool.

- **Capture is not a decision.** Every user message and every final answer is written in `before_model` / `after_model`. No tool call, no extraction LLM.
- **A fact is one claim.** "The budget is $100k, it must be self-hostable, and I only trust primary benchmark data" becomes three facts, each separately embedded and separately supersedable.
- **Facts are typed.** `user_input`, `conclusion`, `summary`, `directive` — with `active` / `superseded` / `archived` status and typed edges (`supersedes`, `rollup_of`, `contradicts`, `relates_to`).

## Three properties

### 1. Retrieval cost is constant

The injected block is *k* facts per call, whatever the store holds. Measured across two scenarios while the store grew by an order of magnitude:

| | store grew to | injected per call |
|---|---|---|
| incident scenario | 9,250 chars (12×) | 8 facts · 596–961 chars |
| vecdb scenario | 7,655 chars (10×) | 8 facts · 725–1,065 chars |

Once there are more than *k* facts to choose from, injection stops tracking the store. Recall cost is set by *k* and by the size of one claim — both constants.

### 2. The payload is small, because a claim is not a document

Same turn, same question:

```
document store   3,653 chars  (~913 tokens)   N documents × whatever was saved
fact graph         793 chars  (~198 tokens)   k claims    × one claim
```

The document-store ceiling is soft: save larger blobs and retrieval grows with them. Per-claim capture makes the fact-graph ceiling hard.

Placement compounds this. A tool result persists in the message history and is re-sent on every later call in the turn — input climbs roughly 710 → 1,600 → 2,400 tokens across three calls. An injected fact block lives in the system prompt, which is rewritten each call: paid once, replaced, never stacked.

Measured across a full turn: **13 model calls against 16**, because recall costs no round trip.

### 3. Relevance comes from types, not similarity alone

Cosine distance alone is a poor ranker for agent memory, because the things that most resemble a question are other questions.

| | effect |
|---|---|
| `directive` facts | held out of retrieval — in one measured run, four of eight injected slots were fragments of the question being asked |
| `superseded` facts | retired from retrieval, kept for audit |
| descendant facts | demoted, so a sub-agent's raw exploration stays reachable without displacing the parent's constraints |
| conversational framing | never stored — "New shift taking over." once ranked first for "what should we try next" |

## Measured effect

One instrumented run, incident scenario, `gemma4:12b`:

| | document store | fact graph |
|---|---|---|
| input tokens | 23,561 | 16,745 |
| output tokens | 1,887 | 1,708 |
| objective metrics passed | 7 of 8 | 7 of 8 |
| stored | 1,854 chars | 9,250 chars |

Same accuracy, 29% less input. Store size is included because it shows retrieval cost is decoupled from it, not because storing more is useful in itself.

Across four paired comparisons on two models, the token direction held at −29%, −29%, −32% wherever the document store had something to retrieve. The exception is the run above where it stored nothing: there the fact graph cost 4% more, because it answered questions the other arm skipped.

## Sub-agents get their own memory

Delegation forks a child namespace. A sub-agent reads its own facts plus its ancestors', never a sibling's — which is what stops three parallel researchers contaminating each other. The parent additionally reads its whole subtree, demoted.

Each sub-agent must call `conclude_task`; one summary crosses into the parent, linked by `rollup_of` edges back to the raw exploration. A sub-agent that fails to conclude gets a summary reconstructed from what it recorded, and one that recorded nothing produces an explicit "did NOT complete — conclusions are MISSING" rather than a silent empty result.

## Auditability

Nothing is deleted. A superseded fact stays queryable, so "what did we believe before, and what replaced it?" is always answerable.

Retrieval is inspectable per run — the ranked facts that actually reached the model, with cosine distances and demotion flags:

```bash
uv run python -m demo.show_memory results/fresh-gemma/incident --turn conv-3
```

## Limits

- **Invalidation is only as granular as capture.** A correction now cascades along `derived_from` edges recorded at capture time, and retires facts that assert the superseded value but not its replacement. What it cannot do is partially retire a fact: one stored claim that mixed a stale budget with two still-valid constraints was retired whole, and the live constraints went with it.
- **Constant retrieval cost is demonstrated at ~9,000 characters of memory.** Where a document store's retrieval begins to strain a context window has not been measured.
- **N is small.** One run per model per scenario so far.

## Repository

Library, benchmark harness, both scenarios, the instrumentation, and the full method: [github.com/masterkidan/langgraph-beads-memory](https://github.com/masterkidan/langgraph-beads-memory)
