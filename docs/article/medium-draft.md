# Memory that doesn't grow: giving LangGraph agents a fact graph instead of a scratchpad

*Continuing the exploration with agents — this one is about what happens to an agent's memory over a long session, and why the usual answer gets more expensive the longer you use it.*

---

## The problem I actually hit

Agents forget between threads. That much is well known, and every framework has an answer: save memories to a store, search it later.

What bothered me was subtler. In LangGraph's built-in setup, the agent has to *decide* to save. `manage_memory` is a tool, so persistence depends on the model choosing to call it at the right moment. And recall is a second, independent decision — `search_memory`, called when the model thinks to.

Nothing reconciles the two.

I did not appreciate how badly that can fail until I watched a run where the model wrote:

> "I've recorded that correction in my memory:
>  — **Deploy time**: 13:20 UTC (not 13:50)"

It made no tool call. The memory was empty. It then searched that empty store three times across later turns and reported, accurately, "I don't have any recorded information about this incident yet."

The model wasn't lying, exactly. It narrated the action instead of performing it, and nothing in the architecture noticed.

## What I built

A memory middleware for LangGraph that stores a **typed graph of individual claims** in Postgres, rather than saved documents in a key-value store.

Two things fall out of that choice, and they turn out to be the whole story:

- **Capture doesn't route through a model decision.** Every user message and every final answer is written in `before_model` / `after_model`. No tool call, no extraction LLM.
- **Retrieval returns claims, not documents.** Because facts are stored one claim at a time, what comes back is small.

Everything else in the design — supersede edges, sub-agent namespaces, directive classification — is downstream of those two.

## Three properties, each measurable

### 1. Retrieval cost is constant

The injected block is *k* facts per call, whatever the store holds. Measured across two scenarios while the store grew by an order of magnitude:

| | store grew to | injected per call |
|---|---|---|
| incident demo | 9,250 chars (12×) | 8 facts · 596–961 chars |
| vecdb demo | 7,655 chars (10×) | 8 facts · 725–1,065 chars |

Once there are more than *k* facts to choose from, injection stops tracking the store entirely. A session can accumulate indefinitely without the per-turn bill following it.

### 2. The payload is small, because a claim is not a document

Same turn, same question:

```
stock       3,653 chars  (~913 tokens)   N documents × whatever the agent saved
fact graph    793 chars  (~198 tokens)   k claims    × one claim
```

The stock ceiling is soft — save bigger blobs and retrieval grows with them. Per-claim capture makes ours hard.

There's a second effect I didn't anticipate and only found by reading per-call token counts. Stock recall arrives as a **tool result in the message history**, so it's re-sent on every later model call in the same turn. Input climbs ~710 → ~1,600 → ~2,400 tokens across three calls. An injected fact block lives in the **system prompt**, which is rewritten each call — paid once, replaced, never stacked.

### 3. What comes back is relevant, because the graph is typed

Ranking isn't similarity alone:

- **`directive` facts are held out.** Questions rank highly against a query precisely *because they resemble it*. In one measured run, four of eight injected slots were fragments of the question being asked.
- **Superseded facts are retired** from retrieval but kept for audit.
- **Sub-agent facts are demoted**, so raw exploration stays reachable without displacing the parent's constraints.
- **Conversational framing is never stored.** "New shift taking over." was once the top-ranked fact for "what should we try next."

## What it measured

One instrumented run, incident scenario, `gemma4:12b`:

| | stock memory | fact graph |
|---|---|---|
| input tokens | 23,561 | 16,745 |
| objective metrics passed | 7 of 8 | 7 of 8 |

Same accuracy, 29% less input. Across four paired comparisons on two models, the token direction held at −29%, −29%, −32% wherever the baseline actually stored something.

The exception is instructive: on the run where the baseline never saved anything, the fact graph cost 4% *more* — because it answered the questions the baseline skipped.

## The part that took the longest: my own bugs

I want to be honest about the ratio here, because it's the actual lesson.

Most of the time went into finding defects in my own measurement, not in the library. A partial list from one day:

- A metric that scored a **correct** answer as wrong, because the window around "connection pool" contained the word "checkout" and my proposal-detector matched `check` as a substring. That single false positive was the *entire* apparent advantage on the headline metric.
- The same substring bug in the library, where a sentence beginning "Checkout p99 latency went…" was classified as the imperative "check" and therefore held out of retrieval. The central fact of the scenario was silently unretrievable.
- A resilience policy I'd written, documented, and credited in a results file — which could never have worked. It set the client to `None` before retrying, and the underlying library raises on a null client rather than rebuilding it. Every retry failed before touching the network.
- Agent conclusions stored as single 1,900-character blobs while user messages were split per claim. That inconsistency grew to 58% of everything stored, and it was why the "efficient" memory layer was initially *more* expensive than the baseline.

The pattern in all of them: I wrote the check from imagination, and real output contradicted it. Every regression test in the repo now uses sentences taken verbatim from runs.

## What I got wrong on purpose, and wrote down first

For the second scenario I committed predictions to git before the first run, including the two metrics I expected the baseline to win.

I got one of them backwards with high confidence. I predicted flat blob storage would retain an incidental measurement better than a summarising sub-agent boundary. It didn't — the fact graph recalled it and the baseline didn't.

Without the commitment timestamp I'd have had a tidy post-hoc story either way. That's the entire value of writing it down.

## What's still open

- **Supersede is shallow.** Retiring a fact retires that row and nothing else, so a *paraphrased* restatement of a corrected value survives. I can see it in the injection log: the stale `$100,000` and its own `$50,000` correction reaching the model two ranks apart, 0.002 apart in cosine distance. Cosine cannot separate them; normalised figures probably can.
- **The scale claim is untested.** Constant retrieval cost is demonstrated at ~9,000 characters of memory. The interesting number is where a document-based store's retrieval starts to strain a context window and this doesn't. That needs a much bigger scenario.
- **N is small.** One run per model per scenario so far.

## Where it landed

There was plenty of trial and error — considerably more of it in the measurement than in the library. But the shape feels solid enough to state as a practice:

**If memory is a graph of claims rather than a pile of documents, recall becomes a bounded, replaceable block instead of an accumulating one — and it stops depending on the model remembering to write things down.**

Code, the full benchmark harness, every disclosed correction, and the diagrams: [github.com/masterkidan/langgraph-beads-memory](https://github.com/masterkidan/langgraph-beads-memory)

*Next: how the benefit differs by model — five models from five vendors, and why the answer isn't "the better model wins."*
