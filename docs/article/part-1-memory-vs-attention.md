# I built a memory layer for LangGraph, then measured it against doing nothing

**Part 1 — what an agent memory layer is actually worth, and when it is worth nothing.**

---

I spent a few weeks building a typed fact graph for LangGraph agents: Postgres, pgvector, per-claim capture, `supersedes` edges that retire stale values, sub-agent namespaces that fork and roll up. It works. Every mechanism is verified against a live database.

Then I benchmarked it properly and discovered it was **losing** to LangGraph's default store.

The reason turned out to be more interesting than the fix, and it changes how I think about agent memory generally. This is the write-up.

> Everything below is reproducible: [github.com/masterkidan/langgraph-beads-memory](https://github.com/masterkidan/langgraph-beads-memory). The benchmark is LongMemEval, which is external and MIT-licensed, and every disclosed correction is in the repo — including three results that evaporated when I measured them again.

---

## The benchmark mistake almost everyone makes

Here is the first comparison I ran. LongMemEval `knowledge-update`, 78 questions, a 12B model:

| arm | correct | mean input tokens |
|---|---|---|
| whole transcript in the prompt | 64/78 (82%) | 6,337 |
| LangGraph `PostgresStore` over turns | 57/78 (73%) | 2,473 |
| my fact graph | 48/78 (62%) | 328 |

Third place. I sat with that for a while before noticing the third column.

**The store used 7.5× more context than my memory layer.** That table doesn't compare memory strategies. It compares *budgets*. Of course the arm that gets 2,473 tokens beats the arm that gets 328 — it would beat it if you filled those tokens with random transcript.

This is the mistake I see in most memory benchmarks, including the ones I'd read approvingly. Each system retrieves "its natural amount," and whoever retrieves more wins on accuracy while quietly costing more. The comparison looks fair because both sides are doing retrieval.

So I rebuilt the harness to hold context constant and vary only **how it is spent**.

---

## The measurement that actually answers the question

Two arms, identical token budget, same reader model, same questions:

- **transcript only** — fill the budget with the most recent conversation
- **facts + transcript** — spend ~950 characters on retrieved facts, fill the rest with transcript

That is the real deployment shape. Nobody runs a memory layer *instead of* the conversation; they run it alongside. Then I swept the budget.

![Accuracy per token of context](../assets/accuracy-per-token.svg)

At roughly 1,200 tokens the three strategies are **36 points apart**. By roughly 5,400 tokens they are within 1.3 points — of each other, and of the ceiling.

And against LangGraph's own store, at a matched budget, the fact graph wins:

| at a 1,200-token budget, n=78 | correct | % of the ceiling | % of the context |
|---|---|---|---|
| transcript only | 23/78 | 36% | 19% |
| LangGraph `PostgresStore` | 42/78 | 66% | 15% |
| **fact graph** | **51/78** | **80%** | 19% |

**Four-fifths of what full attention achieves, on a fifth of the context.** Clipping the window to a fifth costs 64% of the ceiling with raw transcript, and 20% with the fact graph on top.

Totals can hide a lot, so here is the same 78 questions one at a time:

![Question by question outcomes](../assets/pairwise-outcomes.svg)

Against the store at a matched budget: **22 questions won, 13 lost, 29 both got right.** That's a genuine difference in behaviour, not two arms tying with noise on top.

---

## And then the part I didn't expect

Here is the same experiment across the budget range:

![The context-budget curve](../assets/context-budget.svg)

| budget | transcript only | facts + transcript | Δ |
|---|---|---|---|
| 1,200 tokens | 23/78 | **51/78** | **+28** |
| 3,000 tokens | 19/25 | 20/25 | +1 |
| 6,000 tokens | 66/78 | 65/78 | **−1** |

**The value of memory declines to zero as context becomes sufficient.**

At a tight budget the fact graph wins 5 questions for every 1 it loses. At 6,000 tokens the split is 4 to 5 — a coin flip — and 61 of 78 questions both arms already get right.

I want to be careful here, because I initially reported this as *negative* — memory actively hurting — on a 25-question sample where it measured −2. At 78 questions it is −1 with a 4-to-5 split, which is a tie. Memory stops paying. It does not cost you. That correction is in the repo, along with the draft that got it wrong.

---

## Why: your retriever is competing with attention

The mechanism is worth spelling out, because it generalises past this library.

When you put text in a context window, the model doesn't *search* it. It **attends** to all of it, simultaneously, at every layer. Each token emits a query vector, every prior token has a key, the dot products go through a softmax, and the result is a weighted blend of everything present. Dozens of heads do this in parallel, across depth, each refining what the last layer surfaced.

Now compare that with what a retriever does. `search()` makes **one** decision: embed the query, rank by cosine, take the top 8, commit — all before the model has reasoned about anything.

Retrieval is a function of the context. By the data processing inequality it cannot *increase* mutual information with the answer. **Filtering can only lose.** At best it loses nothing.

So when the material fits in the window, an external retriever is strictly worse than doing nothing, and the only question is by how much. That's the convergence in the first chart — not memory failing, but attention doing a job it is simply better at.

Retrieval has exactly one advantage, and it is decisive when it applies: **attention cannot attend to tokens that were never admitted.** Truncation is the one thing it cannot route around. That is the entire job of a memory layer — not to retrieve better than attention, but to decide *what gets into the window* so attention has the right material.

That reframing has teeth. It says a well-built memory layer should be **invisible** on any benchmark where the content fits, and decisive where it doesn't. Which is a testable claim, and it is why so many memory benchmarks show nothing: they are run in the regime where memory should show nothing.

---

## What this means if you're building one

**Gate injection on context pressure.** Inject nothing while the conversation fits; inject as it stops fitting. Unconditional injection spends ~350 tokens a call for no measured return in the regime most agent turns occupy. This is the design my own measurements imply and my own library does not yet do.

**Match budgets when you benchmark.** If your memory layer and your baseline are taking different amounts of context, you are measuring budgets. I published a table that made this mistake before I caught it.

**Preserve local structure.** Attention traverses adjacency — the answer sits near the words that predict it. I split conversations into individual claims, which makes invalidation surgical and destroys exactly that adjacency. `"25:50"` on its own has almost no surface for a query to match; `"I got my 5K down to 25:50"` has plenty. Splitting is a real trade, not a free win.

**Don't ask the model to maintain the graph.** I gave it the tool, the facts it already held with their ids, and an explicit instruction to link contradictions. It proposed a link on **9 of 78** questions that were *all* about a fact changing. Deriving the same links deterministically found **43**. The model wasn't refusing — retrieval hadn't shown it the fact it needed to supersede.

---

## What I got wrong, since that's the useful part

- I led the README with "−9.8% input tokens and 6.33 of 8 metrics against 3.67." It didn't survive being measured at N=3. Retired.
- Every token figure I recorded for weeks was measured while Ollama silently truncated prompts at its default 2,048 tokens. A prompt of 5k, 22k or 135k tokens all reported 2,051.
- I shipped a supersedes-DAG demotion rule that produced **byte-identical** answers on 78 questions. A no-op, discovered only because I finally ran the cheap offline gate first.
- Five structural changes in one day each moved an offline metric by 0 or 1 of 25 — because they were all optimising retrieval in the regime where the curve says retrieval is worth nothing.

---

## Part 2

Everything here runs on a ~6,900-token haystack, so scarcity had to be created by *capping the budget* rather than by lengthening the conversation. That simulates a small-context model; it does not simulate a long session.

Part 2 runs LongMemEval_S — ~115k tokens across 40 sessions, where full context is not an option at all and most of the haystack is distractors. The prediction, stated before the run: the curve holds and the advantage grows, because the fraction of the conversation that fits keeps falling. What would falsify it: the advantage shrinking with session length, which would mean this was an artefact of clipping a short transcript.

There is also a more interesting possibility. On a haystack that is mostly noise, filtering could beat full context *outright* — by removing distractors that steal attention mass — rather than merely surviving truncation. Nothing I've measured supports that yet. Nothing has tested it either.
