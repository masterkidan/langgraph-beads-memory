# A Memory Layer for LangGraph That Beats the Default Store

Part 1 in a series. At a matched context budget, a typed fact graph scores 51/78 against LangGraph's `PostgresStore` at 42/78 on LongMemEval — 22 questions won, 13 lost. That advantage decays to zero by a 6,000-token budget, and the reason it decays is the more useful result.

LangGraph ships `PostgresStore` with pgvector, and LangMem on top of it. Save a memory, cosine-search it back. It works, and for short sessions it is enough. We built something else — a typed fact/conclusion graph on Postgres, per-claim capture, `supersedes` edges, forked sub-agent namespaces — and then spent most of the effort finding out whether it was actually better, which turned out to be a harder question than building it.

Code and every run: [github.com/masterkidan/langgraph-beads-memory](https://github.com/masterkidan/langgraph-beads-memory).

## The benchmark mistake

Our first comparison. LongMemEval `knowledge-update`, 78 questions, gemma4:12b, external and MIT-licensed:

| arm | correct | mean input tokens |
|---|---|---|
| whole transcript in the prompt | 64/78 (82%) | 6,337 |
| LangGraph `PostgresStore` over turns | 57/78 (73%) | 2,473 |
| fact graph | 48/78 (62%) | 328 |

Third of four. Then the third column: the store took **7.5× more context**. That table does not compare memory strategies, it compares budgets. An arm handed 2,473 tokens beats an arm handed 328 whatever you fill them with.

This is the default shape of most memory benchmarks. Each system retrieves "its natural amount," whoever retrieves more wins on accuracy, and the cost difference sits in a separate table or no table at all. The comparison reads as fair because both sides are doing retrieval.

So we rebuilt the harness to hold total context constant and vary only how it is spent:

- **transcript only** — fill the budget with the most recent conversation
- **facts + transcript** — spend ~950 characters on retrieved facts, fill the remainder with transcript

That second arm is the production shape. Nobody deploys a memory layer *instead of* the conversation; `BeadsMemoryMiddleware` sends the windowed raw messages plus an injected block. Then sweep the budget.

## Results

![Accuracy per token of context](../assets/accuracy-per-token.svg)

*Fig 1 — Every arm as a point on (context spent, accuracy). n=78 per point. At ~1,200 tokens the three strategies are 36 points apart. By ~5,400 they sit within 1.3 points of each other and of the ceiling.*

At a 1,200-token budget, against the alternatives:

| at 1,200 tokens, n=78 | correct | % of ceiling | % of context |
|---|---|---|---|
| transcript only | 23/78 | 36% | 19% |
| LangGraph `PostgresStore` | 42/78 | 66% | 15% |
| **fact graph** | **51/78** | **80%** | 19% |

Four-fifths of what full attention achieves, on a fifth of the context. Clipping the window to a fifth costs 64% of the ceiling with raw transcript and 20% with the fact graph on top.

Totals hide whether two arms are solving the *same* questions, so:

![Question by question outcomes](../assets/pairwise-outcomes.svg)

*Fig 2 — The same 78 questions as win / both / neither / lose. Against the store at a matched budget: 22 won, 13 lost, 29 already shared.*

## The advantage decays to zero

![The context-budget curve](../assets/context-budget.svg)

*Fig 3 — Identical token budget in both arms; only the split changes. n=78 at the ends, n=25 at 3,000.*

| budget | transcript only | facts + transcript | Δ |
|---|---|---|---|
| 1,200 tokens | 23/78 | **51/78** | **+28** |
| 3,000 tokens | 19/25 | 20/25 | +1 |
| 6,000 tokens | 66/78 | 65/78 | −1 |

Five questions won for every one lost at a tight budget. At 6,000 tokens the split is 4 to 5 — a coin flip — with 61 of 78 already shared.

One correction worth recording: at n=25 the 6,000 point measured −2 and we wrote it up as interference, facts displacing transcript that held the answer. At n=78 it is −1 with a 4-to-5 split. That is a tie. Memory stops paying; it does not cost you.

## Why the decay happens

A model does not search its context. It attends to all of it, at every layer: each token emits a query vector, every prior token has a key, the dot products go through a softmax, and the output is a weighted blend of everything present. Dozens of heads do this in parallel and compose across depth.

`search()` makes one decision — embed the query, rank by cosine, take the top 8, commit — before the model has reasoned about anything.

Retrieval is a function of the context, so by the data processing inequality it cannot increase mutual information with the answer. Filtering can only lose. When the material fits in the window, an external retriever is strictly worse than passing everything through, and the only question is the margin. That is the convergence in Fig 1.

Retrieval has exactly one advantage, and it is decisive where it applies: attention cannot attend to tokens that were never admitted. Truncation is the one thing it cannot route around. The job of a memory layer is not to retrieve better than attention — it is to decide what enters the window so attention has the right material.

That predicts something testable: a correctly built memory layer should be **invisible** on any benchmark where the content fits, and decisive where it does not. It also explains why so many memory benchmarks show nothing. They are run in the regime where memory should show nothing.

## What made the difference

**Deriving `supersedes` links instead of asking for them.** The agent had `remember_fact`, the held facts with their short ids in the prompt, and an explicit instruction to link contradictions. It proposed a link on **9 of 78** questions that were *all* about a value changing. Deriving the same links from a value comparison found **43**. The model was not refusing — retrieval had never shown it the fact it needed to supersede, which sat at rank 21 of ~180 while the top 8 were captured assistant prose.

**Resolving rather than retiring.** A superseded fact used to be excluded from retrieval. Excluding the stale value leaves the slot empty, which does not help an agent about to answer with it. Now a hit on any version of a claim resolves to the current one — the measured failure was answering "you currently have three bikes" when the user owns four.

**A `derived_from` filter.** Every captured conclusion links to the facts that were in context when it was written, so a restatement can be dropped when the fact it came from is already in the block. Recorded provenance, not redundancy inferred from a vector.

**Capturing tool results into the calling agent's namespace.** A sub-agent reads a TLS handshake p99 of 41ms and drops it when summarising. That figure passes 9 of 56 runs on our incident scenario and no ranking change can reach it, because it was never written. Capture scoped to the caller keeps the sub-agent's raw reads out of the root's ranking — 44 child facts to 27 root in a live run.

## Lessons learnt

- Match context budgets across arms, or you are benchmarking budgets
- Report cost and accuracy in the same row, or a cheaper-and-slightly-worse arm reads as a loss
- Report pairwise win/lose, not just totals — two arms can tie while solving different questions
- Build the offline gate first: ours scores "does the gold answer reach the injected block," runs in minutes with no generation, and killed four designs that would each have cost 40+ minutes of GPU
- Set `num_ctx` explicitly. Ollama defaults to 2048 and truncates silently — prompts of 5k, 22k and 135k tokens all reported `prompt_eval_count=2051`, which invalidated weeks of our token accounting
- Do not ask the model to maintain the graph. Derive what you can from values and provenance
- A structural change that looks obviously right is worth measuring anyway: a supersedes-demotion rule we shipped produced byte-identical answers on all 78 questions

## What's next

Everything above runs on a ~6,900-token haystack, so scarcity was created by capping the budget rather than by lengthening the conversation. That simulates a small-context model, not a long session.

Part 2 runs LongMemEval_S — ~115k tokens across 40 sessions, where full context is not an option and most of the haystack is distractors. The prediction, stated before the run: the curve holds and the advantage grows, because the fraction of the conversation that fits keeps falling. What would falsify it: the advantage shrinking with session length, which would mean this was an artefact of clipping a short transcript.

There is a stronger possibility worth separating. On a haystack that is mostly noise, filtering could beat full context outright by removing distractors that steal attention mass, rather than merely surviving truncation. Nothing we have measured supports that. Nothing has tested it either.

Part 3 is pressure-gated injection: inject nothing while the conversation fits, inject as it stops fitting. That is what Fig 3 implies and what the library does not yet do.
