# Results — 2026-08-10, qwen3:8b, N=3 (directive fix)

6/6 runs, 0 errored turns, 41.2 min. Transcripts in `n3-directive/`.
Third and final iteration on `mentions_primary_sources`, after per-claim
splitting and the `supersedes` similarity guard each failed to move it.

## Objective metrics

| metric | baseline (n=3) | treatment (n=3) | vs pre-directive |
|---|---|---|---|
| uses_revised_budget | 0/3 | **3/3** | held |
| mentions_primary_sources | 0/3 | **3/3** | **1/3 → 3/3** |
| mentions_selfhost | 2/3 | **3/3** | 2/3 → 3/3 |
| mentions_feasible_option | 1/3 | **3/3** | held |
| avoids_stale_budget | 3/3 | 3/3 | held |
| **buried_detail_recalled** | **3/3** | **0/3** | **1/3 → 0/3** |
| mean input tokens | 20,742 | 9,963 | |
| mean output tokens | 1,554 | 1,104 | |

**Blinded judge** (labels stripped, order randomised): treatment 5.00/5.00/5.00
vs baseline 2.67/2.67/2.67 on recall/delegation/final. 3 judged, 0 unjudgeable.

## The directive fix worked

Holding questions, instructions and stated goals out of *default retrieval* —
while keeping them stored and queryable — recovered the metric that two earlier
fixes could not. `mentions_primary_sources` went 1/3 → 3/3 and `mentions_selfhost`
2/3 → 3/3, with no regression in the budget result.

The mechanism is the one predicted: directives rank highly against a query
precisely *because they resemble it*, and were consuming injection slots that
constraints needed. Excluding four such fragments freed those slots.

## And it exposed a real architectural cost

`buried_detail_recalled` fell to **0/3** while the baseline holds 3/3. The trend
across three iterations — 2/3, then 1/3, then 0/3 — tracks retrieval getting
progressively more selective. That is not noise.

**Inspecting the database gives the actual cause, and it is not retrieval.** In
treatment run 0 the Qdrant quantization fact is *absent from every namespace*.
The researcher recorded 2–3 facts, all about deployment, and its `conclude_task`
rollup that crossed to the parent read:

> "Qdrant is a self-hostable vector database with deployment options including a
> single bin..."

The detail was never captured, so conv-3 hallucinated "approximately 30%".

The baseline stores **4 long unsplit blobs** — whole document sections — in one
flat shared namespace. The quantization detail rides along incidentally inside a
larger Qdrant memory, and every agent can see it.

**So the two designs trade precision against recall:**

| | this library | stock LangGraph memory |
|---|---|---|
| granularity | one fact per claim | whole saved blob |
| sub-agent boundary | one summary crosses; raw exploration stays in the child | flat shared store; everything is visible |
| stated constraints | survive reliably (3/3) | lost (0/3) |
| incidental details | dropped unless the summary carries them (0/3) | survive by accident (3/3) |

Neither is strictly better. Fine-grained facts plus a summarising boundary
preserve *what was decided*; coarse shared blobs preserve *what was seen*. This
demo's scenario rewards the first and punishes the second, which is worth stating
plainly given the scenario was designed to exercise supersede-and-recall.

## Honest limits

- **N=3, one scenario, one model.** The `buried_detail` 0/3 is consistent and
  mechanistically explained, but three runs is three runs.
- **The judge remains coarse** — a flat 5/5/5 for the treatment in every pair.
  Direction is credible; precision is not.
- **Not comparable to earlier result sets.** A 4b run was attempted and reverted
  (it wrote ~12,000 output tokens per run against 8b's ~1,400); those partials
  are in `aborted-4b/` and are not pooled here.
- **Sub-agent capture is the weak link, not retrieval.** The obvious follow-up is
  whether a researcher should be required to record specific findings rather than
  a single narrative summary — untested.
