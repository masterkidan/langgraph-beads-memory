# Post-fix results — 2026-08-10, qwen3:8b, N=3

Re-run after two changes aimed at the `supersedes` defect found on 2026-08-09:
per-claim fact splitting, and a similarity guard on `supersedes`.
6/6 runs, 0 errored turns, 36.7 min. Raw transcripts in `n3-postfix/`.

**Headline: one fix worked, the other did not — and cost something.**

## Objective metrics

| metric | baseline (n=3) | treatment pre-fix | treatment post-fix |
|---|---|---|---|
| uses_revised_budget | 0/3 | 3/3 | **3/3** (held) |
| mentions_feasible_option | 1/3 | 3/3 | 3/3 |
| mentions_selfhost | 2/3 | 2/3 | 2/3 |
| mentions_primary_sources | 0/3 | 1/3 | **1/3 — no change** |
| buried_detail_recalled | 3/3 | 2/3 | **1/3 — worse** |
| avoids_stale_budget | 3/3 | 3/3 | 3/3 |
| mean input tokens | 18,424 | 10,083 | 9,175 |

## The supersede guard: worked

The defect was real and is gone.

| | pre-fix | post-fix |
|---|---|---|
| supersede edges targeting a `user_input` fact | present in all 3 runs | **0 in all 3 runs** |
| primary-sources constraint status | `superseded` in all 3 | **`active` in all 3** |

Critically, the guard did **not** block the legitimate budget revision:
`uses_revised_budget` held at 3/3. That was the risk — the legitimate edge has
the identical agent-conclusion-over-user-input shape as the spurious ones, so a
rule based on fact *kind* would have broken the headline result. Similarity
separates them where kind cannot.

## The granularity change: did not work, and cost something

Splitting was supposed to fix `mentions_primary_sources` by giving each
constraint its own embedding. It did not: 1/3 before, 1/3 after. And
`buried_detail_recalled` fell from 2/3 to 1/3.

Inspecting retrieval shows why. Splitting also fragments *questions and
instructions* into retrievable facts, and those rank near the top precisely
because they are lexically closest to the query — they are nearly the query
itself:

```
run0, top-8 for "which vector database should we pick and why?"
  1. [user_input] which vector database should we pick, why?          <- question fragment
  2. [user_input] We need to pick a vector database for our product.  <- goal fragment
  3. [user_input] Please investigate our vector database options...   <- instruction fragment
  ...
  8. [user_input] Be specific about how it fits our constraints.      <- question fragment
  -> the primary-sources constraint does not appear
```

In the live run the dedup exclusion removes the current turn's own fragments,
but the conversation-1 and -2 procedural fragments are not in the window and
consume slots that substantive facts previously held. The change traded
survival for ranking pressure.

The mechanism reasoning behind the fix was sound — the bundled embedding really
was diluted, and the blast radius really was three constraints. The predicted
improvement simply did not materialise, and the honest summary is that
splitting addressed a real problem without improving the outcome metric.

## What this suggests next (unmeasured)

- **Do not capture interrogative or imperative fragments as facts.** A question
  is not a claim. This looks like the highest-value follow-up, since it targets
  exactly the slots being wasted.
- **Rank `user_input` constraints above procedural text**, rather than relying
  on raw cosine similarity where the query resembles the question.
- Revisit whether splitting should be retained at all if the ranking fix
  subsumes its benefit. It is currently justified by the supersede blast-radius
  argument, not by any measured metric gain.

## Caveats

- N=3; `buried_detail` 2/3 -> 1/3 is a one-run difference and within noise.
  It is reported because it moved against us, not because it is conclusive.
- Baseline is unchanged and acts as a control: identical aggregate to the
  pre-fix run, confirming the changes are isolated to the treatment.
- `uses_revised_budget` is now **0/13** across every baseline run in this
  project — that separation is no longer plausibly variance.
