# 2026-08-19 · What memory is worth, as a function of context

This round changes the claim this project makes. The short version:

> **Memory's value is inversely proportional to how much of the conversation
> still fits in the window.** At a 1,200-token budget it is worth **+28 of 78**
> questions — 65% against 29%, winning 35 and losing 7. At 6,000 tokens, where
> the transcript nearly all fits, it is worth **−1 of 78**: a tie.

Everything below is on `gemma4:12b` unless stated. Every number is per-run or
per-cell; nothing is pooled across scenarios or models.

---

## 0. A measurement bug that affects every prior round

`demo/llm.py` never set `num_ctx`. **Ollama's default is 2048.** Proven
directly, by sending the same prompt at three sizes:

| prompt sent | `prompt_eval_count` |
|---|---|
| ~5,600 tokens | 2051 |
| ~22,500 tokens | 2051 |
| ~135,000 tokens | 2051 |
| ~7,900 tokens, `num_ctx=8192` | **7025** |

So for any model call whose prompt exceeded ~2048 tokens, the model **never saw
the excess**, and the recorded `input_tokens` was pinned at the ceiling rather
than measuring the real prompt.

This touches every historical result in `results/`, in both axes. The direction
is not uniform: the baseline accumulates `search_memory` results as tool
messages so its prompts grow within a turn and cross the ceiling first, which
means its true cost is understated and it was reading truncated context. The
treatment rewrites a fixed-size block into the system prompt each call and
crosses less often.

Per-call token counts are not stored in the run files, only totals, so the
affected fraction is not recoverable retroactively. **Prior token figures should
be read as lower bounds on the baseline's real cost, and prior accuracy figures
as measured under an unannounced handicap that fell mainly on the baseline.**

`demo/longmemeval.py` sets `num_ctx` explicitly. `demo/llm.py` still does not —
changing it would make future scenario runs incomparable with the 24-run tier-3
set below, so it is left as an open decision rather than silently flipped.

---

## 1. Tier 3 — N=3, two models, two scenarios, 24 runs, 0 errors

Every run, not a mean:

| scenario | model | baseline | treatment | Δ metrics | Δ input tokens |
|---|---|---|---|---|---|
| incident | gemma4:12b | 7, 7, 7 | **6, 4, 5** | −2.00 | **−39.6%** |
| incident | qwen3.5:9b | 5, 5, 5 | 6, 5, 6 | +0.67 | −2.9% |
| vecdb | gemma4:12b | 5, 5, 5 | 5, 5, 6 | +0.33 | **−44.2%** |
| vecdb | qwen3.5:9b | 5, 5, 5 | 4, 5, 5 | −0.33 | −12.9% |

**The token saving is the only effect consistent across all four cells.** Three
of the four accuracy deltas are under one metric and are not results by this
repo's own standard. The exception is gemma incident, where the distributions do
not overlap and the treatment is **worse**.

Per-metric, the movements that are consistent across runs — and two of them are
consistent across *both models*, which is stronger evidence than either cell
alone:

| metric | gemma incident | qwen incident |
|---|---|---|
| `names_surviving_cause` | 3/3 → **1/3** | 3/3 → **1/3** |
| `proposes_reversible_fix` | 3/3 → **0/3** | 3/3 → **1/3** |
| `uses_corrected_deploy_time` | 3/3 → 3/3 | **0/3 → 3/3** |
| `buried_metric_recalled` | 0/3 → 0/3 | **0/3 → 2/3** |
| `breadth_complete` | 3/3 → 3/3 | **0/3 → 2/3** |

Both regressions belong to the same turn (conv-3), on two independent models, at
the same magnitude. The offline probe puts the fact covering `reversible_fix` at
**rank 11** while `k=8` — two slots out of reach, every run.

The wins are the architecture's own claims: typed invalidation carrying a
correction the flat store never holds, and breadth and buried detail on the
model whose baseline reaches neither.

### A run-to-run variance finding

At temperature 0 the model is deterministic. **All four baseline cells scored an
identical value three times (spread 0); every treatment cell had spread 1–2.**
On one incident turn, only 4 of the 8 injected facts were common to all three
runs, out of a union of 16 — half the block decided by thread scheduling, with a
measured rank-8/rank-9 margin of 0.00126.

So the treatment injects the variance. A content-derived tiebreak (`md5(body)`)
now replaces physical order. Whether it removes the spread is **not established**
— see the quarantined cell in `results/matrix-tiebreak-confounded/`.

---

## 2. LongMemEval — an external benchmark

`knowledge-update`, n=78, `num_ctx=16384`. The first number in this project not
measured on a scenario it designed.

| arm | correct | mean input tokens |
|---|---|---|
| fullcontext | **64/78 (82%)** | 6,337 |
| baseline — `PostgresStore` over whole turns | 57/78 (73%) | 2,473 |
| bm25 — LongMemEval's own reference retriever | 46/78 (59%) | 2,347 |
| memory — replay ingest | 46/78 (59%) | 343 |
| memory — agent ingest | 44/78 (56%) | 337 |
| memory — deterministic autolink | 48/78 (62%) | **328** |

**Read this as the fact graph against a strong idealised document store, not
against LangGraph.** The `baseline` here keeps whole turns verbatim; the real
LangMem path has the agent decide what to save and stores extracted memories, so
it is not the same thing. The 9-point gap is the cost of splitting, measured
cleanly — not a competitive result.

### Typed invalidation was reachable and did not help

`--ingest agent` makes `remember_fact(supersedes=...)` available with the held
facts and their short ids in the prompt:

```
78 supersede-shaped questions -> 9 model proposals -> 18 edges -> 8 questions
accuracy 44/78, against 46/78 with the mechanism switched off
```

Deriving the link instead of asking for it (`--ingest autolink`, using
`contested_values`) found **43** links where the model proposed 9, and moved the
score to 48/78. On the 38 questions where a link fired it scored 25/38 against
replay's 22/38.

The 12% proposal rate turned out **not** to be a compliance problem. Under a
prompt that taught graph shape, the model wrote exactly the right node — *"The
user's 5K personal best is 25:50."* — and linked nothing, because the fact it
had to supersede sat at **rank 21 of ~180** while the 8 facts it was shown were
captured assistant prose. It was never shown the target.

---

## 3. The central result: value against budget

Every comparison above hands the arms *different* amounts of context, which is
the flaw in all of them. This one does not.

`tail` spends its whole budget on recent transcript. `augment` spends part of it
on retrieved facts and fills the rest with transcript. Same reader, same
questions, same total budget, `num_ctx=16384` so nothing is truncated by the
window — the budget is the only constraint.

| budget | tail (100% transcript) | augment (facts + transcript) | Δ | n |
|---|---|---|---|---|
| 1,200 tok | 23/78 (29%) | **51/78 (65%)** | **+28** | 78 |
| 3,000 tok | 19/25 (76%) | 20/25 (80%) | +1 | 25 |
| 6,000 tok | 66/78 (84%) | 65/78 (83%) | −1 | 78 |

Pairwise, at 1,200: augment wins **35** questions tail misses, against **7** the
other way. At 6,000: 4 against 5 — a coin flip.

**Memory's return declines to zero as context becomes sufficient.** It is worth
a third of the benchmark when the transcript is badly clipped and nothing when
it nearly all fits, for a constant ~350 tokens per call.

### CORRECTION, disclosed

The first version of this table was n=25 and showed **−2** at 6,000, described
here as measured interference "appearing exactly where the theory says it
should". At n=78 that is **−1 with a 4-to-5 pairwise split**, which is a tie.
The interference claim does not survive; the correct statement is that memory
*stops paying*, not that it costs you.

The ends were re-run at n=78 with `--ingest autolink` rather than `replay`, so
the +28 confounds two changes: the sample (25 → 78) and the ingest mode. The
`tail` arm is the control — it builds no store, so autolink cannot touch it —
and it moved 32% → 29%, i.e. flat. That suggests the metric is stable across the
sample change and most of the gain is autolink, but a `replay`-at-78 run would
be needed to separate them. The 3,000 midpoint is still n=25 on `replay`.

### Why this reframes the library

An external retriever competes with attention. Attention is soft, per-head,
per-layer, refined across depth and conditioned on the whole question;
`search()` makes one hard top-k commitment from a single query vector before the
model reasons at all. Retrieval is a function of the context, so it cannot
increase mutual information with the answer — **filtering can only lose.**

That is the mechanism behind the curve, and behind the day's other results: five
structural interventions (demotion, DAG resolution, condensation, the
`is_substantive` fix, removing splitting) each moved the offline gate by 0 or 1
of 25, because every benchmark we ran sat in the regime where the curve says
memory should be worth **−2**.

**The design that follows: gate injection on context pressure.** Inject nothing
while the history fits; inject as it stops fitting. Unconditional injection is a
measurable tax in the regime where most agent turns live.

A prompt cannot fix this. An instruction operates on what is in the context; the
loss happened before the prompt was built.

---

## 4. What did not work, recorded so it is not retried

| change | result |
|---|---|
| demote superseded facts by DAG depth instead of excluding | **byte-identical** answers on 78 questions. Any penalty large enough to matter is a filter. |
| back-edge derivation filter (skip a premise when a derivative is selected) | −8 on vecdb. Evicted a user constraint because the conclusion built on it ranked higher. |
| tighten `is_substantive` to reject bare ordinals | reach 18/25 → 18/25. The markers were a symptom, not a cause. |
| condense a corrected value in place | +1/25, identical per-question to plain linking. The mixed-premise case it targets barely occurs in this data. |
| remove regex splitting entirely | +4/25 **at 14× the context**. At matched budget splitting wins: 18/25 for 833 chars vs 13/25 for 968. |

Three N=1 readings also collapsed at N=3 this round: vecdb gemma 6/6 → 5,5,6;
the qwen token penalty (+11.6% → −8.9%, opposite sign); and gemma incident
breadth 1 → 3 on identical code.

---

## 5. Method notes

**The tier-1 gate that made this affordable.** Scoring "does the gold answer
reach the injected block" over frozen stores, with no generation, runs in
minutes and killed four designs that would each have cost 40+ minutes of GPU.
Its known defect: it scored a question wrong whose correct answer sat at **rank
1**, so absolute counts are soft and only within-harness comparisons are sound.

**Ollama wedges, and the recovery must be built in.** One harness ran without a
timeout and hung for 1h49m mid-question, blocked in `sock_recv` on a request
that `/api/version` and `/api/ps` both reported healthy. `ResilientChatOllama`
already solves this; bypassing it to set `num_ctx` reintroduced the bug. Results
are now flushed per question with `--resume`, because a wedge previously
destroyed 21 questions of completed work.

**Python buffers stdout when it is not a tty.** A run that appeared stalled for
minutes was writing into an 8 KB buffer. `python -u`, and `tee` rather than
`tail`.
