# Open questions

Live after the 2026-08-19 context-budget round. Items 1-3 from the previous
edition are **resolved or superseded**; what remains is below, reordered by what
the measurements now say matters.

Follow [development-loop.md](development-loop.md). Item 1 is a new capture
policy and therefore tier 0; items 2-4 are bug-fixing and measurement.

---

## 1. Pressure-gated injection — the design the curve implies (TIER 0)

At a matched token budget, injected facts are worth **+6 of 25** when the
transcript is badly clipped, **+1** when it mostly fits, and **−2** when it
nearly all fits. Injecting unconditionally is a measurable tax in the regime
most agent turns occupy.

The policy that follows: inject nothing while the history fits the window;
inject as it stops fitting. Symmetrically, capture at the points where context
is destroyed — window eviction, sub-agent fork, sub-agent merge, thread end —
rather than eagerly on every message.

A first pass at the capture half exists on the `context-capture` branch and is
**unvalidated and probably inert in this repo**: there is no checkpointer, so
each turn already starts from an empty message list and `window=10` binds on
**1 of 72** turns. It only pays where context persists across turns, or inside
one long invocation.

## 2. `k=8` is a count standing in for a budget

`window=10` binds almost never (1 of 72 turns, median turn is 4 messages).
`k=8` binds on **every call**, over facts ranging from ~40 to ~500 characters —
measured injected blocks vary 596 to 1,065 chars at identical `k`.

Both incident regressions at N=3 (`names_surviving_cause`,
`proposes_reversible_fix`, on two independent models) come from one turn where
the covering fact sits at **rank 11** with `k=8`. This is the one lever
identified on day one of that investigation and still never tested end to end.

## 3. `demo/llm.py` still does not set `num_ctx`

Ollama's default is 2048; prompts of 5k, 22k and 135k tokens all reported
`prompt_eval_count=2051`. Every scenario result in `results/` was measured under
that ceiling.

Not fixed, deliberately: setting it would make future runs incomparable with the
24-run tier-3 set. The decision is whether to re-baseline the whole matrix or
keep the ceiling for continuity. `demo/longmemeval.py` sets it explicitly.

## 4. Does the tiebreak remove the run-to-run spread?

At temperature 0 all four baseline cells scored identically three times; every
treatment cell had spread 1-2. A content-derived `md5(body)` tiebreak now
replaces physical order, but the cell measuring it is **quarantined**: 2 of 3
runs took 1190s and 1226s against a 473s mean while `/api/ps` reported no model
loaded, and the scores track runtime exactly. Needs a clean re-run.

## 5. The scenarios are the only setting that exercises the whole design

LongMemEval is one-shot QA over a finished conversation, so there is no "recent"
context — it tests the older-items half of a two-part design with the transcript
half amputated. `incident` and `vecdb` are multi-turn and exercise both.

Conversely the oracle set fits entirely in a 16k window (~6,900 tokens per
question), which is the regime where the budget curve says memory should be
worth −2. Any future external benchmarking should use LongMemEval_S (~115k
tokens, 40 sessions), where full context is impossible and ~95% of the haystack
is distractors — the one setting where filtering could beat raw context outright
rather than merely surviving truncation.

---

## Resolved since the last edition

**Deterministic tiebreak** — shipped (`md5(body)`); effect unvalidated, see 4.

**Ranking defaults** — `CONTEXT_MAX_PER_SOURCE` now defaults off: measured 50% →
61% incident aspect coverage, bit-identical on vecdb.

**Breadth / the agent floor** — superseded. Five structural interventions
(demotion, DAG resolution, condensation, `is_substantive`, removing splitting)
each moved the offline gate by 0 or 1 of 25, because every benchmark run sat in
the regime where the curve says memory is worth −2. The floor is not the
constraint; context pressure is.

**The README's headline claims** — rewritten. It now leads with the budget curve
and states plainly that no accuracy gain is established at N=3.

**`qwen3.6` is not installed** — resolved as *not viable*: it ships only as a
24 GB build with no 8b/9b/12b variants, which does not run on a 16 GB machine.
`qwen3.5:9b` is the second model; the development loop's model set needs
correcting.

---

## Decided, not open

**Tool-result capture ships off by default** (`treatment-toolcapture` keeps it
measurable) — but its recorded cost **did not reproduce on 2026-08-16** and the
entry needs re-deciding.

It buys `buried_metric_recalled` 0/3 → 3/3, a capability the flat store cannot
match. The recorded price was `breadth_complete` 1/3 → 0/3 and input tokens
going from 35% below baseline to 13% below. Re-measured at N=1 across four
cells with the derivation filter in place, the stores still roughly doubled
(94 → 182 facts) and tokens moved **−4.8%, −4.9%, +5.3%, −17.6%** — flat or
better, not worse. The breadth cost appeared on qwen (3 → 1) and not on gemma
(3/3, complete).

That is consistent with the price having been a crowding artefact of the old
ranker rather than a property of capture: the derivation filter drops
restatements before they take slots, so extra candidates no longer inflate the
block. N=1 per cell, so this is a reason to re-run rather than a reason to flip
the default.

**Interface parity ships off by default** (`treatment-searchtool`). Worth +0.33
metrics on incident and 0.00 on vecdb — inside noise on the means. It moves
individual metrics hard in both directions though: on vecdb it took
`uses_revised_budget` 3/3 → 1/3 while taking `mentions_primary_sources` 1/3 →
3/3. Auto-injection guarantees the corrected constraint is present whether or
not the agent thinks to ask; a tool call retrieves what the agent asks for. Keep
both, default to injection.
