# How different models benefit from a memory harness

**Status: three models, both scenarios. Six cells at N=1, one of them
re-measured at N=3.**

A separate question from "does structured memory help", and it deserves its own
treatment. The library's three properties — constant retrieval cost, small
payload, typed relevance — are architectural and hold regardless of model. What
*varies by model* is which of them you actually cash in.

## The question

Within a single model, what does the harness change? That is a paired
comparison, so the model's own capability cancels and what remains is the
harness's contribution. Pooling across models would attribute a model's strength
to the harness, which is why `demo/aggregate.py` refuses to do it and
`demo/compare_models.py` reports within-model deltas instead.

Efficiency has two axes and a claim needs both:

- **accuracy** — objective metrics passed, as a fraction of those scored
- **cost** — input and output tokens

## Models

The starting set was every model on Ollama that advertises tool calling and fits
16GB — five models across five vendors. `qwen3.6`, `kimi-*` and every
tool-capable GLM are too large; `glm4:9b` fits but has no tool calling, which
would gut both arms. Three of the five remain, for the reasons below.

| model | size | vendor |
|---|---|---|
| `gemma4:12b` | 7.6GB | Google |
| `qwen3.5:9b` | 6.6GB | Alibaba |
| `lfm2.5:8b` | 5.2GB | Liquid |

All three pass `demo/smoke_test.py`, which gates on structured tool calls *and*
fact extraction.

`granite4.1:8b` (IBM) and `ministral-3:14b` (Mistral) were benchmarked and then
dropped. Ministral cannot execute the scenario — it delegated two of three
researchers in all four cells and its baseline arm called `search_memory` zero
times, so that arm has no retrieval to compare against. Granite is competent,
but its vecdb pair is confounded and archived, which would leave it represented
by whichever scenario happened to survive.

Both were dropped after their results were known, so what the exclusion changes
is stated rather than left implicit: it gives up two of four accuracy wins and
takes the token result from 5-cheaper/4-costlier to 5-cheaper/1-costlier. Runs,
reasons and figures in [excluded/README.md](excluded/README.md).

## The one cell measured at N=3

`qwen3.5:9b` · incident was the only cell where the fact graph cost *more* input
than the document store (+3.6%), which made it the single figure standing
against the cost claim. Re-run at N=3 per arm on 2026-08-12, it was a one-off:

| | run 1 | run 2 | run 3 | mean |
|---|---|---|---|---|
| baseline accuracy | 3/8 | 3/8 | 5/8 | 3.67 |
| treatment accuracy | 6/8 | 6/8 | 7/8 | **6.33** |
| baseline input | 14,551 | 14,551 | 17,079 | 15,394 |
| treatment input | 12,999 | 14,304 | 14,332 | **13,878** |

**−9.8% input, not +3.6%**, and neither distribution overlaps: the most
expensive treatment run (14,332) is cheaper than the cheapest baseline run
(14,551), and the worst treatment accuracy (6) beats the best baseline (5).

Two caveats that matter more than the headline:

- The treatment arm ran on the **2026-08-12 code** (derived_from edges,
  value-aware cascade, split summaries, per-agent cap); the baseline arm is
  unchanged. So this re-measures the cell rather than reproducing the N=1 row
  below.
- It also **shrinks the accuracy win**. The N=1 row claimed +62 pts (8/8 vs
  3/8); at N=3 it is +33 pts (6.33 vs 3.67). The 8/8 was the top of a range,
  not the centre of one — which is the whole reason N=1 rows are not worth
  arguing over.

Runs in [n3-qwen-incident/](n3-qwen-incident/).

## Results — three models, both scenarios, N=1 per cell

Everything below is N=1 and predates the 2026-08-12 invalidation work. It is
kept as the record of what was measured, not as a claim about the current code.

| model | scenario | accuracy | input tokens | verdict |
|---|---|---|---|---|
| `gemma4:12b` | incident | +0 pts (7/8 both) | **−29%** | win — same, cheaper |
| `qwen3.5:9b` | incident | **+62 pts** (8/8 vs 3/8) | +4% | superseded by the N=3 cell above |
| `lfm2.5:8b` † | incident | +38 pts | **−42%** | win — but see caveat |
| `gemma4:12b` | vecdb | −17 pts | **−29%** | trade — cheaper, one metric lost |
| `qwen3.5:9b` | vecdb | +0 pts (5/6 both) | **−32%** | win — same, cheaper |
| `lfm2.5:8b` † | vecdb | −50 pts | **−43%** | trade — but see caveat |

† **Both `lfm2.5:8b` cells are compromised, in opposite directions.** It hit the
recursion limit once per scenario, and the errored turn landed on a different
arm each time: on incident the **baseline** lost a turn, inflating the +38 in
the library's favour; on vecdb the **treatment** lost one, inflating the −50
against it. Neither figure should be read at face value. Excluding both is
roughly neutral, which is why they are shown rather than dropped.

The `gemma4:12b` · vecdb row predates the 2026-08-12 invalidation work, which
moved that cell to 5/6 on a treatment-only re-run. The pair has not been re-run
together, so the published figure stands until it is.

An earlier `gemma4:12b` vecdb pair was removed for a confound — the arms did not
run the same experiment — and is archived in
[confounded/README.md](confounded/README.md).

## What the set shows

**Accuracy never regressed on the incident scenario**: +0, +33, +38 (qwen at
N=3, the other two at N=1). Every accuracy loss in the table is in vecdb — which is where the documented
supersede limitation lived, and where a stale restatement could outrank its own
correction.

**Six cells cheaper (−10 to −43%), none costlier.** The one costlier cell (+4%)
did not survive N=3 — it is −9.8% when measured three times per arm.

Read that with the exclusion in mind, though: with granite and ministral present
it was five cheaper and *four* costlier. This is a claim about three chosen
models, not about memory-augmented agents in general, and the two excluded ones
are exactly where the cost went the other way.

The excluded runs are what explain the split, and the analysis survives them.
Decomposing granite's +44%: its injected block averaged 841 chars — about
**13% of mean input**. The rest is the system prompt and the windowed message
history. A verbose model that writes long messages and makes more calls swamps
a memory saving that is real but small in absolute terms.

So the claim the data supports is narrower than "cheaper", and it is worth
stating precisely:

> **Retrieval cost is constant and small in every run** — injection is *k*
> claims per call and does not track the store. Whether a *run* is cheaper
> depends on how much of the context is memory rather than the model's own
> output.

**Where the model reliably calls its memory tools, the harness buys
efficiency.** Where it does not, it buys correctness — and the failure mode is
narrower and more troubling than "the model is weak". On incident,
`qwen3.5:9b`'s baseline made these tool calls:

```
incident baseline:  manage_memory ABSENT   ·  search_memory ×3
vecdb    baseline:  manage_memory ×7       ·  search_memory ×2
```

It **searched three times against a store it never once wrote to**, having
written "I've recorded that correction in my memory: Deploy time 13:20 UTC"
with no tool call at all. Its three "I don't have any recorded information"
answers were an accurate report of an empty store it had failed to populate.

The same model handled vecdb fine. So this is not a capability claim: **save
and recall are independent decisions in the stock design and nothing
reconciles them.** Passive capture removes the possibility, because writing is
not a decision. On the same model and prompts the treatment averaged 6.33 of 8
across three runs, against the baseline's 3.67 — and its worst run beat the
baseline's best.

## What this is not, yet

- **N=1 per model.** The qwen3.5 baseline collapse is one observation. The
  *mechanism* is visible in the transcript rather than inferred, which is worth
  more than the count, but it is still one run.
- **Both scenarios are in the table, one run each.** gemma4:12b's vecdb loss is
  the documented shallow-supersede limitation: the injection log shows the
  stale $100k restatement and its own $50k correction reaching the model two
  ranks apart, 0.002 apart in cosine distance.

## A near miss worth recording

`buried_metric_recalled` failed for **both** arms on gemma4:12b, and the
conclusion drawn was that the scenario was under-specified: the investigator
prompt says "record notable findings" while conv-4 asks what the investigation
*measured*, and nobody was told to keep measurements. An amendment to the
investigator prompt was about to be made.

qwen3.5:9b's investigator recorded them unprompted — "packet loss (0.00%),
retransmits (0.02%), DNS resolution delays (3.1ms p99)" — and its answer cited
fact ids for the 41ms figure. The metric is achievable exactly as written;
gemma's investigator was simply less thorough.

Changing the scenario would have been compensating for one model's behaviour
while believing it was fixing a design flaw. It is recorded here because the
reasoning that led there was entirely plausible, which is what makes the failure
mode worth naming.
