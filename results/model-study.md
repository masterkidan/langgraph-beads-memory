# How different models benefit from a memory harness

**Status: in progress — 2 of 5 models measured at N=1.**

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

Every model on Ollama that advertises tool calling and fits 16GB, spanning five
vendors. `qwen3.6`, `kimi-*` and every tool-capable GLM are too large; `glm4:9b`
fits but has no tool calling, which would gut both arms.

| model | size | vendor |
|---|---|---|
| `gemma4:12b` | 7.6GB | Google |
| `qwen3.5:9b` | 6.6GB | Alibaba |
| `granite4.1:8b` | 5.3GB | IBM |
| `ministral-3:14b` | 9.1GB | Mistral |
| `lfm2.5:8b` | 5.2GB | Liquid |

All five pass `demo/smoke_test.py`, which gates on structured tool calls *and*
fact extraction.

## Results so far (incident scenario, N=1)

| model | accuracy | input tokens | verdict |
|---|---|---|---|
| `gemma4:12b` | +0 pts (7/8 both) | **−29%** | win — same, cheaper |
| `qwen3.5:9b` | **+62 pts** (8/8 vs 3/8) | +4% | trade — far more accurate, marginally costlier |

## The emerging shape

Two models, two different payoffs, and the split is informative rather than
noisy.

**Where the model reliably calls its memory tools, the harness buys
efficiency.** gemma4:12b's baseline saved and searched correctly on every turn
and scored 7 of 8; the treatment matched it on 29% less context. Nothing to fix,
so what is left is cost.

**Where the model does not, it buys correctness.** qwen3.5:9b's baseline
produced an empty answer on conv-1 and called no memory tool at all, so nothing
was saved. On the correction turn it wrote:

> "I've recorded that correction in my memory:
>  - **Deploy time**: 13:20 UTC (not 13:50)"

while making **no tool call**. Memory was empty; the model believed otherwise.
conv-3 and conv-4 then answered "I don't have any recorded information about
this incident yet." The treatment, on the same model and the same prompts,
scored 8 of 8 — because capture runs in `before_model`/`after_model` and never
asks the model to decide.

That is the dependency stated in the main README as "capture is model-dependent
— a missed call is a lost fact", appearing unprompted on a model chosen for
being newer, not weaker.

## What this is not, yet

- **N=1 per model.** The qwen3.5 baseline collapse is one observation. The
  *mechanism* is visible in the transcript rather than inferred, which is worth
  more than the count, but it is still one run.
- **Three models unmeasured.**
- **One scenario in this table.** vecdb numbers are being collected; on
  gemma4:12b it showed −29% tokens with one metric lost to the documented
  shallow-supersede limitation.

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
