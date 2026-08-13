# How different models benefit from a memory harness

**Status: in progress — 2 of 5 models, both scenarios, N=1 per cell.**

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

## Results so far (N=1 per cell)

| model | scenario | accuracy | input tokens | verdict |
|---|---|---|---|---|
| `gemma4:12b` | incident | +0 pts (7/8 both) | −29% | win — same, cheaper |
| `gemma4:12b` | vecdb | −17 pts (4/6 vs 5/6) | −29% | trade — cheaper, one metric lost |
| `qwen3.5:9b` | incident | +62 pts (8/8 vs 3/8) | +4% | trade — far more accurate |
| `qwen3.5:9b` | vecdb | +0 pts (5/6 both) | −32% | win — same, cheaper |
| `granite4.1:8b` | incident | +12 pts | +44% | trade — more accurate, costlier |

`granite4.1:8b` · vecdb is **excluded and archived**: its treatment delegated
all three researchers on conv-1, a constraints-only turn, while its baseline
did not — so the arms did not run the same experiment. Evidence and reasoning
in [confounded/README.md](confounded/README.md). The same confound removed a
`gemma4:12b` vecdb pair earlier. Its incident pair is unaffected and stays,
including the +44% token result.

Five usable cells: two wins, three trades. Tokens split three cheaper, two
costlier. More mixed than the first four suggested, and the mix is the finding
rather than an inconvenience in it.

## The emerging shape

**The token saving is not unconditional.** It appeared as −29%, −29%, −32% on
`gemma4:12b` and `qwen3.5:9b`, and reversed to +44% and +28% on
`granite4.1:8b`. Decomposing the granite incident run explains why: the
injected block averaged 841 chars (~210 tokens), about **13% of mean input**.
The rest is the system prompt and the windowed message history. granite writes
longer messages and made 19 model calls against the baseline's 18, so its own
output dominated the context and swamped a memory saving that is real but
small in absolute terms.

The claim that survives is narrower than "cheaper": **retrieval cost is
constant and small**, which holds in every run — injection does not track the
store. Whether the *run* is cheaper depends on how much of the context is
memory versus the model's own verbosity.

**Where the model reliably calls its memory tools, the harness buys
efficiency.** Same answers, ~30% less context on two of three models.

**Where it does not, it buys correctness.** On the incident scenario
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
not a decision. On the same model and prompts the treatment scored 8 of 8.

## What this is not, yet

- **N=1 per model.** The qwen3.5 baseline collapse is one observation. The
  *mechanism* is visible in the transcript rather than inferred, which is worth
  more than the count, but it is still one run.
- **Three models unmeasured.**
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
