# Demo results

This directory holds the evidence base for the `langgraph-beads-memory` comparison.

- `raw/` — per-run transcripts + metrics as JSON (gitignored; regenerate with the harness)
- `archive/` — superseded runs kept for provenance (see below)
- `<date>-results.md` — the written-up comparison

## Reproducing

```bash
docker compose up -d                      # Postgres + pgvector on :5433
ollama serve                              # or: brew services start ollama
ollama pull qwen3:8b && ollama pull nomic-embed-text
uv sync --all-extras
uv run python -m demo.smoke_test          # pre-flight gate — must pass first
uv run python -m demo.harness --runs 3    # ~30 min per run, 6 runs
uv run python -m demo.aggregate results/raw
uv run python -m demo.judge results/raw
```

The smoke test is a genuine gate, not a formality. A model that cannot make
structured tool calls guts the treatment; a model that cannot extract facts
guts the baseline. Either failure produces a meaningless comparison, so fix the
model rather than proceeding.

## Two scenarios, four arms

`--scenario` selects which demo runs; `--conditions` selects arms.

| scenario | what it is | why it exists |
|---|---|---|
| `vecdb` (default) | pick a vector database across 3 threads | demo 1 — built around a planted budget correction |
| `incident` | debug a production incident across 4 threads | demo 2 — eliminations are native to the domain, so nothing is planted |

| arm | memory configuration |
|---|---|
| `baseline` | LangMem + `PostgresStore` |
| `treatment` | full typed fact graph |
| `treatment-nosupersede` | supersede edges recorded, targets stay `active` — isolates typed invalidation from per-claim granularity |
| `treatment-subrecall` | `recall_from_subagents` named in the prompt — tests the explicit sub-agent path against ranked demotion |

```bash
uv run python -m demo.harness --runs 5 --scenario incident \
  --conditions baseline treatment treatment-nosupersede treatment-subrecall
```

**Demo 2's predictions were committed before its first run** — see
[the pre-registration](2026-08-11-demo2-preregistration.md), which names the two
metrics where the baseline is expected to win. Check its commit timestamp
against the run files.

Always run one scenario per `results/raw` directory: `aggregate` refuses to pool
two scenarios, because their metrics are not comparable.

**Two things demo 2 cost before it produced a single number**, both worth knowing
if you extend the harness:

- The agent delegated on the *first* turn, which only reports symptoms — 280s
  and three investigators for a turn needing one model call. Demo 1 had the
  identical bug. Any new scenario should be traced for spurious delegation on
  its opening turn before a real run.
- The demo built an `ollama.Client` per turn *and* per sub-agent invocation, so
  a run leaked ~24 connection pools. Clients are cached now; if you add a call
  site, use `make_llm()` rather than constructing one.

## Running this without losing hours to a wedged server

**Restart Ollama between runs. It wedges under sustained load — roughly every 20 minutes of active use on this hardware.**

```bash
brew services restart ollama          # after ANY sleep/wake, not just when it looks stuck
caffeinate -dimsu uv run python -m demo.harness --runs 3
```

Six separate runs hung with the same signature: the client blocked in
`sock_recv` on an accepted-but-unanswered request, `ollama runner` absent, and
`/api/ps` reporting no model loaded. Control endpoints stay healthy while
generation is dead:

```
/api/version   200
/api/ps        200
/api/generate  000   <- accepted, never answered
```

Only a full daemon restart clears it. This is
[ollama#15950](https://github.com/ollama/ollama/issues/15950).

**A wrong diagnosis worth recording.** This was first attributed to system
sleep, because every early hang followed a wake. That correlation was an
artifact — the daemon was being restarted at the same time sleep was eliminated,
so the recovery came from the restart, not from staying awake. It then
reproduced on a 21-minute-old daemon with zero sleep events. Sustained load is
sufficient on its own.

Requests **hang rather than failing**, which is indistinguishable from a slow
generation until minutes have passed — so a stall detector that probes
`/api/generate` is worth more than one that only watches for log silence.

The code fixes made while chasing this are real and worth keeping — per-thread
Postgres connections, a pooled store for the baseline, bounded timeouts on both
chat and embedding calls, and a faulthandler watchdog that dumps every thread's
Python stack on an unbreakable hang. None of them was the cause.

**The recovery policy, and a correction about it.** `ResilientChatOllama` (see
`demo/llm.py`) treats a wedge as repairable rather than something to wait out:
on any transport failure it probes `/api/generate` (the control endpoints lie),
restarts the daemon only if it is genuinely wedged, rebuilds the client so the
retry cannot land on a dead pooled connection, and retries once. A bare retry
does not work — it dispatches onto the same wedged connection.

The 2026-08-11 round was originally described here as the first
**0-errored-turn** N=3 set *because of* that policy. That was wrong. The rebuild
set `_client = None`, and langchain_ollama raises `RuntimeError` on a null
client instead of reconstructing it, so every retry failed before touching the
network. The policy recovered nothing; those runs were clean because Ollama did
not wedge, helped by the driver restarting it between runs. Fixed on 2026-08-11
by re-running `_set_clients()` — the validator that builds the clients — and
verified end to end.

**A likely contributor to the wedging itself.** Until 2026-08-11 the demo built
a fresh `ollama.Client`, and therefore a fresh `httpx.Client` connection pool,
per turn *and* per sub-agent invocation — roughly 24 per run, most for
researchers never called on that turn, none ever closed. One in-flight profile
run held 5 ESTABLISHED connections for a single conversation. Clients are now
cached per `(model, temperature, reasoning)` and closed between runs. Whether
this removes the wedge is untested; it is a hypothesis, and the connection churn
was waste either way.

## Results index

Four scored rounds, each superseding the last. They are **not poolable** — the
memory behaviour changed between them, which is the point of running them
separately.

| round | change under test | headline |
|---|---|---|
| [2026-08-19](2026-08-19-context-budget.md) | context budget, external benchmark, N=3 matrix | memory's value is **negative** when the transcript fits; `num_ctx` truncation invalidates prior token figures |
| [2026-08-09](2026-08-09-results.md) | first scored comparison | budget recall 0/3 vs 3/3; `primary_sources` 1/3 |
| [2026-08-10 postfix](2026-08-10-postfix-results.md) | per-claim splitting + `supersedes` similarity guard | guard eliminated spurious edges; `primary_sources` unmoved at 1/3 |
| [2026-08-10 directive](2026-08-10-directive-results.md) | `directive` kind held out of retrieval | `primary_sources` 1/3 → 3/3; `buried_detail` fell to 0/3 |
| [2026-08-11 descendant](2026-08-11-descendant-results.md) | demoted descendant recall (`DESCENDANT_RANK_PENALTY`) | `buried_detail` 0/3 → 2/3, but only **1 of 3** runs attributable to the mechanism |

The latest round is the first with **0 errored turns**, after the resilience
work below. Read its attribution section before quoting its numbers: the
headline metric moved, and tracing the database showed most of that movement was
not caused by the feature under test.

An aborted qwen3:4b attempt is kept in `aborted-4b/` — it wrote ~12,000 output
tokens per run against 8b's ~1,400 and was reverted. It is not pooled with
anything.

## SUPERSEDED — read [2026-08-19](2026-08-19-context-budget.md) first

Everything below predates the context-budget round and three of its conclusions
did not survive it:

- **The headline separation is gone.** At N=3 across two models and two
  scenarios, no accuracy gain is established; one cell is a clean regression.
- **Every token figure here was measured under silent truncation.** Ollama's
  default `num_ctx` is 2048 and `demo/llm.py` never set one, so any prompt above
  that was cut before evaluation. It fell hardest on the baseline.
- **The value is conditional on context pressure.** At a matched budget, the
  memory layer is worth +28 of 78 when the window is tight and −1 (a tie) when
  the conversation nearly fits.

The method notes, the operational guidance, and the disclosed corrections below
all still stand. The *numbers* should be read as historical.

## A caveat that may supersede all of the above

Every round in the index ran on `qwen3:8b`. On 2026-08-12 a first clean run on
`gemma4:12b` had the **baseline carry the revised budget correctly** — the
metric that is 0/3 for the baseline in all four qwen3 rounds and is the
headline separation of this whole comparison. The same baseline also committed
to a feasible option, which it never managed on qwen3.

One run is not a refutation, and the paired treatment arm in that run was
confounded (it delegated on a turn that only states constraints, so the two
arms did not run the same experiment). But the direction is clear enough to
state: **part of the measured advantage here may be a measurement of
`qwen3:8b`'s limitations rather than of memory architecture.**

What would survive a stronger model regardless: bounded context cost —
injection is k facts per turn no matter how much the session accumulates —
and audit provenance. What may not: recall separations that a more capable
model achieves on its own from a flat store.

Two further lessons from the model change, both cheap to repeat and expensive
to skip:

- **Prompts are model-tuned.** Every delegation rule in both scenarios was
  shaped against qwen3's failure modes. gemma4:12b ignored them and delegated
  three researchers on a constraints-only turn (602s versus 33s). Trace the
  opening turn of every scenario after any model change.
- **Four of the bugs found while validating demo 2 degraded only the baseline.**
  The treatment captures passively and enforces its own sub-agent contract, so
  it fails gracefully exactly where the baseline needs the model to get a tool
  call right. A comparison that does not repair those is measuring the model's
  tool-calling, not the architecture.

## Method

One scripted narrative, run identically under two conditions. The **only**
variable is the memory layer.

- **Baseline** — LangGraph `PostgresStore` + LangMem, wired the idiomatic
  supervisor way: sub-agent results return as messages, memory store shared by
  all agents. Given a real pgvector index so its search is genuinely semantic.
- **Treatment** — `langgraph-beads-memory`: typed fact graph, passive capture,
  forked sub-agent namespaces with enforced rollup.

Both get the same model (`qwen3:8b`, temperature 0, thinking disabled), the same
corpus tool, the same prompts, the same 3-way sub-agent split, and **no
checkpointer on either side** — so cross-conversation continuity must come from
the memory layer, which is the whole point of the comparison.

Scenario: one session, three conversations (three LangGraph threads). Conversation 1
states constraints ($100k budget, self-hostable, primary benchmark data only).
Conversation 2 delegates pgvector/Qdrant/Weaviate research to three sub-agents,
then the user revises the budget to $50k. Conversation 3 — a new thread — asks
which database to pick, then asks for a detail that exists only in one
researcher's findings.

Scoring is in two parts: **objective metrics** (literal substring checks over the
final answers, plus token accounting — no judge involved) and a **blinded LLM
judge** scoring recall, delegation, and final-answer quality, with condition
labels stripped and presentation order randomized. The judge marks a pair
*unjudgeable* rather than ever fabricating a score.

## This is a designed demonstration, not a neutral benchmark

The scenario was built to exercise a specific mechanism, and it is tuned to do
so. That is what a demonstration is for — but it means these numbers show that
the mechanism works on a case it was designed for, not that it wins in general.
Anyone citing them should say so. What we do *not* do: hide unfavourable
measurements, or present this as an independent benchmark.

## Corrections and changes, disclosed

Everything below was found by inspecting real runs, and is recorded because
some of it changed results after the fact.

**Three metric bugs, all false readings from literal substring matching:**

1. `buried_detail_recalled` required the literal `"32x"`; the model wrote
   `"32 times"` — correct, scored wrong. Found in a **baseline** run; fixing it
   **raised the baseline's score**.
2. `uses_revised_budget` required `"50k"`; the treatment wrote `"$50,000"` —
   correct, scored wrong. Fixing it raised the treatment's score.
3. `pick_is_feasible` scored True for an answer that named two databases while
   refusing to recommend either. Substring matching cannot distinguish a
   recommendation from a mention. Renamed `mentions_feasible_option`; whether an
   answer actually commits is left to the judge's `final` dimension.

The pattern is the point: three literal metrics, three wrong readings. Treat the
objective metrics as coarse signals and weight the judge's dimensions more
heavily.

**Two scenario changes, both made after seeing data.**

*The buried-detail question (after run 0).* It originally asked about "the
strongest runner-up" without naming it. Its referent depends on which database
the agent picked, so it measured *choice* rather than *recall* — in run 0 the
treatment picked Qdrant, read "runner-up" as Weaviate, and hallucinated an
optimization for it, while the baseline scored the point only because it
declined to pick anything. The question now names Qdrant.

*Delegation discipline in the system prompt (after profiling).* A span profile
of conversation 1 — a turn that only states constraints and asks for nothing —
showed the agent spawning all three researchers, ~140s each, ~199s for a turn
that needs a single 11.6s model call. The prompt now states when *not* to
delegate; the same turn then took 13s with one model call and no delegation.

This was a **correctness** fix as much as a speed one: with conversation 1 doing
the full research, conversation 2 was re-researching rather than researching, so
the delegation the demo claims to measure was not the delegation being measured.
The identical wording goes to both conditions (the baseline's version is a
string substitution of the same text, verified not to leak `remember_fact` or
`supersedes` into it), so it favours neither side.

Rationale for both is recorded inline in `demo/scenario.py`.

Because of these changes, **run 0 is archived rather than pooled** — it answers
a different question under different delegation behaviour and cannot be averaged
with later runs. It is kept in `archive/` because it is the evidence behind
every correction above.

## Profiling

`uv run python -m demo.profile_run [--condition baseline] [--conversations conv-1]`
runs one scenario under a span profiler and prints where the time went, writing
the raw spans to `results/profile-*.json` so a slow run can be re-analysed
without paying for it twice.

It records a span per model call and per tool call via LangChain callbacks, plus
wrappers on the embedder and the Postgres connection, so time is attributed to
the work that caused it rather than inferred from turn boundaries. Because
sub-agents run concurrently and a tool span contains the model calls made inside
it, a naive sum exceeds the wall clock — the report shows *busy* time (union of
intervals, overlap counted once) next to the raw sum, and their ratio as
effective concurrency.

What it established on this hardware (M4, 16GB, qwen3:8b):

- Runs are **GPU-bound**. `OLLAMA_NUM_PARALLEL=3` does produce real concurrency
  (~2.4x measured), but per-call throughput falls proportionally — 4.6 output
  tok/s under three concurrent streams versus ~17 tok/s single-stream. Total
  throughput is roughly fixed; concurrency redistributes it.
- **Postgres and embeddings are negligible**: 107 queries totalling 0.3s and 35
  embeddings totalling 5.0s in one conversation, together under 3% of wall time.
  Optimising either would be wasted effort.
- The one large win available was **not doing unnecessary work** — the
  over-delegation above.

`demo/aggregate.py` rescores every transcript with current metric code rather
than trusting the `constraint_carry` snapshot stored in each JSON, so runs
recorded before a metric fix are never silently pooled with runs recorded after
one.

## `supersedes` is shallow — a known architectural limitation

Measured 2026-08-12, gemma4:12b, vecdb N=1. The treatment cited the **stale**
$100k budget twice in its final answer, on the metric that is this library's
flagship result (3/3 on qwen3).

The correction fired correctly. The original user statement was retired:

```
[superseded] user_input   the budget is $100k per year
```

But every claim the agent had already DERIVED from it stayed active:

```
[active] conclusion  remember_tool     The annual budget ... is $100,000.
[active] conclusion  passive_capture   I have recorded your requirements: a $100,000 annual budget...
[active] conclusion  remember_tool     pgvector ... fits well within the budget
[active] conclusion  passive_capture   All three ... fall within the $100,000 annual budget
```

**A `supersedes` edge retires the row it points at, and nothing else.** The
stale value survives in every restatement, those restatements remain
retrievable, and the ranker can surface them ahead of the corrected fact —
which is exactly what happened. Both values were live in the store at once.

The schema defines a `derived_from` relation, but passive capture never emits
those edges, so there is no path along which an invalidation could cascade.
"Typed invalidation retires the stale fact" is true only of the fact you cite.

**A change made the same day probably worsened it.** Splitting agent
conclusions per claim (which cut input tokens ~40%) turned one restatement
blob into several individually-retrievable facts, each carrying the stale
figure. Token cost improved and supersede effectiveness plausibly degraded in
the same commit. Both effects are measured; the causal link between them is
inference, not measurement.

**Partially fixed 2026-08-12** by cascading a supersede to *earlier* facts
similar to the one being retired (`BeadsStore._cascade_supersede`). The time
ordering is what makes it safe: embeddings cannot tell "$100k" from "$50k", so
similarity alone would retire the correction itself.

It is a partial fix and the gap is measured, not assumed. Two real runs demand
incompatible thresholds:

| run | stale restatement | nearest fact that must be KEPT |
|---|---|---|
| incident | 0.973 / 0.972 (verbatim repeat) | 0.689 — the root-cause synthesis |
| vecdb | 0.720 (paraphrase) | 0.500 |

A threshold low enough to catch vecdb's paraphrase would retire incident's
root-cause synthesis — deleting the finding that lets the agent name the cause,
which is worse than leaving a stale value. So the floor was 0.85: near-verbatim
restatements retired, reworded ones surviving.

### Resolved 2026-08-12 — read the value instead of the distance

The threshold search was the wrong frame. 0.720 and 0.689 are 0.03 apart, so no
threshold splits them, and no amount of tuning was going to. The two sentences
differ in exactly the token embeddings are worst at — the number itself — so it
is now read directly rather than inferred from a distance.

Two changes:

**`derived_from` edges are emitted at capture time.** The facts injected into a
model call are exactly what memory contributed to the answer, so `after_model`
links each captured conclusion to them. No extraction call, which is what makes
this viable in the no-LLM hot path — the earlier note that this "cannot be done"
assumed provenance had to be inferred from the text, when it was already sitting
in the injection set.

**The cascade is value-aware** (`_is_stale_restatement`). A fact is retired when
it carries a *contested* value and not its replacement. Contested matters: the
incident's corrected fact also carries "2.14", which is still true, so matching
on "shares any figure" would retire "Release 2.14 introduced a caching layer".
A value counts only when the correction asserts a different value of the same
shape — `13:50 → 13:20`, leaving `2.14` alone. A fact carrying *both* values is
the corrected version restating what it supersedes and must survive.

Replayed over every fact in the five stored treatment runs, respecting the
created_at ordering, the rule retires 1–3 facts per incident run and 1–5 per
vecdb run, and every retirement is a genuine stale assertion.

Live re-run, gemma4:12b, vecdb N=1 — the same configuration as the regression
at the top of this section:

| | before | after |
|---|---|---|
| `uses_revised_budget` | ✗ | **✓** |
| `avoids_stale_budget_as_current` | ✗ | **✓** |
| `mentions_primary_sources` | ✓ | **✗** |
| input tokens | 11,115 | 9,848 |
| `derived_from` edges | 0 | 267 |

5 of 6 against 4 of 6. The token drop is a side effect worth naming: a retired
fact stops being injected, so invalidation pays for itself in context.

### The cost, which is real and is not a cascade bug

`mentions_primary_sources` regressed, and the cascade caused it. The retired set
included:

> "I have recorded your requirements: a $100,000 annual budget, the requirement
> for a self-hostable solution, and the constraint to rely exclusively on
> internal primary benchmark data rather than vendor marketing."

One fact asserting three constraints, one of which went stale. Retiring it took
the other two with it. The three sub-agent summaries went the same way — each
paired "fits within the $100,000 budget" with "Internal benchmarks confirm...".

The user's original "I only trust primary benchmark data" stayed active
throughout, so the constraint was still in the store; it just stopped being
restated where the model would trip over it.

This is a **splitting** failure surfacing as an invalidation cost. A fact that
mixes a stale premise with live constraints cannot be invalidated cleanly by any
mechanism, because there is no granularity at which to be correct. Two specific
gaps:

1. `_split_enumeration` merges noun-phrase enumerations back together, since
   "the requirement for a self-hostable solution" has no clause verb. The
   under-splitting bias is right in general and wrong for a labelled list.
2. Sub-agent summaries are written whole by `conclude_task` and never split,
   unlike conclusions.

Both are worth fixing before reading too much into a single metric on N=1.

### Follow-up 2026-08-12: summaries split, and one voice cannot take the floor

**Summaries are now split per claim** (`conclude_task`), the last un-split write
path. Verified in the store rather than inferred from a score: each researcher's
summary becomes 4–5 facts, and correcting the budget now retires only the
budget-bearing clause of each. "Internal benchmarks confirm high performance
(40ms p95 at 10M vectors)" and "binary quantization (up to 32x)" stay active
where previously all three summaries were retired whole.

**No single sub-agent may take more than `DESCENDANT_MAX_PER_AGENT` of the
parent's slots.** Splitting removed a property that had held by accident: one
researcher crossed as one summary fact and so could occupy one slot. Once split,
one researcher's fragments could take several while others took none.

The first version of this cap keyed on namespace and was a **no-op**, which the
injection logs caught. `conclude_task` writes a summary into the PARENT
namespace tagged with the child's agent_id, so summary fragments are not
descendant rows at all — the cap rationed raw child exploration, which was never
the problem. Keyed on agent_id and restricted to delegated sources, it binds:
max facts from one agent in any injection went 3 → 2.

### What is and is not established

Four gemma4:12b runs per scenario, N=1 each, changes applied cumulatively.

Structural claims, verified by reading the store directly:

- `derived_from` edges are emitted where none existed (203–267 per run).
- The cascade retires the stale assertions and keeps the corrected ones.
- Summaries split, and invalidation reaches one clause instead of the whole.
- The per-agent cap binds.

Metric claims, and how much weight each carries:

| | orig | four subsequent runs |
|---|---|---|
| `uses_revised_budget` | ✗ | ✓ ✓ ✓ ✓ |
| `avoids_stale_budget_as_current` | ✗ | ✓ ✓ ✓ ✓ |
| `mentions_primary_sources` | ✓ | ✗ ✓ ✗ ✗ |
| vecdb input tokens | 11,115 | 9,848 → 8,800 |
| incident input tokens | 16,745 | 14,320 → 12,861 |

The two budget metrics are the result this work targeted and they hold across
four consecutive runs. **`mentions_primary_sources` flaps**, and it flaps in
runs whose stores demonstrably contain the primary-source facts as active —
so it is measuring the model's phrasing, not retrieval, and no claim should
rest on it.

Incident objective metrics went 7 → 4 → 7 → 6 across the same sequence, which
is scatter, not a trend: N=1 cannot separate a 1-metric movement from noise, as
the 2026-08-11 round already established (three identical configurations scored
3/6, 6/6 and 4/6). `breadth_subsystems_named` sat at 2 in all three post-split
runs against 3 in the original, which is the one candidate regression worth
chasing — but retrieval delivered all three researchers to the parent in those
runs, so if it is real it is downstream of memory.

**Nothing here should be published at N=1.** The honest next step is N≥3 per
arm on both scenarios.

## Known limitations

- **Small-model noise.** `qwen3:8b` is unreliable at root-level tool calling when
  many tools are bound — it sometimes emits a malformed pseudo-tool-call as plain
  text. The treatment's passive capture paths are designed to be independent of
  tool-calling, but individual runs are noisy. This is why N>1 and means matter.
- **Observed model failures worth reporting, not hiding:** a sub-agent once
  invented a spurious `supersedes` edge against an unrelated fact; the treatment
  once hallucinated a memory optimization that appears nowhere in the corpus;
  near-duplicate `remember_fact` writes recur because content-derived ids dedupe
  only *identical* text.
- **N is small.** Each full run is ~30 minutes of local inference. N=3 is thin
  evidence; inconsistent results should be reported as inconsistent, not
  resolved in either direction. The 2026-08-11 round makes this concrete: three
  runs of an identical configuration produced 3/6, 6/6 and 4/6 on the objective
  metrics, so a 1/3 movement in any single metric is not distinguishable from
  noise.
- **The judge is not a fact checker.** In the 2026-08-11 round it scored a run
  `recall: 5` whose answer named the right technique with a fabricated
  magnitude, and gave all three treatment runs 5/5/5 — no discriminating power
  at the top of its range. Where a metric and the judge disagree, inspect the
  database.
- **A metric moving is not the feature working.** Also from 2026-08-11: the
  targeted metric improved, but tracing where each recalled fact actually lived
  showed only one of three runs was attributable to the change under test.
  Attribute movements by inspection, not by timing.
- **The baseline's delegation channel was empty in every round up to and
  including 2026-08-11.** Its sub-agent wrapper returned the last message's
  content, and a researcher that ends on a tool call leaves that empty —
  9 empty returns out of 9, against 0 of 9 for the treatment, which enforces
  `conclude_task` and synthesises a fallback. The baseline could still reach
  those findings via `search_memory`, so it was not blind, but its direct
  channel carried nothing. Fixed 2026-08-11; earlier rounds are not re-run, so
  their delegation-dependent baseline numbers are a lower bound. Full note in
  the [descendant results](2026-08-11-descendant-results.md).

**Why so many of these corrections favour the baseline.** Four separate bugs
found while validating demo 2 all degraded the baseline and none degraded the
treatment: empty sub-agent returns, `manage_memory` calls rejected at the schema
boundary, a retry loop that burned a whole turn, and sub-agents running under a
different recursion limit. That asymmetry has a cause worth naming — the
treatment captures passively and enforces its own sub-agent contract, so it
degrades gracefully where the baseline depends on the model getting a tool call
exactly right. A comparison that does not repair those is measuring the model's
tool-calling, not the memory architecture.
- **A confound to keep checking.** `read_document` stays available in
  conversation 3, so an agent could bypass memory by re-reading the corpus. In
  run 0 neither condition did (the baseline used `search_memory`), but this
  should be verified per run rather than assumed.
