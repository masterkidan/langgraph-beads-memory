# The development loop

Three tiers, cheapest first. A change earns promotion to the next tier by
succeeding at the one below it. Nothing skips a tier, and **tier 0 is a
conversation, not a command.**

The reason for the structure: a full N=3 run is ~30 minutes of local inference
and produces a number with a 4-to-7-of-8 spread. Using it to evaluate a ranking
tweak is both slow and, at that spread, close to uninformative. Ranking does not
need an LLM to evaluate, so it should not be paying for one.

---

## Tier 0 — new abstract concepts get run by the human first

**Before implementing**, not after. A new *concept* is anything that changes the
vocabulary of the system rather than its parameters:

- a new point of capture (a hook, a tool, a lifecycle event)
- a new edge relation, or a change to which edges `search_memory` returns
- a new weighting *scheme* for edges or kinds (not a new value for an existing weight)
- a new fact kind, status, or source
- a new scope rule (who can read whose namespace)

These are cheap to write and expensive to remove, because each one becomes a
thing every later change has to reason about. They also tend to look obviously
correct in advance and turn out to be conditional — the per-source cap and the
descendant-penalty exemption both read as clearly right and were both measured
worthless or harmful within the hour.

Tuning an existing knob is **not** a new concept and does not need this step.

---

## Tier 1 — the inner loop: frozen store, no LLM

**Run this until it stops improving. Iterate freely; overfitting is acceptable
here, with one condition.**

```bash
# capture a store once (bodies, embeddings, namespace tree, edges)
uv run python -m demo.retrieval_fixture \
    --schema run_incident_treatment_1_fb6dbb \
    --baseline-prefix memories.baseline-run1-24535a \
    --out results/fixtures/incident-run1.json

# score ranking against it, ~1 second, deterministic
uv run python -m demo.retrieval_eval results/fixtures/incident-run1.json --sweep
```

The fixture is a real store produced by a real run — whatever the model actually
did with the memory tools and the capture hooks. Changes to **ranking** and to
**what gets written** are both evaluable against it without generation.

### The condition on overfitting

Overfitting is fine *because* the changes are supposed to be structural rather
than fitted. The distinction that matters:

| | example | tier-1 verdict is |
|---|---|---|
| **concept** | "rollups should be guaranteed a slot" | evidence, promote it |
| **parameter** | "the penalty should be 0.15, not 0.12" | a hint, nothing more |

Fitting a scalar to one fixture produces a number that describes the fixture.
`CLAIM_WEIGHT=0.5` with a per-source cap of 3 was chosen by reasoning, shipped,
and then measured as the worst of twenty cells on the very fixture it was meant
to serve.

**A concept passing tier 1 is not a concept that generalises.** Both floors
tried during the ranking work passed tier 1 convincingly and each helped exactly
the scenario that motivated it: the per-agent floor was +11 points on incident
and 0 on vecdb, the category floor +12 on vecdb and **−6** on incident. Tier 1
tells you a concept *can* work. Tier 2 is what tells you *where*.

### Always score against at least two fixtures

One per scenario, minimum. A change that improves one and degrades the other has
not been validated, it has been localised.

### Separate capture failures from ranking failures

If no item in the store covers an aspect, that is a **capture** bug and no
ranking change can touch it. Pool the two and every strategy looks worse by a
constant, and the real defect hides. `buried_metric_recalled` sat at 0/3 through
an entire benchmark for this reason: a sub-agent read a TLS handshake p99 of
41ms and dropped it when summarising, so the fact was never written.

---

## Tier 2 — N=1, two models, both scenarios

```bash
scripts/setup_local.sh gemma4:12b qwen3.6           # pulls + smoke gate
scripts/run_model_matrix.sh incident 1 "gemma4:12b qwen3.6" "baseline treatment"
scripts/run_model_matrix.sh vecdb    1 "gemma4:12b qwen3.6" "baseline treatment"
uv run python -m demo.compare_models results/matrix
```

Two models because a change that helps one model's quirks is not a change to the
memory layer. Both scenarios because they fail in opposite directions — on
incident, constraints crowd out findings; on vecdb, findings crowd out a
constraint. A single scenario cannot see that.

**Run the smoke gate first, and believe it.** A model that cannot emit tool
calls produces an empty treatment arm, which reads as "the memory layer lost".
`glm4:9b` advertises `tools` in `/api/show` and its template handles them, and
it still returned zero tool calls on the gate prompt across two runs.

**What tier 2 can and cannot tell you.** It can tell you a change is directionally
wrong, or that it helps one model and not another. It **cannot** tell you a
change works — the run-to-run spread on incident is 4 to 7 of 8, so a single run
landing high means nothing on its own. Three separate N=1 results were reported
as wins during one session and all three were smaller or absent at N=3.

Treat tier 2 as a filter that kills bad changes cheaply, never as confirmation.

---

## Tier 3 — N=3+, statistical

```bash
scripts/run_model_matrix.sh incident 3 "gemma4:12b qwen3.6" "baseline treatment treatment-searchtool"
```

Report **every run, not a mean hiding a range.** `6, 6, 5` and `7, 4, 6` have the
same mean and are not the same result.

At N=3 a difference of one metric is not a result. What is reportable:

- non-overlapping distributions
- a per-metric change that is consistent across all runs (0/3 → 3/3 is a finding; 1/3 → 0/3 is not)
- a spread that narrows (6,4,4 → 6,6,5 is real information even though the means differ by one)

**Never pool across scenarios or models.** `aggregate.py` refuses to, and
`compare_models.py` compares within-model deltas, because pooling attributes a
model's own strength to the harness. The library beats baseline on vecdb and
trails on incident; a pooled number would report neither.

---

## Rules that apply at every tier

**Never edit source while a matrix is running.** Each run is a fresh process, so
an edit mid-matrix puts two codebases inside one N=3 arm. This has already
happened once — `schema.sql` was edited one second after a run finished, and the
matrix had to be stopped and four runs quarantined.

**Hold the interface constant when measuring the store.** The arms differed in
store, ranking *and* interface simultaneously, so no number could be attributed
to any of them. `treatment-searchtool` exists to pin the interface; use it
whenever the claim is about ranking or capture.

**Check the mechanism, not just the metric.** A run once scored 7/8 and it was
credited to a ranking fix; the transcript showed the answer came from a
`recall_from_subagents` tool call, and the injected block still lacked the root
cause. The score was right and the attribution was wrong.

**Store growth is not free.** The root namespace fits ~94 facts into 8 slots with
a rank-8/rank-9 margin of 0.00126. Every change that adds candidates makes
selection more fragile, and every store-growth change so far has improved one
metric and degraded another. When adding capture, scope it to the namespace of
the agent that produced it rather than the root.

**Write down what was disproved.** Comments in this codebase carry the measured
reason for their filter, and that is what stops a bad idea being re-implemented.
Three plausible hypotheses died in one afternoon; without the comments the next
person tries all three again.
