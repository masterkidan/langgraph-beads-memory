# Dropped from the benchmark set

`granite4.1:8b` and `ministral-3:14b` were removed on 2026-08-12. Runs are kept
here in full rather than deleted, because the decision was made after their
results were known.

## `ministral-3:14b` — cannot execute the scenario

Not a judgement about its answers. It fails the harness in two ways that are
visible in the tool-call log of every cell it ran:

**It never delegates the full researcher set.** Both scenarios specify three
sub-agents. Ministral called two, in all four cells, missing a different one in
each scenario:

| cell | delegated | never delegated |
|---|---|---|
| incident (both arms) | `researcher_network`, `researcher_apptier` | `researcher_db` |
| vecdb (both arms) | `researcher_qdrant`, `researcher_weaviate` | `researcher_pgvector` |

Every other model in the set delegates all three. A model that never
investigates one of three subsystems cannot name it in an answer, so its
breadth metrics measure delegation compliance, not memory.

**Its baseline arm never retrieves.** `search_memory` call count is **0** in
both scenarios, against `manage_memory` 5 and 1. The baseline writes memory and
never reads it, so the pair is not a comparison of two retrieval strategies —
one side has no retrieval at all. That is the same failure mode already
documented for the built-in tools, but total here rather than intermittent.

Its vecdb cell scored 1/6 for **both** arms, which is what a cell looks like
when neither side could do the task.

## `granite4.1:8b` — a half-set, and no consistent way to keep it

Granite is competent on the harness: it delegates all three researchers and
calls `search_memory` five times. Its incident cell is a genuine treatment win
(5/8 → 7/8).

The problem is that its vecdb pair is confounded and archived — the treatment
delegated all three researchers on a constraints-only turn while the baseline
did not, so the arms did not run the same experiment (see
[../confounded/README.md](../confounded/README.md)). That leaves granite
represented by exactly one scenario: the one that happened to survive.

Reporting a model on whichever scenario came out usable is selection. The
consistent options were to re-run its vecdb pair or drop the model; dropping is
the choice made here, and re-running it would be a legitimate way to bring it
back.

## What the exclusion changes

| | accuracy (treatment vs baseline) | input tokens |
|---|---|---|
| all five models | 4 W / 3 T / 2 L | 5 cheaper / 4 costlier |
| after excluding these two | 2 W / 2 T / 2 L | 5 cheaper / **1** costlier |

It cuts both ways. It gives up two of four accuracy wins — granite's incident
cell is the second-largest treatment win in the set — and it materially
flatters the token claim: with these two present the fact graph costs more
input in four of nine cells; without them the only costlier cell left is qwen
incident at +3.6%.

Any "retrieval is cheaper" statement resting on the reduced set has to carry
that with it.

## The excluded cells

| scenario | model | baseline | treatment | input tokens |
|---|---|---|---|---|
| incident | granite4.1:8b | 5/8 | 7/8 | +44.1% |
| incident | ministral-3:14b | 3/8 | 4/8 | +27.1% |
| vecdb | ministral-3:14b | 1/6 | 1/6 | +64.8% |

One note on granite's `+44.1%`, because it is the single figure the token story
turns on and it is not what it looks like: granite's injected fact block
averaged 841 characters, about **13% of mean input**. The rest is the system
prompt and the windowed message history. The increase is mostly its own
verbosity, not the memory layer. That analysis is worth keeping even though the
model is not — it is why the honest claim is "retrieval cost is constant and
small", not "runs get cheaper".
