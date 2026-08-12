# Results — 2026-08-11, qwen3:8b, N=3 (demoted descendant recall)

6/6 runs, **0 errored turns**, 56.0 min. Transcripts in `n3-descendant/`.
Fourth round. Under test: `DESCENDANT_RANK_PENALTY = 0.15` — an orchestrator's
search now also reaches *down* into its sub-agents' namespaces, with child facts
penalised in the ranking so they surface only when nothing better competes.

Aimed squarely at `buried_detail_recalled`, which the directive round drove to
0/3. **It moved — but the evidence that this mechanism is what moved it is much
thinner than the headline number suggests.** That analysis is the substance of
this document.

## Objective metrics

| metric | baseline (n=3) | treatment (n=3) | vs directive round |
|---|---|---|---|
| uses_revised_budget | 0/3 | **2/3** | 3/3 → 2/3 |
| avoids_stale_budget | 3/3 | 2/3 | 3/3 → 2/3 |
| mentions_selfhost | 3/3 | **3/3** | held |
| mentions_primary_sources | 0/3 | 1/3 | **3/3 → 1/3** |
| mentions_feasible_option | 0/3 | **3/3** | held |
| **buried_detail_recalled** | **3/3** | 2/3 | **0/3 → 2/3** |
| mean input tokens | 18,184 | 11,587 | |
| mean output tokens | 1,334 | 1,424 | |

**Blinded judge**: treatment 5.00/5.00/5.00 vs baseline 3.00/2.33/3.00 on
recall/delegation/final. 3 judged, 0 unjudgeable. See the caveat below — the
judge scored a hallucinating run 5/5 on recall, so these numbers are not load
bearing.

Per-run treatment, showing the variance the means hide:

| run | budget | avoids stale | selfhost | primary | feasible | buried |
|---|---|---|---|---|---|---|
| 0 | ✓ | ✓ | ✓ | ✗ | ✓ | ✗ |
| 1 | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| 2 | ✗ | ✗ | ✓ | ✗ | ✓ | ✓ |

Run 1 is the first 6/6 of the project. Run 2 lost the budget result the previous
three rounds held. That spread across three runs of an identical configuration
is itself a finding: at N=3 with this model, single-metric movements of 1/3 are
not distinguishable from noise.

## The confound check passed

`read_document` stays bound in conversation 3, so a correct answer could always
have come from re-reading the corpus rather than from memory. It did not.
Across all six runs, conversation 3 issued **no `read_document` call in either
condition** — the baseline answered via `search_memory`, and the treatment made
zero tool calls at all, because its facts arrive pre-injected in the system
prompt. Every scored recall in this round is a genuine memory recall.

## Attribution: only one of three runs demonstrates the mechanism

`buried_detail_recalled` going 0/3 → 2/3 looks like the fix working. Tracing
where the `32x` figure actually lived before each answer says otherwise:

| run | where the number was, pre-answer | scored | demotion responsible? |
|---|---|---|---|
| 0 | root copy **superseded**; live copy only in a child namespace | ✗ | **no** — child fact was active and reachable, still missed |
| 1 | **active at root** from 00:37, carried up by the rollup | ✓ | **no** — would have succeeded without the feature |
| 2 | **child namespace only** | ✓ | **yes** |

So exactly **one run of three** is a clean demonstration that demoted descendant
recall retrieved something no other path could have supplied. One succeeded for
an unrelated reason, and one failed with the data sitting active in a child
namespace where the feature was supposed to find it.

The honest claim this supports: *demoted descendant recall can recover a
sub-agent's buried detail, and was observed doing so once.* It does not support
a 2/3 recovery rate.

Run 0 is the informative failure. Its answer named the technique correctly and
then invented the magnitude:

> "...it is generally reported to reduce memory usage by **up to 50%** or more."

The corpus says 32x. The fact was in the store, active, one namespace down. A
0.15 penalty was evidently not enough to lift it into the top-8 against the
root-level Qdrant facts competing for the same slots — which is the same
crowding argument that motivated the directive fix, now cutting the other way.

Also worth recording, because it complicates the directive round's conclusion:
in **all three** runs here the rollup *did* carry the quantization detail up to
the root namespace, where in the directive round it never did. The sub-agent
rollups were simply better this time. That is model variance in what a
researcher chooses to summarise, and it sat underneath a result previously
attributed to architecture.

## The cost: `mentions_primary_sources` 3/3 → 1/3

This is the same top-8 budget being spent differently. Adding descendant facts
to the candidate pool means more competitors for a fixed number of injection
slots, and the constraint facts are what lost them — precisely the risk flagged
when the penalty was introduced. `uses_revised_budget` also slipped 3/3 → 2/3.

Two readings, and the data does not separate them:

1. **Real dilution.** The penalty is too weak; child facts compete when they
   should be a last resort.
2. **Noise.** Run-to-run spread in this round is wide enough to produce a 3/3 →
   1/3 swing on its own.

Either way the design conclusion is the same, and it argues against relying on
ranking for this at all: `recall_from_subagents` addresses the same need without
touching the ranked pool, because an orchestrator that delegated a topic can ask
for that researcher's findings directly instead of hoping similarity surfaces
them. **That tool is not exercised by this scenario** — `demo/scenario.py`'s
system prompt never mentions it — so this round tested only the ranking half of
the sub-agent work.

## The judge did not catch a hallucination

The blinded judge gave treatment run 0 `recall: 5` — a perfect score to the run
that fabricated "up to 50%". All three treatment runs scored 5/5/5, so the judge
had no discriminating power at the top of its range in this round, and it did
not penalise a false number in the dimension named "recall".

Weight the objective metrics and the database inspection above the judge here.
The judge's value in earlier rounds was separating clearly different answers; it
is not a fact checker.

## Tokens

Treatment used **36% fewer input tokens** (11,587 vs 18,184). This supersedes
the ~45% figure from the directive round; the direction has been consistent
across every round since the scenario fix, the magnitude has not. Output tokens
are now slightly *higher* for the treatment (1,424 vs 1,334), reversing earlier
rounds — the treatment writes longer final answers when it has more recalled
material to cite.

Treatment runs are slower in wall-clock (9.0/10.9/13.0 min vs 7.8/7.8/7.6),
which is delegation depth and embedding calls, not token volume.

## What this round changes

- **Keep** the descendant penalty. It is cheap, it demonstrably worked once, and
  no metric regression is cleanly attributable to it.
- **Do not** claim it fixes `buried_detail`. One clean demonstration out of
  three runs, and one outright failure with the data present, is not a fix.
- **Next**: put `recall_from_subagents` in the scenario prompt and measure the
  explicit path against the ranked one. That is the comparison this round was
  supposed to make and did not.
- The infrastructure work landed: `ResilientChatOllama` plus the per-turn
  deadline produced the first 6/6 set with **0 errored turns** after three
  consecutive rounds lost turns to a wedged daemon.

## Not poolable with earlier rounds

The retrieval behaviour changed between every round. These three runs are
comparable to each other and to the baseline run alongside them, and to nothing
else.
