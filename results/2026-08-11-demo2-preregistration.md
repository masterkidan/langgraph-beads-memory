# Pre-registration — demo 2 (incident investigation)

**Written and committed before the first run.** Check `git log` on this file
against the timestamps in `results/raw/*incident*.json` to verify.

## Why this exists

Demo 1 was designed to exercise `supersedes`, so it could not answer whether
the result generalises. Worse, my own analysis of it repeatedly attributed
metric movements after the fact — and when I finally traced the database, only
one of three runs supported the attribution I had been ready to publish. Stating
the predictions in advance, including where I expect to lose, is the correction
for that.

If a prediction below is wrong, the results document says so in those words.

## Scenario

A production incident (INC-4471) across four threads. Checkout p99 180ms → 4.2s,
error rate 0.3% → 7%. Three investigators cover db / network / apptier. Two
hypotheses die (connection pool exhaustion, DNS); one survives (a synchronous
fraud-scoring call added in release 2.14, 3.9s p99, no circuit breaker). One
timeline fact is corrected mid-investigation (deploy 13:50 → 13:20).

The mechanism is exercised the way an investigation exercises it — eliminations
are native to debugging — rather than by planting a correction. The one planted
correction is a secondary beat, not the headline.

## Arms

| arm | what it changes | what it answers |
|---|---|---|
| `baseline` | LangMem + PostgresStore | the comparison |
| `treatment` | full fact graph | does demo 1 generalise? |
| `treatment-nosupersede` | edges recorded, targets stay `active` | is *typed invalidation* carrying the result, or just per-claim granularity? |
| `treatment-subrecall` | `recall_from_subagents` named in the prompt | does the explicit sub-agent path beat ranked demotion? |

N=5 per arm. Runs are randomised in neither order nor seed — the model is at
temperature 0 and the variance is the model's own.

## Predictions

Stated as `treatment` vs `baseline` unless noted.

### Where I predict this library wins

1. **`avoids_reproposing_ruled_out`** — treatment ≥ baseline, and this is the
   headline. Retiring a superseded fact should keep an eliminated hypothesis off
   the next-steps list. *Confidence: moderate.* The baseline's blobs literally
   contain the words "ruled out", so it is not structurally prevented from
   getting this right — which is what makes it a fair test rather than a
   rigged one.
2. **`uses_corrected_deploy_time` / `avoids_stale_deploy_time`** — treatment
   wins. This is demo 1's budget result in another costume; if it does not
   reproduce, demo 1's headline finding is in doubt. *Confidence: high.*
3. **`names_surviving_cause`** and **`proposes_reversible_fix`** — treatment
   ≥ baseline, weakly. Both depend more on the model than on memory.
   *Confidence: low.*

### Where I predict this library LOSES

4. **`buried_metric_recalled`** (TLS handshake p99 = 41ms) — **baseline wins.**
   The number is incidental: it is not why anything was ruled out, so an
   investigator summarising its conclusion has no reason to carry it up, while a
   flat blob store retains it for free. *Confidence: high.* This is the same
   precision/recall trade demo 1 measured, and I expect it to reproduce.
5. **`breadth_subsystems_named`** — **baseline wins.** "List everything we
   investigated" rewards keeping everything over ranking the most relevant few.
   Top-K retrieval is structurally the wrong tool for a recall-all question.
   *Confidence: moderate.*

### Ablations

6. **`treatment-nosupersede` loses the timeline metrics but keeps the rest.**
   If it *also* loses `avoids_reproposing_ruled_out`, typed invalidation is
   doing that work. If it does not, per-claim granularity is, and the supersede
   machinery is less load-bearing than demo 1 implied. *Confidence: low — this
   is the question, not a hunch.*
7. **`treatment-subrecall` beats plain `treatment` on `buried_metric_recalled`,
   and may beat the baseline.** If the explicit path does not win here it
   probably never will, and the honest conclusion is that summarising sub-agent
   boundaries simply cost raw recall. *Confidence: moderate.*

### No prediction

8. **`numerically_grounded`** — new screen, no prior. Recorded for all arms.
   Its job is to catch the demo-1 failure the blinded judge scored 5/5: a
   confident magnitude that appears nowhere in the source.

## What would falsify the overall hypothesis

If `treatment` does not beat `baseline` on **both** the reproposal metric and
the corrected-timestamp metrics, the claim that a typed fact graph improves
cross-thread continuity does not generalise beyond the scenario built for it,
and the README says so.

## Committed limitations

- N=5, one model (`qwen3:8b`), one machine. Demo 1 showed three identical runs
  scoring 3/6, 6/6, 4/6, so movements smaller than roughly 2/5 will be reported
  as noise regardless of which direction they favour.
- `avoids_reproposing_ruled_out` is a proximity heuristic and can misread both
  ways; its limits are documented on the function and every flagged answer is
  kept in full for inspection.
- The scenario is still authored by the same person who wrote the library. It is
  less tilted than demo 1, not neutral.
