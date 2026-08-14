# Open questions

Live at the end of the retrieval-diagnosis work. Ordered by cost-to-value: the
first three need no new inference and interact with each other, so they are
worth doing as one change and validating with a single tier-3 run.

Follow [development-loop.md](development-loop.md). Items 1–3 are tuning and
bug-fixing within existing concepts, so none of them needs a tier-0
conversation; item 4 is documentation; item 5 is setup.

---

## 1. `search` has no deterministic tiebreak

`BeadsStore.search` ends its `ORDER BY` at the blended distance with no final
key, so near-ties resolve to whatever physical order Postgres returns — which
follows insert order.

Insert order is not stable across runs. `random_fork_suffix()` uses
`secrets.token_hex(4)` per fork (deliberately — concurrent sub-agents must not
collide), namespace id feeds `derive_fact_id`, and the three researchers race on
ToolNode's thread pool. So identical content produces different sub-agent fact
ids and different write order every run.

It matters because the margins are tiny. Measured on one incident fixture: 93
candidates for 8 slots, **rank 8 to rank 9 differing by 0.00126**, with 8 facts
inside 0.01 of the cut. Selection is decided in the third decimal place, and
part of the run-to-run variance is therefore self-inflicted rather than the
model's.

**`, id` will not fix it** — sub-agent fact ids are reseeded per run. The
tiebreak has to be content-derived (`md5(body)`, or `body` itself) to be stable.

Testable against a frozen fixture in about a second, no LLM.

## 2. The shipped ranking defaults are the worst measured cell

`CLAIM_WEIGHT=0.5` and `CONTEXT_MAX_PER_SOURCE=3` were chosen by reasoning,
shipped, and then measured as the **worst of twenty cells** in the offline grid
on the fixture they were meant to serve — 33% aspect coverage against 67% for
`claim_weight=0.3` with the per-source cap off.

The per-source cap is the specific culprit: it collapses coverage at cap 2–3 for
every `claim_weight >= 0.3`, and only helps at 0.0. It is also a knob invented
in the same session with no measured backing, so turning it **off** returns to
evidence-backed behaviour rather than substituting one guess for another.

Not changed yet because tuning a scalar on a single fixture is exactly the
failure mode [development-loop.md](development-loop.md) warns about. Needs at
least two fixtures before picking a value.

## 3. Breadth: the agent floor is validated but unshipped

`breadth_complete` is essentially the whole remaining incident gap — baseline
3/3, both treatment arms 1/3 at N=3.

The cause is structural rather than a ranking bug: "list everything we
investigated and ruled out" is an *enumeration* query wanting one item per
subsystem, and top-k by similarity has no notion of spread. Three facts about
the database beat one each from three subsystems.

Two floors were tried offline. Each fixed its motivating scenario and did
nothing or harm on the other:

| | incident | vecdb |
|---|---|---|
| per-agent floor (>=1 slot per reporting sub-agent) | 89% → **100%** | 88% → 88% |
| category floor (reserve slots for user constraints) | 89% → **83%** | 88% → **100%** |

That looked like overfitting and it is why neither shipped. The N=3 result
weakens that reading for the per-agent floor specifically: vecdb has no
enumeration query, so a fix aimed at enumeration *should* be inert there. Worth
re-deciding with that in mind.

Note it interacts with item 4 of the toolcapture trade below — capture grows the
candidate pool and makes breadth worse, so the floor and tool capture should be
tested together rather than separately.

## 4. The README's headline claims did not survive measurement

It leads with **"−9.8% input tokens and 6.33 of 8 objective metrics against
3.67"** and presents typed ranking as producing better retrieval. Three
problems:

- **Every two-arm number confounds three variables** — store, ranking, and
  interface all differ between `baseline` and `treatment`. `treatment-searchtool`
  exists to pin the interface; no README figure uses it.
- **Retrieval is not more relevant, it is more compact.** Offline aspect
  coverage is 89% (incident) and 88% (vecdb) against baseline's 100%. The honest
  claim is payload: 753 chars against 2,051 for 8 points less coverage, ~2.4×
  the relevant content per character.
- **On gemma4:12b incident, treatment trails baseline 5.67 to 7.00 at N=3.**

The genuinely defensible results are different and mostly stronger than what is
claimed: vecdb accuracy above baseline (5.33 vs 5.00) at −37% input; consistent
token savings across both scenarios; and `buried_metric_recalled` 3/3 with tool
capture on, where the flat store is structurally 0/3 because it never stored the
figure at all.

## 5. `qwen3.6` is not installed

[development-loop.md](development-loop.md) specifies tier 2 on `gemma4:12b` and
`qwen3.6`. The registry has `qwen3.6:latest`; nothing is pulled locally, and the
repo's previously documented set was `qwen3.5:9b`.

```bash
scripts/setup_local.sh gemma4:12b qwen3.6
```

Run the smoke gate and believe it. `glm4:9b` advertises `tools` in `/api/show`,
its template handles them, and it emitted zero tool calls across two gate runs —
which would have produced an empty treatment arm reading as a memory-layer
failure.

---

## Decided, not open

**Tool-result capture ships off by default** (`treatment-toolcapture` keeps it
measurable). It buys `buried_metric_recalled` 0/3 → 3/3 — a capability the flat
store cannot match — at the cost of `breadth_complete` 1/3 → 0/3, a store that
doubles (94 → 179 facts), and input tokens going from 35% below baseline to 13%
below. That is most of the library's token advantage for one metric, so it is a
per-deployment choice.

**Interface parity ships off by default** (`treatment-searchtool`). Worth +0.33
metrics on incident and 0.00 on vecdb — inside noise on the means. It moves
individual metrics hard in both directions though: on vecdb it took
`uses_revised_budget` 3/3 → 1/3 while taking `mentions_primary_sources` 1/3 →
3/3. Auto-injection guarantees the corrected constraint is present whether or
not the agent thinks to ask; a tool call retrieves what the agent asks for. Keep
both, default to injection.
