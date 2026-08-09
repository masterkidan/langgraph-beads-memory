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

**One scenario change, made after seeing results (run 0):** the buried-detail
question originally asked about "the strongest runner-up" without naming it. Its
referent depends on which database the agent picked, so it measured *choice*
rather than *recall* — in run 0 the treatment picked Qdrant, read "runner-up" as
Weaviate, and hallucinated an optimization for it, while the baseline scored the
point only because it declined to pick anything. The question now names Qdrant.
Rationale is recorded in `demo/scenario.py`.

Because of that change, **run 0 is archived rather than pooled** — it answers a
different question and cannot be averaged with later runs. It is kept in
`archive/` because it is the evidence behind all four corrections above.

`demo/aggregate.py` rescores every transcript with current metric code rather
than trusting the `constraint_carry` snapshot stored in each JSON, so runs
recorded before a metric fix are never silently pooled with runs recorded after
one.

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
  resolved in either direction.
- **A confound to keep checking.** `read_document` stays available in
  conversation 3, so an agent could bypass memory by re-reading the corpus. In
  run 0 neither condition did (the baseline used `search_memory`), but this
  should be verified per run rather than assumed.
