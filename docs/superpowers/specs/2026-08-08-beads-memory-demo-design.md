# langgraph-beads-memory demo: proving it beats plain LangGraph memory

Status: Design approved, not yet implemented.
Date: 2026-08-08

Related: [2026-07-31-beads-memory-design.md](2026-07-31-beads-memory-design.md) (core architecture), [2026-07-31-beads-memory-competitive-brief.md](2026-07-31-beads-memory-competitive-brief.md) (market landscape)

## 1. Goal

Build a local, runnable demo that shows beads-memory producing measurably better
outcomes than LangGraph's own memory story (`BaseStore` + LangMem) on three
claims: **better remembrance**, **better task delegation**, **better
sub-agent task outcomes**. The demo output feeds an explainer animation for a
Medium post announcing the project.

This is a demo-first build: implement only the subset of the full design
(`2026-07-31-beads-memory-design.md`) that this scenario actually
exercises. Compaction and the async embedding background worker are deferred
— not needed at this scale (see §4).

## 2. Scenario

*(Terminology fixed 2026-08-08 after critical review: what were called
"sessions" are **conversations** — separate LangGraph threads. All three
share **one** beads-memory `session_id`, because a session is the long-lived
memory scope spanning the user's ongoing work on a topic; recall across
conversations is the whole point. The checkpointer's `thread_id` changes per
conversation; `beads_session_id` in config does not.)*

A single scripted narrative, run twice (once per condition, §3): one
session, three conversations (three `thread_id`s), driven by a fixed script
(not live human input, so both runs are byte-identical in what the "user"
says):

1. **Conversation 1** — user states a research goal plus specific constraints/
   preferences (e.g. "I only trust primary sources," "I'm evaluating this
   for a cost-sensitive audience"). Agent responds; conversation ends.
2. **Conversation 2** (new thread, same session) — user asks the agent to
   investigate the goal in depth. Agent delegates to 2-3 parallel sub-agents,
   each assigned a distinct sub-topic. Sub-agents conclude and roll up
   summaries to the parent. Agent synthesizes an answer; conversation ends.
3. **Conversation 3** (new thread, later) — user asks a follow-up that can
   only be answered well by recalling **both** conversation 1's constraints/
   preferences **and** conversation 2's rolled-up conclusions, without
   repeating either in the conversation-3 prompt.

## 3. Conditions compared

- **Baseline**: LangGraph `PostgresSaver` (checkpointer, thread-scoped) +
  `PostgresStore` + LangMem for cross-thread memory. Configured as a team
  would reasonably set it up — not a strawman. **Delegation wiring**: the
  idiomatic `langgraph-supervisor` pattern — sub-agents receive tasks via
  handoff, their results return as messages into the parent's history, and
  the LangMem store is shared across all agents. This is what a real team
  would build from the docs; it is the fair "best effort with what ships
  today" comparison point.
- **Treatment**: beads-memory, demo-scoped subset (§4).

Both conditions use the same local LLM, same scripted user turns, same
sub-agent task assignments, and roughly comparable injected-context token
budgets — the only variable is the memory layer.

## 4. Demo-scoped beads-memory subset

From the full design, implemented for the demo:

- Postgres schema: `namespaces`, `facts`, `fact_edges` (session_id-only, per
  the 2026-08-08 amendment).
- Passive user-input capture, `remember_fact` tool, `conclude_task` tool with
  enforced fallback.
- Fork/rollup model for sub-agents (§5.1 of the core design) — this is
  exactly what the conversation-2 delegation scenario exercises.
- Semantic search retrieval (pgvector) + ancestor read-through, with short
  fact ids rendered in injected context and window-dedup (facts whose source
  message is still in the raw window are not re-injected).
- Idempotent writes (content-derived fact ids).

Deferred (not needed at this scale, revisit post-demo):
- Compaction (window-overflow and threshold-based) — three short
  conversations won't approach the fact counts that make compaction matter.
- Async embedding worker — embed **synchronously** in the demo; latency
  doesn't matter for a scripted local run, and it removes a moving part.

## 5. Stack

- **LLM**: local, via Ollama. Same model for both conditions' agents and for
  the judge, to keep it a memory-layer comparison, not a model comparison.
  Model choice needs a tool-calling-capable model (`remember_fact` /
  `conclude_task` are tool calls) — pick at implementation time based on
  what's actually reliable in testing (candidates: `qwen2.5`, `llama3.1`);
  swap freely if the first choice hallucinates tool args.
- **Pre-flight smoke test** (required before any scored run): verify the
  chosen model can (a) perform LangMem's extraction acceptably and (b) make
  beads-memory's tool calls reliably. A weak local model can silently gut
  *either* condition — LangMem extraction collapsing makes the baseline an
  accidental strawman; an agent that never calls `remember_fact` guts the
  treatment. Neither failure mode is a legitimate result; both mean "pick a
  better model."
- **Embeddings**: `nomic-embed-text` via Ollama (768-d) for pgvector.
- **Postgres**: local via Docker, with `pgvector` extension.
- **Framework**: LangGraph (`create_agent` + middleware), Python.

## 6. Measurement

**Runs**: the full scenario runs **N ≥ 3 times per condition** (cheap on a
local model); report per-dimension means, not a single run. This absorbs LLM
nondeterminism and makes one lucky/unlucky run non-decisive.

**Transcripts**: full message + tool-call logs captured for both conditions,
all three conversations, every run, saved as structured (JSON) and
human-readable output.

**LLM judge — blinded**: a separate judging pass (same local model) scores
each condition's conversation-3 answer against a fixed rubric, 1-5 per
dimension. Condition labels are stripped and presentation order randomized
before judging — the judge must not know which transcript is which.

The rubric is deliberately **outcome-focused**, not mechanism-mirroring
(scoring "no noise leaked into the parent" would just restate the
treatment's feature list and invite a rigged-rubric objection):

- **Recall accuracy** — does the conversation-3 answer correctly reflect
  conversation 1's stated constraints/preferences without being re-told them?
- **Delegation outcome quality** — from conversation-2 and -3 output: does
  the final synthesis correctly incorporate *every* sub-agent's findings?
  Are any results lost, contradicted, or double-counted?
- **Final answer quality** — is the conversation-3 answer substantively
  correct and well-supported given everything the agent should know?

**Objective metrics** (no judge involved): total tokens consumed per
condition per run (prompt + completion, agents + sub-agents), and count of
scripted constraints correctly carried into the final answer. Token
accounting doubles as a fairness check — if one condition consumes wildly
more context, that gets reported, not hidden.

**Qualitative**: alongside the scores, call out specific divergence moments
in the transcripts (e.g. "baseline agent re-asks a constraint already stated
in conversation 1"; "beads-memory agent's conversation-3 answer cites the
conversation-2 rollup fact directly").

## 7. Deliverables

1. `langgraph-beads-memory` package (demo-scoped subset, §4) — installable locally.
2. Comparison harness: runs the scripted scenario against both conditions,
   produces transcripts + judge scores.
3. A results write-up (scores + annotated transcript excerpts) — the
   evidence base for the Medium post.
4. Explainer animation: an animated SVG/HTML asset (built as an Artifact,
   captured to GIF/video) showing the *mechanism* — namespace forking, facts
   flowing into the typed graph, sub-agent rollup — side by side with the
   baseline's context silently getting overwritten/forgotten. Explains why
   the mechanism produces the measured results; independent of any one run's
   transcript.

## 8. Publication stance

The scenario is a **designed demonstration, not a neutral benchmark** — and
the Medium post will say so plainly. If early runs come out mixed, we
iterate the scenario until it cleanly exercises the mechanism's advantage
(that is what a demonstration is *for*), and we disclose that the scenario
was constructed to showcase the mechanism. What we do **not** do: hide
unfavorable metrics from the runs we publish (token counts and all rubric
dimensions get reported as measured), or present the demo as an independent
benchmark. The blinded judge and N-run means (§6) keep the numbers honest
*within* the designed scenario.

## 9. Non-goals for this phase

- Publishing to Medium or making any repo public — needs explicit sign-off
  when we're actually ready to publish, handled separately.
- Production concerns from the core design (compaction, async embeddings,
  fail-open error handling under real load) — the demo only needs to run
  correctly and repeatably on one local machine.
