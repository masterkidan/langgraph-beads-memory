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

A single scripted narrative, run twice (once per condition, §3), three
LangGraph sessions (`session_id`s), driven by a fixed script (not live human
input, so both runs are byte-identical in what the "user" says):

1. **Session 1** — user states a research goal plus specific constraints/
   preferences (e.g. "I only trust primary sources," "I'm evaluating this
   for a cost-sensitive audience"). Agent responds; session ends.
2. **Session 2** (new thread, same scenario) — user asks the agent to
   investigate the goal in depth. Agent delegates to 2-3 parallel sub-agents,
   each assigned a distinct sub-topic. Sub-agents conclude and roll up
   summaries to the parent. Agent synthesizes a session-2 answer; session
   ends.
3. **Session 3** (new thread, later) — user asks a follow-up that can only
   be answered well by recalling **both** session 1's constraints/
   preferences **and** session 2's rolled-up conclusions, without repeating
   either in the session-3 prompt.

## 3. Conditions compared

- **Baseline**: LangGraph `PostgresSaver` (checkpointer, thread-scoped) +
  `PostgresStore` + LangMem for cross-thread memory. Configured as a team
  would reasonably set it up — not a strawman.
- **Treatment**: beads-memory, demo-scoped subset (§4).

Both conditions use the same local LLM, same scripted user turns, same
sub-agent task assignments — the only variable is the memory layer.

## 4. Demo-scoped beads-memory subset

From the full design, implemented for the demo:

- Postgres schema: `namespaces`, `facts`, `fact_edges` (session_id-only, per
  the 2026-08-08 amendment).
- Passive user-input capture, `remember_fact` tool, `conclude_task` tool with
  enforced fallback.
- Fork/rollup model for sub-agents (§5.1 of the core design) — this is
  exactly what the session-2 delegation scenario exercises.
- Semantic search retrieval (pgvector) + ancestor read-through.
- Idempotent writes (content-derived fact ids).

Deferred (not needed at this scale, revisit post-demo):
- Compaction (window-overflow and threshold-based) — three short sessions
  won't approach the fact counts that make compaction matter.
- Async embedding worker — embed **synchronously** in the demo; latency
  doesn't matter for a scripted local run, and it removes a moving part.

## 5. Stack

- **LLM**: local, via Ollama. Same model for both conditions' agents and for
  the judge, to keep it a memory-layer comparison, not a model comparison.
  Model choice needs a tool-calling-capable model (`remember_fact` /
  `conclude_task` are tool calls) — pick at implementation time based on
  what's actually reliable in testing (candidates: `qwen2.5`, `llama3.1`);
  swap freely if the first choice hallucinates tool args.
- **Postgres**: local via Docker, with `pgvector` extension.
- **Framework**: LangGraph (`create_agent` + middleware), Python.

## 6. Measurement

**Transcripts**: full message + tool-call logs captured for both conditions,
all three sessions, saved as structured (JSON) and human-readable output.

**LLM judge**: a separate judging pass (same local model) scores each
condition's session-3 answer against a fixed rubric, 1-5 per dimension:

- **Recall accuracy** — did the answer correctly reflect session 1's stated
  constraints/preferences without being re-told them?
- **Delegation quality** — (scored from session 2 transcripts) did
  sub-agents receive focused, non-overlapping tasks, and did the parent's
  synthesis reflect all sub-agent conclusions without raw exploration noise
  leaking into the parent's context?
- **Final answer quality** — is the session-3 answer substantively correct
  and well-supported given everything the agent should know by that point?

**Qualitative**: alongside the scores, call out specific divergence moments
in the transcripts (e.g. "baseline agent re-asks a constraint already stated
in session 1"; "beads-memory agent's session-3 answer cites the session-2 rollup
fact directly").

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

## 8. Non-goals for this phase

- Publishing to Medium or making any repo public — needs explicit sign-off
  when we're actually ready to publish, handled separately.
- Production concerns from the core design (compaction, async embeddings,
  fail-open error handling under real load) — demo only needs to run
  correctly once, locally, deterministically.
