# Working in this repo

## Follow the development loop

**[docs/development-loop.md](docs/development-loop.md) governs every change to
ranking or capture.** Read it before proposing one. In short:

- **Tier 0** — a new *concept* (capture point, edge relation, weighting scheme,
  fact kind, scope rule) is discussed with the human **before** implementation.
  Tuning an existing knob is not a new concept.
- **Tier 1** — iterate against frozen fixtures with `demo.retrieval_eval`.
  ~1 second, no LLM. At least two fixtures, one per scenario.
- **Tier 2** — N=1, two models, both scenarios. A filter that kills bad changes,
  never confirmation.
- **Tier 3** — N=3+. Report every run, not a mean hiding a range.

## Things that will bite you

**Never edit source while a matrix is running.** Each run is a fresh process, so
a mid-matrix edit puts two codebases in one N=3 arm.

**`uv` lives at `~/.local/bin/uv`** and is not on the default PATH. Export it.

**Postgres is the docker-compose container on :5433**, not a host install.
`docker compose up -d`, and wait on `pg_isready` rather than on "Started".

**Ollama wedges under sustained load.** The matrix scripts restart it between
runs via `brew services`; that only works if brew owns the process.

**Run the smoke gate before benchmarking a new model.** `glm4:9b` advertises
`tools` and still emits none, which produces an empty treatment arm that reads
as a memory-layer failure.

## Reporting results

Never pool across scenarios or models — the library beats baseline on vecdb and
trails on incident, and a pooled number reports neither. `aggregate.py` and
`compare_models.py` both refuse to pool; don't work around them.

Distinguish a **capture** failure (no item in the store covers the aspect — no
ranking change can help) from a **ranking** failure. Check the mechanism in the
transcript, not just the metric: a run once scored 7/8 from a
`recall_from_subagents` call while the injected block still lacked the answer.

## Style

Comments carry the *measured reason* a filter exists, not a description of what
the line does. That is what stops a disproved idea being re-implemented. Keep
corrections in place when a previous claim turned out wrong — several comments
document exactly that, and they are load-bearing.
