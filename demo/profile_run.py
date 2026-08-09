"""Run one scenario under the profiler and print a timing breakdown.

Usage:
    uv run python -m demo.profile_run                    # treatment, full scenario
    uv run python -m demo.profile_run --condition baseline
    uv run python -m demo.profile_run --conversations conv-1
    uv run python -m demo.profile_run --model qwen3:4b

Writes the raw span list to results/profile-<condition>-<stamp>.json so a slow
run can be re-analysed without paying for it twice.
"""

from __future__ import annotations

import argparse
import os
import pathlib
import time

RESULTS = pathlib.Path(__file__).parent.parent / "results"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", default="treatment", choices=["treatment", "baseline"])
    ap.add_argument(
        "--conversations",
        nargs="*",
        default=None,
        help="subset of conversation ids (default: all)",
    )
    ap.add_argument("--model", default=None, help="override BEADS_DEMO_MODEL")
    args = ap.parse_args()

    # Must be set before demo.llm is imported, since MODEL is read at import time.
    if args.model:
        os.environ["BEADS_DEMO_MODEL"] = args.model

    from demo import conditions
    from demo.llm import MODEL
    from demo.profiler import RunProfiler, TimedEmbedder
    from demo.scenario import CONVERSATIONS

    profiler = RunProfiler()

    # Attribute embedding and Postgres time instead of leaving them as
    # "unaccounted": patch the classes the condition builder will instantiate.
    import beads_memory

    original_embedder_cls = beads_memory.OllamaEmbedder
    original_connect = conditions.psycopg.connect

    def embedder_factory(*a, **kw):
        return TimedEmbedder(original_embedder_cls(*a, **kw), profiler)

    def connect_factory(*a, **kw):
        # Instrument execute() in place rather than wrapping the connection:
        # pgvector's register_vector does a strict isinstance check and rejects
        # a proxy object.
        conn = original_connect(*a, **kw)
        inner_execute = conn.execute

        def timed_execute(*ea, **ekw):
            with profiler.span("db", "query"):
                return inner_execute(*ea, **ekw)

        conn.execute = timed_execute
        return conn

    beads_memory.OllamaEmbedder = embedder_factory
    conditions.OllamaEmbedder = embedder_factory
    conditions.psycopg.connect = connect_factory

    stamp = time.strftime("%Y%m%d-%H%M%S")
    session_id = f"profile-{args.condition}-{stamp}"
    build = (
        conditions.build_treatment if args.condition == "treatment" else conditions.build_baseline
    )
    invoke, cleanup = build(session_id, f"profile_{args.condition}_{stamp}".replace("-", "_"))

    wanted = set(args.conversations) if args.conversations else None
    started = time.monotonic()
    try:
        for conv_id, turns in CONVERSATIONS:
            if wanted and conv_id not in wanted:
                continue
            for i, user_text in enumerate(turns):
                profiler.phase = f"{conv_id}.t{i}"
                turn_start = time.monotonic()
                print(f"  {profiler.phase}: {user_text[:56]!r}...", flush=True)
                # The profiler rides on the model/tool callbacks for this call.
                invoke(f"{session_id}-{conv_id}", user_text, callbacks=[profiler])
                print(f"    -> {time.monotonic() - turn_start:.0f}s", flush=True)
    finally:
        cleanup()
        beads_memory.OllamaEmbedder = original_embedder_cls
        conditions.psycopg.connect = original_connect

    wall = time.monotonic() - started
    print()
    print(f"model: {MODEL}   condition: {args.condition}")
    print(profiler.report(wall))

    RESULTS.mkdir(exist_ok=True)
    out = RESULTS / f"profile-{args.condition}-{stamp}.json"
    out.write_text(profiler.to_json())
    print(f"\nspans -> {out}")


if __name__ == "__main__":
    main()
