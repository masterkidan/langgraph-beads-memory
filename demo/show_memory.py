"""Show what a run stored, and what it put in front of the model each turn.

The run JSON now carries both, so "why did it answer that?" is answerable by
reading rather than by re-querying a database that has since moved on. This
prints them.

    uv run python -m demo.show_memory results/fresh-gemma/incident
    uv run python -m demo.show_memory results/fresh-gemma/incident --turn conv-3

Only the treatment has an injection log: its recall is automatic, so there is
no tool call to read. The baseline's recall IS a `search_memory` tool call, and
its result is already in the transcript.
"""

from __future__ import annotations

import argparse
import json
import pathlib


def _runs(path: str) -> list[dict]:
    p = pathlib.Path(path)
    files = sorted(p.rglob("*.json")) if p.is_dir() else [p]
    return [json.loads(f.read_text()) for f in files]


def show_store(rec: dict) -> None:
    mem = rec.get("memory") or {}
    print(f"  STORED  ({mem.get('arm', rec['condition'])})")
    if "documents" in mem:
        print(f"     {mem['documents']} documents, {mem['chars']} chars")
        return
    facts = mem.get("facts") or []
    if not facts:
        print("     (no snapshot — run predates the instrumentation)")
        return
    by_kind: dict[str, list[int]] = {}
    for f in facts:
        agg = by_kind.setdefault(f"{f['kind']}/{f['status']}", [0, 0])
        agg[0] += f["n"]
        agg[1] += f["chars"]
    for key in sorted(by_kind, key=lambda k: -by_kind[k][1]):
        n, chars = by_kind[key]
        print(f"     {key:28s} {n:4d} facts  {chars:6d} chars")
    print(f"     {'TOTAL':28s} {mem['total_facts']:4d} facts  {mem['total_chars']:6d} chars")


def show_injections(rec: dict, only_turn: str | None) -> None:
    for turn in rec["transcript"]:
        if only_turn and turn["conversation"] != only_turn:
            continue
        injections = turn.get("injections") or []
        if not injections:
            continue
        print(f"\n  {turn['conversation']}  {turn['user'][:66]!r}")
        # One entry per model call; the first is the one that shaped the answer
        # most, so show it and summarise the rest rather than dumping every call.
        for i, inj in enumerate(injections[:2]):
            label = "first model call" if i == 0 else f"model call {i + 1}"
            print(f"     [{label}]  k={inj['k']}  excluded={inj['excluded']}")
            for rank, f in enumerate(inj["injected"], 1):
                mark = " (demoted)" if f["demoted"] else ""
                print(
                    f"       {rank}. d={f['distance']:.3f}{mark:10s} "
                    f"({f['kind']}/{f['source']}) {f['body'][:74]}"
                )
        if len(injections) > 2:
            print(f"     ... {len(injections) - 2} further model call(s) this turn")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="a run JSON, or a directory of them")
    ap.add_argument("--turn", default=None, help="only this conversation id")
    ap.add_argument("--no-injections", action="store_true")
    args = ap.parse_args()

    for rec in _runs(args.path):
        print("=" * 78)
        print(f"{rec['condition']}  scenario={rec.get('scenario')}  model={rec.get('model')}")
        show_store(rec)
        if not args.no_injections:
            show_injections(rec, args.turn)
        print()


if __name__ == "__main__":
    main()
