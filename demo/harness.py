"""Runs the full scenario N times per condition; saves transcripts + metrics."""

from __future__ import annotations

import argparse
import contextlib
import faulthandler
import json
import os
import pathlib
import signal
import time
import traceback
import uuid

from langchain_core.messages import message_to_dict

from demo import metrics
from demo.conditions import DSN, build
from demo.llm import MODEL, close_llms

RAW = pathlib.Path(__file__).parent.parent / "results" / "raw"


# Hard wall-clock ceiling for one scripted turn. A per-read HTTP timeout is not
# enough: it resets on every byte, so a response that trickles never trips it —
# a real run hung 8.5 minutes with a 300s read timeout configured, blocked in
# sock_recv while Ollama had already unloaded the model. SIGALRM interrupts the
# blocking syscall itself, which is the only thing that reliably breaks that.
# The harness already records a failed turn and continues, so a fired deadline
# costs one turn; no deadline costs the whole run.
TURN_DEADLINE_S = int(os.environ.get("BEADS_DEMO_TURN_DEADLINE", "900"))


class TurnTimeout(BaseException):
    """Deliberately a BaseException, not an Exception.

    The deadline is raised from a signal handler, so it surfaces at whatever
    line happens to be executing — deep inside LangGraph, a tool, or httpx. This
    codebase has broad `except Exception` handlers on purpose (`_crash_safe`
    around langmem's tools, and the sub-agent wrapper, which must not let a
    crashed researcher kill the run). Any of them would have caught a plain
    Exception and quietly resumed waiting, defeating the deadline entirely.
    Inheriting from BaseException makes it pass through them, like
    KeyboardInterrupt.
    """


# Fires after the SIGALRM deadline, so the in-process path gets first refusal
# and only a genuinely unbreakable hang reaches the hard dump-and-exit.
DEADLOCK_DUMP_S = int(os.environ.get("BEADS_DEMO_DEADLOCK_DUMP", str(TURN_DEADLINE_S + 120)))


def arm_deadlock_dump(seconds: int) -> None:
    """Dump every thread's Python stack and exit if a turn outlives `seconds`.

    A C-level watchdog thread, so it works when the interpreter itself is stuck.
    exit=True is deliberate: a deadlocked run cannot be salvaged in-process, and
    the driver loop starts the next one.
    """
    faulthandler.enable()
    faulthandler.dump_traceback_later(seconds, repeat=False, exit=True)


def disarm_deadlock_dump() -> None:
    faulthandler.cancel_dump_traceback_later()


@contextlib.contextmanager
def turn_deadline(seconds: int):
    """Abort the enclosing block if it outlasts `seconds`.

    Main-thread only (SIGALRM's restriction); the harness drives turns from the
    main thread. Falls back to no-op where signals are unavailable rather than
    refusing to run.
    """
    if seconds <= 0 or not hasattr(signal, "SIGALRM"):
        yield
        return

    def _fire(signum, frame):
        raise TurnTimeout(f"turn exceeded {seconds}s deadline")

    previous = signal.signal(signal.SIGALRM, _fire)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


def snapshot_memory(condition: str, schema: str, session_id: str) -> dict:
    """What the run actually stored, captured with the run.

    Queried here rather than after the fact because schemas are reused across
    runs of the same index — an earlier ad-hoc analysis ranked across three
    runs' facts at once and reported a rank that did not exist. A snapshot
    taken while the run owns the schema cannot drift.

    Both arms are described in their own terms: the treatment's typed facts by
    kind/source, the baseline's LangMem documents by count and size.
    """
    import psycopg

    out: dict = {"arm": condition}
    try:
        conn = psycopg.connect(DSN, autocommit=True)
        if condition == "baseline":
            rows = conn.execute(
                "SELECT count(*), coalesce(sum(length(value::text)), 0)"
                " FROM public.store WHERE prefix LIKE %s",
                (f"%{session_id}%",),
            ).fetchone()
            out["documents"], out["chars"] = rows[0], rows[1]
        else:
            conn.execute(f'SET search_path TO "{schema}", public')
            rows = conn.execute(
                "SELECT kind, source, status, count(*), sum(length(body))"
                " FROM facts WHERE session_id = %s GROUP BY 1,2,3",
                (session_id,),
            ).fetchall()
            out["facts"] = [
                {"kind": k, "source": src, "status": st, "n": n, "chars": c}
                for k, src, st, n, c in rows
            ]
            out["total_facts"] = sum(r["n"] for r in out["facts"])
            out["total_chars"] = sum(r["chars"] for r in out["facts"])
        conn.close()
    except Exception as e:  # noqa: BLE001 - a snapshot must never fail a run
        out["error"] = f"{type(e).__name__}: {e}"
    return out


def run_once(condition: str, run_idx: int, scenario_name: str = "vecdb") -> dict:
    session_id = f"{condition}-run{run_idx}-{uuid.uuid4().hex[:6]}"
    schema = f"run_{condition.replace('-', '_')}_{scenario_name}_{run_idx}"
    (invoke, cleanup), scenario = build(condition, scenario_name, session_id, schema)
    transcript, all_msgs, errors = [], [], []
    # Wall-clock timing: the run is inference-bound, so per-turn duration is how
    # we tell whether an execution change (e.g. running sub-agents concurrently)
    # actually helped, rather than assuming it did.
    run_started = time.monotonic()
    try:
        for conv_id, turns in scenario.conversations:
            thread_id = f"{session_id}-{conv_id}"
            for user_text in turns:
                print(
                    f"  [{condition} run {run_idx}] {conv_id}: {user_text[:60]!r}...",
                    flush=True,
                )
                turn_started = time.monotonic()
                try:
                    # Belt and braces. turn_deadline uses SIGALRM, which cannot
                    # interrupt a thread parked in lock_PyThread_acquire_lock —
                    # Python only runs signal handlers between bytecodes, and a
                    # blocking lock acquire never yields. Every observed hang had
                    # that exact shape, so SIGALRM alone has never fired on one.
                    # faulthandler's watchdog is a C thread: it fires regardless
                    # of GIL or lock state, dumps every thread's Python stack,
                    # and exits. The dump is the point — each earlier hang cost a
                    # diagnosis because the process had to be killed blind.
                    arm_deadlock_dump(DEADLOCK_DUMP_S)
                    turn_recorder: list = []
                    with turn_deadline(TURN_DEADLINE_S):
                        result = invoke(thread_id, user_text, recorder=turn_recorder)
                    disarm_deadlock_dump()
                    msgs = result["messages"]
                except (Exception, TurnTimeout) as e:  # noqa: BLE001 - record, do not abort the run
                    disarm_deadlock_dump()
                    errors.append(
                        {
                            "conversation": conv_id,
                            "user": user_text,
                            "error": f"{type(e).__name__}: {e}",
                            "traceback": traceback.format_exc(),
                        }
                    )
                    transcript.append(
                        {
                            "conversation": conv_id,
                            "user": user_text,
                            "messages": [],
                            "final": "",
                            "errored": True,
                        }
                    )
                    print(f"    -> ERROR: {type(e).__name__}: {e}", flush=True)
                    continue
                turn_seconds = time.monotonic() - turn_started
                all_msgs.extend(msgs)
                final = str(msgs[-1].content) if msgs else ""
                # An empty final answer is NOT the same as a bad one, but it
                # scores identically — zero on every metric — so it has to be
                # visible rather than inferred from a row of failures. Observed
                # for real: a delegation turn where the model returned an
                # AIMessage with `done: True`, no tool calls, and no content.
                # Recorded, not repaired: substituting earlier assistant text
                # would fabricate an answer the agent never gave and then score
                # it. (The sub-agent path is different and does fall back — see
                # `_subagent_output` — because a sub-agent's earlier prose is a
                # real report to its supervisor, not a scored answer.)
                transcript.append(
                    {
                        "conversation": conv_id,
                        "user": user_text,
                        "messages": [message_to_dict(m) for m in msgs],
                        "final": final,
                        "errored": False,
                        "empty_final": not final.strip(),
                        # What retrieval actually put in front of the model this
                        # turn, with distances and whether the descendant penalty
                        # applied. Recorded rather than reconstructable: replaying
                        # a query later runs it against a store that has since
                        # changed, which produced one confidently wrong ranking
                        # analysis before this existed.
                        "injections": turn_recorder,
                        "seconds": round(turn_seconds, 1),
                    }
                )
                print(
                    f"    -> [{turn_seconds:.0f}s] "
                    + ("<EMPTY final answer>" if not final.strip() else repr(final[:70])),
                    flush=True,
                )
    finally:
        cleanup()
        # Release this run's Ollama sockets now. The clients are cached for the
        # life of the process, so without this a multi-run process accumulates
        # pools across runs and the next run inherits the previous one's mess.
        close_llms()

    return {
        "condition": condition,
        "scenario": scenario.name,
        # Recorded per run. Without it, transcripts from different models are
        # indistinguishable in results/raw, and a mixed directory would be
        # pooled silently — the same failure mode the scenario guard prevents.
        "model": MODEL,
        "run": run_idx,
        "session_id": session_id,
        "transcript": transcript,
        "errors": errors,
        "tokens": metrics.token_usage(all_msgs),
        # Key stays `constraint_carry` so aggregate/judge keep working across
        # both scenarios; the scenario decides what the dict contains.
        "constraint_carry": scenario.score(transcript),
        "memory": snapshot_memory(condition, schema, session_id),
        "seconds": round(time.monotonic() - run_started, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--conditions", nargs="+", default=["baseline", "treatment"])
    ap.add_argument("--scenario", default="vecdb", help="vecdb | incident")
    ap.add_argument(
        "--only",
        default=None,
        metavar="CONDITION:INDEX",
        help="run exactly one run, e.g. 'treatment:2'. Lets a driver script bound "
        "each run in its own process and restart Ollama between them, so memory "
        "pressure cannot accumulate across a six-run set. The run index is "
        "preserved in the filename so the judge can still pair runs correctly.",
    )
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")

    if args.only:
        condition, _, idx = args.only.partition(":")
        plan = [(condition.strip(), int(idx))]
    else:
        plan = [(c, i) for c in args.conditions for i in range(args.runs)]

    for condition, i in plan:
        print(f"=== {args.scenario} / {condition} run {i} ===", flush=True)
        record = run_once(condition, i, args.scenario)
        out = RAW / f"{stamp}-{args.scenario}-{condition}-{i}.json"
        out.write_text(json.dumps(record, indent=2, default=str))
        print(
            f"  tokens={record['tokens']}  carry={record['constraint_carry']}"
            f"  errors={len(record['errors'])}",
            flush=True,
        )


if __name__ == "__main__":
    main()
