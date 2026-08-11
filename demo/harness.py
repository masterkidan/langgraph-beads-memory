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
from demo.conditions import build_baseline, build_treatment
from demo.scenario import CONVERSATIONS

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


def run_once(condition: str, run_idx: int) -> dict:
    session_id = f"{condition}-run{run_idx}-{uuid.uuid4().hex[:6]}"
    build = build_treatment if condition == "treatment" else build_baseline
    invoke, cleanup = build(session_id, f"run_{condition}_{run_idx}")
    transcript, all_msgs, errors = [], [], []
    # Wall-clock timing: the run is inference-bound, so per-turn duration is how
    # we tell whether an execution change (e.g. running sub-agents concurrently)
    # actually helped, rather than assuming it did.
    run_started = time.monotonic()
    try:
        for conv_id, turns in CONVERSATIONS:
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
                    with turn_deadline(TURN_DEADLINE_S):
                        result = invoke(thread_id, user_text)
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
                transcript.append(
                    {
                        "conversation": conv_id,
                        "user": user_text,
                        "messages": [message_to_dict(m) for m in msgs],
                        "final": str(msgs[-1].content),
                        "errored": False,
                        "seconds": round(turn_seconds, 1),
                    }
                )
                print(
                    f"    -> [{turn_seconds:.0f}s] {str(msgs[-1].content)[:70]!r}",
                    flush=True,
                )
    finally:
        cleanup()

    conv3 = [t for t in transcript if t["conversation"] == "conv-3"]
    final_answer = conv3[0]["final"] if conv3 else ""
    buried_answer = conv3[1]["final"] if len(conv3) > 1 else ""
    return {
        "condition": condition,
        "run": run_idx,
        "session_id": session_id,
        "transcript": transcript,
        "errors": errors,
        "tokens": metrics.token_usage(all_msgs),
        "constraint_carry": metrics.constraint_carry(final_answer, buried_answer),
        "seconds": round(time.monotonic() - run_started, 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--conditions", nargs="+", default=["baseline", "treatment"])
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for condition in args.conditions:
        for i in range(args.runs):
            print(f"=== {condition} run {i} ===", flush=True)
            record = run_once(condition, i)
            out = RAW / f"{stamp}-{condition}-{i}.json"
            out.write_text(json.dumps(record, indent=2, default=str))
            print(
                f"  tokens={record['tokens']}  carry={record['constraint_carry']}"
                f"  errors={len(record['errors'])}",
                flush=True,
            )


if __name__ == "__main__":
    main()
