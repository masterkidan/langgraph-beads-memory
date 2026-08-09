"""Runs the full scenario N times per condition; saves transcripts + metrics."""

from __future__ import annotations

import argparse
import json
import pathlib
import time
import traceback
import uuid

from langchain_core.messages import message_to_dict

from demo import metrics
from demo.conditions import build_baseline, build_treatment
from demo.scenario import CONVERSATIONS

RAW = pathlib.Path(__file__).parent.parent / "results" / "raw"


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
                    result = invoke(thread_id, user_text)
                    msgs = result["messages"]
                except Exception as e:  # noqa: BLE001 - a crashed turn must be recorded, not fatal
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
