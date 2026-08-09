"""Blinded rubric judge: labels stripped, order randomized."""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys

from langchain_core.messages import HumanMessage

from demo.llm import make_llm

RUBRIC = """Score each transcript excerpt 1-5 on each dimension. Respond with
ONLY a JSON object: {"X": {"recall": n, "delegation": n, "final": n},
"Y": {"recall": n, "delegation": n, "final": n}}.

Dimensions:
- recall: does the conversation-3 answer correctly reflect the constraints
  stated in conversation 1 (self-hostable; only primary benchmark data) and
  use the REVISED budget from the correction, without being re-told any of it?
- delegation: does the synthesis correctly incorporate every researcher's
  findings? Are any findings lost, contradicted, or double-counted?
- final: is the final recommendation substantively correct and well-supported
  given the constraints and the research?
"""

_RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed as JSON. Respond with "
    "ONLY the JSON object described above - no prose, no markdown fences, "
    "no <think> block."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

_DIMENSIONS = ("recall", "delegation", "final")


def blind_pair(rec_a: dict, rec_b: dict) -> tuple[dict, dict]:
    """Strip condition labels; randomize which is X and which is Y."""
    pair = [rec_a, rec_b]
    if random.random() >= 0.5:
        pair.reverse()
    blinded = {
        "X": {k: v for k, v in pair[0].items() if k != "condition"},
        "Y": {k: v for k, v in pair[1].items() if k != "condition"},
    }
    mapping = {"X": pair[0]["condition"], "Y": pair[1]["condition"]}
    return blinded, mapping


def _excerpt(record: dict) -> str:
    convs = {}
    for t in record["transcript"]:
        convs.setdefault(t["conversation"], []).append(f"USER: {t['user']}\nAGENT: {t['final']}")
    return "\n\n".join(f"[{c}]\n" + "\n".join(v) for c, v in convs.items())


def _parse_scores(text: str) -> dict | None:
    """Best-effort parse of a scores JSON object out of a raw LLM response.

    Returns None (never raises) if the text does not contain a well-formed
    scores object - callers decide whether to retry or give up."""
    cleaned = _THINK_BLOCK.sub("", text).strip()
    try:
        start = cleaned.index("{")
        end = cleaned.rindex("}") + 1
    except ValueError:
        return None
    try:
        scores = json.loads(cleaned[start:end])
    except json.JSONDecodeError:
        return None
    if not isinstance(scores, dict) or set(scores.keys()) < {"X", "Y"}:
        return None
    for label in ("X", "Y"):
        dims = scores.get(label)
        if not isinstance(dims, dict) or not all(d in dims for d in _DIMENSIONS):
            return None
    return scores


def judge_pair(rec_a: dict, rec_b: dict) -> dict | None:
    """Score a blinded pair. Returns {condition: {dim: score}} on success, or
    None if the judge model never produced parseable JSON (never fabricated)."""
    blinded, mapping = blind_pair(
        {"condition": rec_a["condition"], "transcript": rec_a["transcript"]},
        {"condition": rec_b["condition"], "transcript": rec_b["transcript"]},
    )
    prompt = (
        RUBRIC
        + "\n\n=== Transcript X ===\n"
        + _excerpt(blinded["X"])
        + "\n\n=== Transcript Y ===\n"
        + _excerpt(blinded["Y"])
    )
    llm = make_llm()
    resp = llm.invoke([HumanMessage(prompt)])
    scores = _parse_scores(str(resp.content))
    if scores is None:
        # One retry with a stricter re-prompt before giving up.
        resp = llm.invoke([HumanMessage(prompt + _RETRY_SUFFIX)])
        scores = _parse_scores(str(resp.content))
    if scores is None:
        return None
    return {mapping[label]: scores[label] for label in ("X", "Y")}


def main(raw_dir: str):
    raw = pathlib.Path(raw_dir)
    records = [json.loads(p.read_text()) for p in sorted(raw.glob("*.json"))]
    by_run: dict[int, dict[str, dict]] = {}
    for r in records:
        by_run.setdefault(r["run"], {})[r["condition"]] = r
    all_scores = []
    unjudgeable = 0
    for run, conds in sorted(by_run.items()):
        if {"baseline", "treatment"} <= conds.keys():
            s = judge_pair(conds["baseline"], conds["treatment"])
            if s is None:
                unjudgeable += 1
                print(f"run {run}: UNJUDGEABLE (judge did not return parseable JSON)", flush=True)
                continue
            print(f"run {run}: {s}", flush=True)
            all_scores.append(s)
    for cond in ("baseline", "treatment"):
        for dim in _DIMENSIONS:
            vals = [s[cond][dim] for s in all_scores]
            if vals:
                print(f"mean {cond}.{dim}: {sum(vals)/len(vals):.2f}")
    print(f"judged runs: {len(all_scores)}, unjudgeable: {unjudgeable}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/raw")
