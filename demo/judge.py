"""Blinded rubric judge: labels stripped, order randomized."""

from __future__ import annotations

import json
import pathlib
import random
import re
import sys

from langchain_core.messages import HumanMessage

from demo.llm import make_llm

_HEADER = """Score each transcript excerpt 1-5 on each dimension. Respond with
ONLY a JSON object: {"X": {"recall": n, "delegation": n, "final": n,
"grounding": n}, "Y": {"recall": n, "delegation": n, "final": n,
"grounding": n}}.

"""

# Applies to every scenario. Added after a measured failure: in the 2026-08-11
# round the judge gave `recall: 5` to an answer that named the right technique
# with a fabricated magnitude ("up to 50%" where the source said 32x). Nothing
# in the rubric asked it to check whether a number was real, so nothing did.
_GROUNDING_DIMENSION = """- grounding: are the specific figures in the answers actually supported by
  what the agents were told or read? Score 1 if any confident-sounding number
  appears that is not traceable to the conversation or the documents, even if
  everything else is correct. A vague answer with no numbers scores 3; an
  answer whose every figure checks out scores 5. Being wrong about a number is
  worse than omitting it.
"""

RUBRICS = {
    "vecdb": _HEADER
    + """Dimensions:
- recall: does the conversation-3 answer correctly reflect the constraints
  stated in conversation 1 (self-hostable; only primary benchmark data) and
  use the REVISED budget from the correction, without being re-told any of it?
- delegation: does the synthesis correctly incorporate every researcher's
  findings? Are any findings lost, contradicted, or double-counted?
- final: is the final recommendation substantively correct and well-supported
  given the constraints and the research?
"""
    + _GROUNDING_DIMENSION,
    "incident": _HEADER
    + """This is a production incident investigation.

Dimensions:
- recall: does the later answer correctly reflect what was already ELIMINATED
  (and why), and use the CORRECTED deploy timestamp rather than the original
  one, without being re-told any of it? Putting an already-eliminated cause
  back on the to-do list is the worst failure here and should score 1.
- delegation: does the synthesis correctly incorporate all three subsystem
  investigations? Are any findings lost, contradicted, or double-counted?
- final: does the recommended next step identify the actual cause and respect
  the stated constraints (reversible within 30 minutes, no full outage)?
"""
    + _GROUNDING_DIMENSION,
}

# Kept for callers that import RUBRIC directly.
RUBRIC = RUBRICS["vecdb"]

_RETRY_SUFFIX = (
    "\n\nYour previous response could not be parsed as JSON. Respond with "
    "ONLY the JSON object described above - no prose, no markdown fences, "
    "no <think> block."
)

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)

_DIMENSIONS = ("recall", "delegation", "final", "grounding")


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
    scenario = rec_a.get("scenario", "vecdb")
    prompt = (
        RUBRICS.get(scenario, RUBRICS["vecdb"])
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
    arms = sorted({r["condition"] for r in records} - {"baseline"})
    for run, conds in sorted(by_run.items()):
        if "baseline" not in conds:
            continue
        for arm in arms:
            if arm not in conds:
                continue
            s = judge_pair(conds["baseline"], conds[arm])
            if s is None:
                unjudgeable += 1
                print(f"run {run} [{arm}]: UNJUDGEABLE (no parseable JSON)", flush=True)
                continue
            print(f"run {run} [{arm}]: {s}", flush=True)
            all_scores.append(s)
    for cond in ["baseline", *arms]:
        for dim in _DIMENSIONS:
            vals = [s[cond][dim] for s in all_scores if cond in s]
            if vals:
                print(f"mean {cond}.{dim}: {sum(vals)/len(vals):.2f}")
    print(f"judged runs: {len(all_scores)}, unjudgeable: {unjudgeable}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/raw")
