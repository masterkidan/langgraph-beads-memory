"""Pool objective metrics across N runs into the results table.

Rescores every saved transcript with the CURRENT metric code rather than
trusting the `constraint_carry` snapshot stored in each JSON. Runs recorded
before a metric fix would otherwise be pooled with runs recorded after it,
silently mixing two measurement definitions.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from demo.metrics import constraint_carry

CONDITIONS = ("baseline", "treatment")


def load_runs(raw_dir: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(pathlib.Path(raw_dir).glob("*.json"))]


def rescore(record: dict) -> dict:
    """Recompute constraint_carry from the transcript with current metric code."""
    conv3 = [t for t in record["transcript"] if t["conversation"] == "conv-3"]
    final = conv3[0]["final"] if conv3 else ""
    buried = conv3[1]["final"] if len(conv3) > 1 else ""
    return constraint_carry(final, buried)


def summarize(records: list[dict]) -> dict:
    """-> {condition: {"n": int, "metrics": {name: passes}, "tokens": {...},
    "errors": int}}"""
    out: dict[str, dict] = {}
    for cond in CONDITIONS:
        runs = [r for r in records if r["condition"] == cond]
        if not runs:
            continue
        scored = [rescore(r) for r in runs]
        names = scored[0].keys() if scored else []
        out[cond] = {
            "n": len(runs),
            "metrics": {name: sum(bool(s[name]) for s in scored) for name in names},
            "tokens": {
                key: sum(r["tokens"].get(key, 0) for r in runs) / len(runs)
                for key in ("input_tokens", "output_tokens")
            },
            "errors": sum(len(r.get("errors", [])) for r in runs),
        }
    return out


def format_table(summary: dict) -> str:
    if not summary:
        return "No runs found."
    base, treat = summary.get("baseline"), summary.get("treatment")
    if not (base and treat):
        only = base or treat
        return f"Only one condition present (n={only['n']}); need both to compare."

    nb, nt = base["n"], treat["n"]
    lines = [
        f"| metric | baseline (n={nb}) | treatment (n={nt}) |",
        "|---|---|---|",
    ]
    for name in base["metrics"]:
        b, t = base["metrics"][name], treat["metrics"][name]
        mark = "" if b == t else " ←"
        lines.append(f"| {name} | {b}/{nb} | {t}/{nt}{mark} |")
    lines.append(
        f"| mean input tokens | {base['tokens']['input_tokens']:.0f} "
        f"| {treat['tokens']['input_tokens']:.0f} |"
    )
    lines.append(
        f"| mean output tokens | {base['tokens']['output_tokens']:.0f} "
        f"| {treat['tokens']['output_tokens']:.0f} |"
    )
    lines.append(f"| errored turns (total) | {base['errors']} | {treat['errors']} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", nargs="?", default="results/raw")
    args = ap.parse_args()
    records = load_runs(args.raw_dir)
    print(f"Pooled {len(records)} run file(s) from {args.raw_dir}\n")
    print(format_table(summarize(records)))


if __name__ == "__main__":
    main()
