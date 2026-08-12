"""Pool objective metrics across N runs into the results table.

Rescores every saved transcript with the CURRENT metric code rather than
trusting the `constraint_carry` snapshot stored in each JSON. Runs recorded
before a metric fix would otherwise be pooled with runs recorded after it,
silently mixing two measurement definitions.

Scenario- and arm-aware: each record names its scenario, and that scenario's own
scoring function rescores it, so demo 1 and demo 2 runs cannot be pooled by
accident. Arms are read from the data rather than hard-coded, so an ablation arm
shows up as a column without a code change.
"""

from __future__ import annotations

import argparse
import json
import pathlib

from demo.scenarios import get_scenario

# Stable display order; any other arm found in the data is appended after these.
ARM_ORDER = ("baseline", "treatment", "treatment-nosupersede", "treatment-subrecall")


def load_runs(raw_dir: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(pathlib.Path(raw_dir).glob("*.json"))]


def rescore(record: dict) -> dict:
    """Recompute the scenario's metrics from the transcript with current code."""
    scenario = get_scenario(record.get("scenario", "vecdb"))
    return scenario.score(record["transcript"])


def _arms_in(records: list[dict]) -> list[str]:
    present = {r["condition"] for r in records}
    ordered = [a for a in ARM_ORDER if a in present]
    return ordered + sorted(present - set(ordered))


def summarize(records: list[dict]) -> dict:
    out: dict[str, dict] = {}
    for arm in _arms_in(records):
        runs = [r for r in records if r["condition"] == arm]
        scored = [rescore(r) for r in runs]
        # Keys prefixed with "_" are inspection detail (which cause was
        # re-proposed, which numbers were unsupported), not scored metrics.
        names = [k for k in scored[0] if not k.startswith("_")] if scored else []
        out[arm] = {
            "n": len(runs),
            "metrics": {name: [s[name] for s in scored] for name in names},
            "tokens": {
                key: sum(r["tokens"].get(key, 0) for r in runs) / len(runs)
                for key in ("input_tokens", "output_tokens")
            },
            "errors": sum(len(r.get("errors", [])) for r in runs),
            "detail": [{k: v for k, v in s.items() if k.startswith("_")} for s in scored],
        }
    return out


def _cell(values: list, n: int) -> str:
    """Booleans read as a pass count; numeric metrics read as a mean."""
    if all(isinstance(v, bool) for v in values):
        return f"{sum(values)}/{n}"
    return f"{sum(values) / len(values):.1f} avg"


def format_table(summary: dict) -> str:
    if not summary:
        return "No runs found."
    arms = list(summary)
    if len(arms) == 1:
        arm = arms[0]
        lines = [f"Only one arm present: {arm} (n={summary[arm]['n']}).", ""]
    else:
        lines = []

    header = "| metric | " + " | ".join(f"{a} (n={summary[a]['n']})" for a in arms) + " |"
    lines += [header, "|" + "---|" * (len(arms) + 1)]
    for name in summary[arms[0]]["metrics"]:
        cells = [_cell(summary[a]["metrics"][name], summary[a]["n"]) for a in arms]
        differs = len(set(cells)) > 1
        lines.append(f"| {name} | " + " | ".join(cells) + (" |" if not differs else " ←|"))
    for label, key in (
        ("mean input tokens", "input_tokens"),
        ("mean output tokens", "output_tokens"),
    ):
        lines.append(
            f"| {label} | " + " | ".join(f"{summary[a]['tokens'][key]:.0f}" for a in arms) + " |"
        )
    lines.append("| errored turns | " + " | ".join(str(summary[a]["errors"]) for a in arms) + " |")

    # Surface the inspection detail: a re-proposed cause or an unsupported number
    # is the thing worth reading, and burying it in JSON means nobody does.
    notes = []
    for arm in arms:
        for i, det in enumerate(summary[arm]["detail"]):
            flagged = {k: v for k, v in det.items() if v}
            if flagged:
                notes.append(
                    f"  {arm} run{i}: " + ", ".join(f"{k}={v}" for k, v in flagged.items())
                )
    if notes:
        lines += ["", "Flagged for inspection:", *notes]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", nargs="?", default="results/raw")
    args = ap.parse_args()
    records = load_runs(args.raw_dir)
    scenarios = {r.get("scenario", "vecdb") for r in records}
    if len(scenarios) > 1:
        raise SystemExit(
            f"{args.raw_dir} mixes scenarios {sorted(scenarios)}; they are not poolable. "
            "Point this at one scenario's directory."
        )
    print(f"Pooled {len(records)} run file(s) from {args.raw_dir}\n")
    print(format_table(summarize(records)))


if __name__ == "__main__":
    main()
