"""Cross-model view: does the memory harness make each model more efficient?

`aggregate.py` deliberately refuses to pool across models, because a model
change moves every metric. This asks the different question: **within** each
model, what does the harness change? That is a paired comparison per model, so
the model's own strength cancels out and what remains is the harness's effect.

Efficiency here has two axes, and a claim needs both:

    accuracy   — objective metrics passed, as a fraction of those scored
    cost       — input and output tokens

A harness that raises accuracy while costing more tokens is a trade. One that
holds accuracy while cutting tokens is a win. One that does both is the claim
worth making, and it has to be shown per model rather than argued from one.

Usage:
    uv run python -m demo.compare_models results/matrix
"""

from __future__ import annotations

import argparse
import json
import pathlib
from collections import defaultdict


def _load(raw_dir: str) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(pathlib.Path(raw_dir).rglob("*.json"))]


def _score(record: dict) -> tuple[int, int]:
    """(passed, scored) over the boolean metrics, rescored with current code."""
    from demo.scenarios import get_scenario

    metrics = get_scenario(record.get("scenario", "vecdb")).score(record["transcript"])
    bools = [v for k, v in metrics.items() if not k.startswith("_") and isinstance(v, bool)]
    return sum(bools), len(bools)


def summarize(records: list[dict]) -> dict:
    out: dict[tuple[str, str], dict] = defaultdict(
        lambda: {"passed": 0, "scored": 0, "tin": 0, "tout": 0, "n": 0, "errors": 0}
    )
    for r in records:
        key = (r.get("model", "unknown"), r["condition"])
        p, s = _score(r)
        agg = out[key]
        agg["passed"] += p
        agg["scored"] += s
        agg["tin"] += r["tokens"].get("input_tokens", 0)
        agg["tout"] += r["tokens"].get("output_tokens", 0)
        agg["errors"] += len(r.get("errors", []))
        agg["n"] += 1
    return dict(out)


def format_table(summary: dict) -> str:
    models = sorted({m for m, _ in summary})
    lines = [
        "| model | arm | n | accuracy | input tok | output tok |",
        "|---|---|---|---|---|---|",
    ]
    for model in models:
        for arm in sorted({a for m, a in summary if m == model}):
            a = summary[(model, arm)]
            acc = 100 * a["passed"] / a["scored"] if a["scored"] else 0
            lines.append(
                f"| {model} | {arm} | {a['n']} | {acc:.0f}% "
                f"({a['passed']}/{a['scored']}) | {a['tin'] / a['n']:.0f} "
                f"| {a['tout'] / a['n']:.0f} |"
            )

    lines += ["", "**Harness effect, per model** (treatment vs baseline):", ""]
    lines += ["| model | accuracy | input tokens | verdict |", "|---|---|---|---|"]
    for model in models:
        b, t = summary.get((model, "baseline")), summary.get((model, "treatment"))
        if not (b and t):
            lines.append(f"| {model} | — | — | needs both arms |")
            continue
        ba = b["passed"] / b["scored"] if b["scored"] else 0
        ta = t["passed"] / t["scored"] if t["scored"] else 0
        bi, ti = b["tin"] / b["n"], t["tin"] / t["n"]
        d_acc, d_tok = 100 * (ta - ba), 100 * (ti - bi) / bi if bi else 0
        # Cheaper AND at least as accurate is the only unambiguous win. Anything
        # else is a trade, and saying so is the point of showing both columns.
        if d_acc >= 0 and d_tok < 0:
            verdict = "win — same or better, cheaper"
        elif d_acc > 0 and d_tok >= 0:
            verdict = "trade — more accurate, costlier"
        elif d_acc < 0 and d_tok < 0:
            verdict = "trade — cheaper, less accurate"
        else:
            verdict = "loss — costlier and no better"
        lines.append(f"| {model} | {d_acc:+.0f} pts | {d_tok:+.0f}% | {verdict} |")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("raw_dir", nargs="?", default="results/matrix")
    args = ap.parse_args()
    records = _load(args.raw_dir)
    if not records:
        raise SystemExit(f"no run files under {args.raw_dir}")
    scenarios = {r.get("scenario", "vecdb") for r in records}
    if len(scenarios) > 1:
        raise SystemExit(
            f"{args.raw_dir} mixes scenarios {sorted(scenarios)}. Metrics differ between "
            "scenarios, so accuracy percentages would not be comparable."
        )
    print(f"{len(records)} runs, scenario={scenarios.pop()}\n")
    print(format_table(summarize(records)))


if __name__ == "__main__":
    main()
