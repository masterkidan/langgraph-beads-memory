"""Where does a scenario run actually spend its time?

Print statements can only tell you when a *turn* started; they cannot see inside
a turn, where an agent may make a dozen model calls and spawn three sub-agents.
This attributes every span to the work that caused it, using LangChain's
callback hooks plus wrappers on the two non-LLM subsystems (embeddings and
Postgres), so the totals account for the whole wall clock rather than a subset.

Spans nest: a sub-agent tool call contains the model calls made inside it. The
report therefore separates *inclusive* time (a span and everything under it)
from *self* time (the span minus its children), because only self time can be
summed across categories without double-counting.
"""

from __future__ import annotations

import json
import threading
import time
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from langchain_core.callbacks import BaseCallbackHandler


@dataclass
class Span:
    category: str  # 'llm' | 'tool' | 'embed' | 'db'
    label: str
    started: float
    ended: float | None = None
    parent: UUID | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    children: list[Span] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return (self.ended or time.monotonic()) - self.started

    @property
    def self_seconds(self) -> float:
        """Inclusive time minus time attributable to nested spans."""
        return max(0.0, self.seconds - sum(c.seconds for c in self.children))


class RunProfiler(BaseCallbackHandler):
    """Callback handler recording a span per model call and per tool call.

    Thread-safe because LangGraph's ToolNode dispatches tool calls across a
    thread pool — three researchers run on three threads, and an unsynchronised
    dict would drop spans under exactly the workload we care about measuring.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._open: dict[UUID, Span] = {}
        self.spans: list[Span] = []
        self.phase = "unassigned"
        self._phase_of: dict[UUID, str] = {}

    # -- manual spans (embeddings, database) --------------------------------
    @contextmanager
    def span(self, category: str, label: str):
        started = time.monotonic()
        try:
            yield
        finally:
            s = Span(category=category, label=label, started=started, ended=time.monotonic())
            with self._lock:
                s.label = f"[{self.phase}] {s.label}"
                self.spans.append(s)

    # -- LLM ----------------------------------------------------------------
    def on_chat_model_start(self, serialized, messages, *, run_id, parent_run_id=None, **kw):
        self._start(run_id, parent_run_id, "llm", "model call")

    def on_llm_start(self, serialized, prompts, *, run_id, parent_run_id=None, **kw):
        self._start(run_id, parent_run_id, "llm", "model call")

    def on_llm_end(self, response, *, run_id, **kw):
        with self._lock:
            span = self._open.pop(run_id, None)
            if span is None:
                return
            span.ended = time.monotonic()
            usage = _usage_from(response)
            span.input_tokens = usage.get("input_tokens", 0)
            span.output_tokens = usage.get("output_tokens", 0)
            self.spans.append(span)

    def on_llm_error(self, error, *, run_id, **kw):
        self._finish(run_id)

    # -- tools --------------------------------------------------------------
    def on_tool_start(self, serialized, input_str, *, run_id, parent_run_id=None, **kw):
        name = (serialized or {}).get("name", "tool")
        self._start(run_id, parent_run_id, "tool", name)

    def on_tool_end(self, output, *, run_id, **kw):
        self._finish(run_id)

    def on_tool_error(self, error, *, run_id, **kw):
        self._finish(run_id)

    # -- internals ----------------------------------------------------------
    def _start(self, run_id, parent_run_id, category, label):
        with self._lock:
            span = Span(
                category=category,
                label=f"[{self.phase}] {label}",
                started=time.monotonic(),
                parent=parent_run_id,
            )
            self._open[run_id] = span
            self._phase_of[run_id] = self.phase
            parent = self._open.get(parent_run_id) if parent_run_id else None
            if parent is not None:
                parent.children.append(span)

    def _finish(self, run_id):
        with self._lock:
            span = self._open.pop(run_id, None)
            if span is not None:
                span.ended = time.monotonic()
                self.spans.append(span)

    # -- reporting ----------------------------------------------------------
    def report(self, wall_seconds: float) -> str:
        by_cat: dict[str, list[Span]] = defaultdict(list)
        for s in self.spans:
            by_cat[s.category].append(s)

        lines = [f"wall clock: {wall_seconds:.1f}s", ""]
        lines.append(f"{'category':10} {'count':>6} {'summed s':>10} {'busy s':>9} {'% wall':>7}")
        lines.append("-" * 48)
        for cat in ("llm", "tool", "embed", "db"):
            spans = by_cat.get(cat, [])
            if not spans:
                continue
            summed = sum(s.seconds for s in spans)
            busy = _union_seconds(spans)
            lines.append(
                f"{cat:10} {len(spans):6} {summed:10.1f} {busy:9.1f} "
                f"{100 * busy / wall_seconds:6.1f}%"
            )
        lines.append("-" * 48)
        # Spans overlap: sub-agents run concurrently on a thread pool, and a tool
        # span contains the model calls made inside it. Summing across categories
        # would therefore exceed the wall clock, so report *busy* time (the union
        # of intervals, counting overlap once) alongside the raw sum. The ratio
        # of the two is the effective concurrency.
        llm = by_cat.get("llm", [])
        if llm:
            summed = sum(s.seconds for s in llm)
            busy = _union_seconds(llm)
            idle = wall_seconds - busy
            lines.append(
                f"{'model busy':10} {'':6} {'':10} {busy:9.1f} {100 * busy / wall_seconds:6.1f}%"
            )
            lines.append(
                f"{'idle/other':10} {'':6} {'':10} {idle:9.1f} {100 * idle / wall_seconds:6.1f}%"
            )
            lines.append("")
            lines.append(
                f"effective concurrency: {summed / busy:.2f}x "
                f"({summed:.0f}s of model work in {busy:.0f}s of busy time)"
            )

        lines += ["", "slowest individual spans (inclusive):"]
        for s in sorted(self.spans, key=lambda x: x.seconds, reverse=True)[:12]:
            tok = f"  in={s.input_tokens} out={s.output_tokens}" if s.input_tokens else ""
            lines.append(f"  {s.seconds:7.1f}s  {s.category:5} {s.label[:56]}{tok}")

        lines += ["", "self time by phase:"]
        per_phase: dict[str, float] = defaultdict(float)
        for s in self.spans:
            phase = s.label.split("]")[0].lstrip("[") if s.label.startswith("[") else "?"
            per_phase[phase] += s.self_seconds
        for phase, secs in sorted(per_phase.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {secs:7.1f}s  {phase}")

        llm_spans = by_cat.get("llm", [])
        if llm_spans:
            tin = sum(s.input_tokens for s in llm_spans)
            tout = sum(s.output_tokens for s in llm_spans)
            llm_self = sum(s.self_seconds for s in llm_spans)
            lines += [
                "",
                f"model calls: {len(llm_spans)}  "
                f"mean {llm_self / len(llm_spans):.1f}s  "
                f"in_tok={tin} out_tok={tout}  "
                f"{tout / llm_self if llm_self else 0:.1f} out-tok/s",
            ]
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps(
            [
                {
                    "category": s.category,
                    "label": s.label,
                    "seconds": round(s.seconds, 3),
                    "self_seconds": round(s.self_seconds, 3),
                    "input_tokens": s.input_tokens,
                    "output_tokens": s.output_tokens,
                }
                for s in self.spans
            ],
            indent=2,
        )


def _union_seconds(spans: list[Span]) -> float:
    """Total wall time during which at least one of these spans was active.

    Concurrent spans overlap, so a plain sum overstates elapsed time. Merging
    intervals counts overlap once and gives a figure that can honestly be
    compared against the wall clock.
    """
    intervals = sorted((s.started, s.ended or s.started) for s in spans if s.ended is not None)
    total = 0.0
    cur_start, cur_end = None, None
    for start, end in intervals:
        if cur_end is None or start > cur_end:
            if cur_end is not None:
                total += cur_end - cur_start
            cur_start, cur_end = start, end
        else:
            cur_end = max(cur_end, end)
    if cur_end is not None:
        total += cur_end - cur_start
    return total


def _usage_from(response: Any) -> dict:
    usage = getattr(response, "llm_output", None) or {}
    if isinstance(usage, dict) and usage.get("input_tokens"):
        return usage
    try:
        gen = response.generations[0][0]
        meta = getattr(gen.message, "usage_metadata", None) or {}
        return meta
    except Exception:  # noqa: BLE001 - profiling must never break the run
        return {}


class TimedEmbedder:
    """Wraps an Embedder so embedding time is attributed, not lumped into 'unaccounted'."""

    def __init__(self, inner, profiler: RunProfiler):
        self._inner = inner
        self._profiler = profiler
        self.dim = inner.dim

    def embed(self, text: str) -> list[float]:
        with self._profiler.span("embed", "embed"):
            return self._inner.embed(text)


class TimedConnection:
    """Wraps a psycopg connection so query time is attributed separately."""

    def __init__(self, inner, profiler: RunProfiler):
        self._inner = inner
        self._profiler = profiler

    def execute(self, *args, **kwargs):
        with self._profiler.span("db", "query"):
            return self._inner.execute(*args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._inner, name)
