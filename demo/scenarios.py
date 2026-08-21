"""Scenario + arm registry.

Demo 1 hard-wired one scenario and two conditions into the harness. Demo 2 adds
a second scenario and two ablation arms, so both become data rather than
control flow — otherwise every new arm means another branch in `run_once`.

An "arm" is a memory configuration. `baseline` and `treatment` are the two from
demo 1; the other two exist to answer questions demo 1 could not:

- `treatment-nosupersede` keeps the whole fact graph but stops retiring
  superseded facts, so stale and current values coexist exactly as they do in
  the baseline's blob store. If the result survives this ablation, typed
  invalidation is not what produced it.
- `treatment-subrecall` puts `recall_from_subagents` in the orchestrator's
  prompt. The tool has been bound since the descendant round but no scenario
  ever mentioned it, so the explicit sub-agent path has never actually run.
"""

from __future__ import annotations

import dataclasses
import pathlib
from collections.abc import Callable


@dataclasses.dataclass(frozen=True)
class Scenario:
    name: str
    root_prompt: str
    subagent_prompt: str
    subtopics: tuple[str, ...]
    corpus_dir: pathlib.Path
    conversations: tuple
    score: Callable[[list[dict]], dict]
    # Base tools available to the agent, as a factory so a scenario can supply
    # its own. Defaults to the corpus reader; the playground supplies web
    # search instead, which is the only difference between a scripted
    # benchmark and a live session as far as the memory layer is concerned.
    tools_factory: Callable[[Scenario], list] | None = None


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    kind: str  # "baseline" | "treatment"
    retire_superseded: bool = True
    subrecall: bool = False
    # Swap automatic injection for an agent-invoked `search_memory` whose name
    # and description are identical to langmem's. See `treatment-searchtool`.
    search_tool: bool = False
    # Capture what non-memory tools return. Set per-arm rather than defaulted,
    # because the flag is meaningless for `baseline` — that arm never builds a
    # BeadsMemoryMiddleware, so a default of True would advertise behaviour it
    # does not have. The library's own default is ON (see middleware.py).
    tool_capture: bool = False


ARMS: dict[str, Arm] = {
    "baseline": Arm("baseline", "baseline"),
    "treatment": Arm("treatment", "treatment", tool_capture=True),
    "treatment-nosupersede": Arm(
        "treatment-nosupersede", "treatment", retire_superseded=False, tool_capture=True
    ),
    "treatment-subrecall": Arm(
        "treatment-subrecall", "treatment", subrecall=True, tool_capture=True
    ),
    # Interface parity with the baseline: same tool name, same description, same
    # agent-authored query — our ranking behind it. The two arms previously
    # differed in interface AND ranking at once, so neither was attributable.
    # Measured cause for adding it: on incident/conv-3 the baseline's agent
    # wrote itself "incident status investigation progress ruled out facts" and
    # retrieved the root cause, while the treatment embedded the raw user
    # message ("New shift taking over...") and filled six of eight slots with
    # one restated constraint.
    "treatment-searchtool": Arm(
        "treatment-searchtool", "treatment", search_tool=True, tool_capture=True
    ),
    # Capture non-memory tool output into the calling agent's namespace. Kept as
    # an arm rather than a default because the trade is real in both directions,
    # measured on incident/gemma4:12b at N=3: `buried_metric_recalled` 0/3 -> 3/3
    # (the only metric no ranking change could move, and one the flat store
    # cannot reach at all), against `breadth_complete` 1/3 -> 0/3 and input
    # tokens falling from 35% below baseline to 13% below, because the store
    # doubled and selection got harder.
    # The ablation, kept so the trade stays measurable now that capture is the
    # default. Turning it off costs `buried_metric_recalled`, which passes 9 of
    # 56 incident runs and is unreachable by any ranking change.
    "treatment-notoolcapture": Arm("treatment-notoolcapture", "treatment", tool_capture=False),
}


def _vecdb() -> Scenario:
    from demo import metrics
    from demo.scenario import CONVERSATIONS, RESEARCH_SYSTEM_PROMPT, SUBAGENT_SYSTEM_PROMPT

    def score(transcript: list[dict]) -> dict:
        conv3 = [t for t in transcript if t["conversation"] == "conv-3"]
        return metrics.constraint_carry(
            conv3[0]["final"] if conv3 else "",
            conv3[1]["final"] if len(conv3) > 1 else "",
        )

    return Scenario(
        name="vecdb",
        root_prompt=RESEARCH_SYSTEM_PROMPT,
        subagent_prompt=SUBAGENT_SYSTEM_PROMPT,
        subtopics=("pgvector", "qdrant", "weaviate"),
        corpus_dir=pathlib.Path(__file__).parent / "corpus",
        conversations=tuple(CONVERSATIONS),
        score=score,
    )


def _incident() -> Scenario:
    from demo.metrics_incident import incident_carry
    from demo.scenario_incident import (
        CONVERSATIONS,
        INCIDENT_SYSTEM_PROMPT,
        INVESTIGATOR_SYSTEM_PROMPT,
        SUBSYSTEMS,
    )

    def score(transcript: list[dict]) -> dict:
        def final(conv: str, idx: int) -> str:
            turns = [t for t in transcript if t["conversation"] == conv]
            return turns[idx]["final"] if len(turns) > idx else ""

        return incident_carry(
            next_steps=final("conv-3", 0),
            buried=final("conv-4", 0),
            breadth=final("conv-4", 1),
            timeline=final("conv-4", 2),
            conversation_texts=[t["user"] for t in transcript],
        )

    return Scenario(
        name="incident",
        root_prompt=INCIDENT_SYSTEM_PROMPT,
        subagent_prompt=INVESTIGATOR_SYSTEM_PROMPT,
        subtopics=SUBSYSTEMS,
        corpus_dir=pathlib.Path(__file__).parent / "corpus_incident",
        conversations=tuple(CONVERSATIONS),
        score=score,
    )


SCENARIOS: dict[str, Callable[[], Scenario]] = {
    "vecdb": _vecdb,
    "incident": _incident,
}


def get_scenario(name: str) -> Scenario:
    if name not in SCENARIOS:
        raise SystemExit(f"unknown scenario {name!r}; available: {', '.join(SCENARIOS)}")
    return SCENARIOS[name]()


def get_arm(name: str) -> Arm:
    if name not in ARMS:
        raise SystemExit(f"unknown arm {name!r}; available: {', '.join(ARMS)}")
    return ARMS[name]
