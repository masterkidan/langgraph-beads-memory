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


@dataclasses.dataclass(frozen=True)
class Arm:
    name: str
    kind: str  # "baseline" | "treatment"
    retire_superseded: bool = True
    subrecall: bool = False


ARMS: dict[str, Arm] = {
    "baseline": Arm("baseline", "baseline"),
    "treatment": Arm("treatment", "treatment"),
    "treatment-nosupersede": Arm("treatment-nosupersede", "treatment", retire_superseded=False),
    "treatment-subrecall": Arm("treatment-subrecall", "treatment", subrecall=True),
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
