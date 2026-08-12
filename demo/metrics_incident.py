"""Objective metrics for demo 2 (incident investigation).

Two of these are new in kind, and both exist because demo 1 showed the old
metric style was not enough:

`reproposes_ruled_out` needs to tell "let's check the connection pool" from
"the connection pool is ruled out". Plain substring matching cannot, which is
the exact bug that made `pick_is_feasible` overclaim in demo 1. It is handled
here with a proximity window rather than by pretending the problem away, and
the heuristic's limits are documented on the function.

`numeric_grounding` catches invented figures. Demo 1's blinded judge gave a
perfect recall score to an answer that named the right technique with a
fabricated magnitude ("up to 50%" where the corpus said 32x). A cheap check
that every number in an answer traces to the source material would have caught
it, and no judge was needed.
"""

from __future__ import annotations

import pathlib
import re

from demo.scenario_incident import PLANTED

CORPUS_INCIDENT = pathlib.Path(__file__).parent / "corpus_incident"

# Bare integers at or below this are treated as list numbering / ordinals rather
# than claims, so "1." "2." "3." in a numbered answer do not read as fabricated.
_LIST_NUMBER_CEILING = 20

# No trailing \b: it breaks on unit-suffixed numbers, which is most of them here.
# "4.2s" backtracked to "4" (the \b after "2" fails against "s"), while an answer
# writing "4.2 seconds" yielded "4.2" — the same figure, two different tokens, so
# a correct answer scored as ungrounded. "180ms" matched NOTHING at all. Measured
# on a real run that flagged 4.2 as fabricated when it appears in both the corpus
# and the user's own message.
_NUM = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d+(?:\.\d+)?")

# Short fact ids ("fact-23347999") are citations, not claims. Scoring them as
# fabricated figures penalised the treatment for using its own citation
# mechanism — measured: a run flagged 23347999 as an unsupported number when
# the answer had correctly written "(`fact-23347999`)" beside a constraint.
_FACT_ID = re.compile(r"`?fact-[0-9a-f]+`?", re.IGNORECASE)


def _numbers(text: str) -> set[str]:
    return set(_NUM.findall(_FACT_ID.sub(" ", text)))


# Language that puts a cause back on the to-do list. This is the narrow
# vocabulary; "ruled out" has an open-ended number of paraphrases and the first
# version of this metric tried to enumerate them. It produced a FALSE POSITIVE
# on real output — "Connection pool exhaustion, query performance, replication
# lag, and autovacuum activity were all within acceptable thresholds" is a
# correct recall of an elimination, and was scored as a re-proposal because
# that phrasing was not in the list. That single call was the entire evidence
# for the headline separation in an N=1 run.
_PROPOSAL_MARKERS = (
    "check",
    "re-check",
    "recheck",
    "investigate",
    "look at",
    "look into",
    "examine",
    "re-examine",
    "reexamine",
    "inspect",
    "verify",
    "revisit",
    "try ",
    "next step",
    "we should",
    "recommend",
    "suggest",
    "test ",
    "monitor",
    "action item",
    "to do",
    "todo",
)


def _window(text: str, start: int, end: int, radius: int = 140) -> str:
    return text[max(0, start - radius) : min(len(text), end + radius)]


def _marker_re(markers) -> re.Pattern:
    """Whole-word alternation over a marker list.

    Word boundaries are not optional here. Substring matching flagged a correct
    recall as a re-proposal because the window contained "the CHECKOUT service's
    own response time" and "check" is a proposal marker. This is the same bug
    class that made `classify_fragment` read "Checkout p99 latency..." as an
    imperative — a lesson evidently worth learning twice.
    """
    alts = sorted((m.strip() for m in markers), key=len, reverse=True)
    return re.compile(r"\b(?:" + "|".join(re.escape(a) for a in alts) + r")\b")


_PROPOSAL_RE = _marker_re(_PROPOSAL_MARKERS)
_ELIMINATION_RE = _marker_re(PLANTED["elimination_markers"])


def _reads_as_proposal(text: str, start: int, end: int) -> bool:
    """Does this mention read as "let's go do it" rather than "we did it"?

    Deliberately conservative: a mention is a re-proposal only when proposal
    language is nearby AND no elimination language is. Anything ambiguous
    counts as recall, so the metric under-detects rather than penalising an
    answer for how it happens to phrase a rule-out.
    """
    window = _window(text, start, end)
    if _ELIMINATION_RE.search(window):
        return False
    return bool(_PROPOSAL_RE.search(window))


def reproposes_ruled_out(answer: str) -> dict:
    """Does the answer put an already-eliminated cause back on the to-do list?

    A mention alone is not a failure — "we ruled out the connection pool" is
    exactly the recall we want. The failure is a mention with no elimination
    language anywhere near it, which reads as a live proposal.

    HEURISTIC, with known limits, stated because demo 1's lesson was that
    literal metrics quietly overclaim. It is tuned to UNDER-detect:
      - A mention framed as a proposal but sitting close to an unrelated
        elimination sentence scores as recalled (false negative).
      - A re-proposal phrased without any of the proposal verbs — "the
        connection pool remains a candidate" — scores as recalled (false
        negative).
    The opposite error was measured and removed: keying on elimination
    vocabulary flagged "...were all within acceptable thresholds" as a
    re-proposal, penalising a correct recall for its phrasing.
    Every flagged answer is therefore recorded in full in the run JSON so a
    disagreement can be checked by reading it, and the judge sees the same
    answers independently.
    """
    low = answer.lower()
    offenders = []
    for cause, variants in PLANTED["ruled_out_terms"].items():
        for variant in variants:
            for m in re.finditer(re.escape(variant), low):
                if _reads_as_proposal(low, m.start(), m.end()):
                    offenders.append(cause)
                    break
            if cause in offenders:
                break
    return {"reproposed": sorted(set(offenders)), "any": bool(offenders)}


def _any_variant(text: str, variants) -> bool:
    return any(v.lower() in text for v in variants)


def numeric_grounding(answer: str, extra_sources: list[str] | None = None) -> dict:
    """Every number in the answer should trace to the corpus or the conversation.

    Screening signal, not a verdict. A model that correctly *derives* a figure
    ("4.2s is roughly 23x the 180ms baseline") produces an unsupported token
    that is not a hallucination, so the unsupported set is recorded rather than
    silently counted against the run. What it reliably catches is the demo-1
    failure: a confident magnitude that appears nowhere in the source material.
    """
    allowed: set[str] = set()
    for path in sorted(CORPUS_INCIDENT.glob("*.md")):
        allowed |= _numbers(path.read_text())
    for src in extra_sources or []:
        allowed |= _numbers(src)

    def _is_list_number(tok: str) -> bool:
        # Times (13:20) are never list numbering, so they are always checked.
        return ":" not in tok and tok.isdigit() and int(tok) <= _LIST_NUMBER_CEILING

    unsupported = {n for n in _numbers(answer) - allowed if not _is_list_number(n)}
    return {"unsupported": sorted(unsupported), "grounded": not unsupported}


def incident_carry(
    next_steps: str,
    buried: str,
    breadth: str,
    timeline: str = "",
    conversation_texts: list[str] | None = None,
) -> dict:
    """Score the four scored answers of demo 2.

    `next_steps`  — conv-3: what to try next (predicted: treatment favoured)
    `buried`      — conv-4 Q1: an incidental measurement (predicted: baseline)
    `breadth`     — conv-4 Q2: everything ruled out (predicted: baseline)
    `timeline`    — conv-4 Q3: the corrected deploy time (predicted: treatment)

    The timeline is scored from its own question. It was briefly folded into
    the conv-3 question instead, which measurably crowded out the next-steps
    answer: both arms stopped naming the surviving cause and stopped proposing
    the reversible fix. One question, one thing measured.
    """
    ns = next_steps.lower()
    br = breadth.lower()
    tl = timeline.lower()
    reproposal = reproposes_ruled_out(ns)
    grounding = numeric_grounding(next_steps, conversation_texts)

    subsystems_named = sum(
        1 for variants in PLANTED["subsystem_variants"].values() if _any_variant(br, variants)
    )

    return {
        # --- predicted to favour the typed fact graph ---
        "avoids_reproposing_ruled_out": not reproposal["any"],
        "names_surviving_cause": _any_variant(ns, PLANTED["surviving_cause_variants"]),
        "proposes_reversible_fix": _any_variant(ns, PLANTED["reversible_fix_variants"]),
        "uses_corrected_deploy_time": _any_variant(tl, PLANTED["corrected_deploy_time_variants"]),
        "avoids_stale_deploy_time": not (
            _any_variant(tl, PLANTED["stale_deploy_time_variants"])
            and not _any_variant(tl, PLANTED["corrected_deploy_time_variants"])
        ),
        # --- predicted to favour the flat blob store ---
        "buried_metric_recalled": all(
            _any_variant(buried.lower(), v) for v in PLANTED["buried_metric_variants"]
        ),
        "breadth_subsystems_named": subsystems_named,
        "breadth_complete": subsystems_named == 3,
        # --- hallucination screen, no prediction ---
        "numerically_grounded": grounding["grounded"],
        # --- detail for inspection, not scored ---
        "_reproposed": reproposal["reproposed"],
        "_unsupported_numbers": grounding["unsupported"],
    }
