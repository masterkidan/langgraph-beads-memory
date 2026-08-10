"""Split a captured message into individually-retrievable facts.

One message stored as one fact means one embedding averaged across every topic
it contains. Measured consequences in a real run: a constraint stated once
inside a longer sentence never reached the top-K for a related question, and a
single bad `supersedes` edge retired three constraints at once because they
shared a row.

Splitting is deliberately heuristic and verbatim — no LLM, no paraphrasing.
Passive capture runs in the model-call hot path, so it cannot afford an
extraction call, and the verbatim-record guarantee has to survive per fragment
or the audit trail stops being an audit trail.

The bias is toward under-splitting. A fragment that is merely long is harmless;
a fragment shredded into meaningless pieces pollutes retrieval permanently.
"""

from __future__ import annotations

import re

# A fragment must look like a claim to stand alone. Trailing qualifiers such as
# "not vendor marketing numbers" have no verb and belong with the clause they
# modify — promoting them to facts creates retrievable nonsense.
_CLAUSE_VERBS = (
    " is ",
    " are ",
    " was ",
    " were ",
    " must ",
    " should ",
    " will ",
    " need ",
    " needs ",
    " want ",
    " wants ",
    " trust ",
    " have ",
    " has ",
    " can ",
    " cannot ",
    " prefer ",
    " require ",
    " requires ",
    " use ",
    " uses ",
    " costs ",
    " cost ",
)

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+")
# Enumeration separators, longest first so ", and " wins over " and ".
_ENUM_SEP = re.compile(r";\s+|,\s+and\s+|\s+and\s+|,\s+")
# "Strong" separators explicitly join coordinate items. A bare comma is weak: it
# equally introduces a subordinate clause ("promising, though I want...") that
# must NOT become its own fact, so a lone comma is not enough to split on.
_STRONG_SEP = re.compile(r";\s+|,\s+and\s+|\s+and\s+")

_MIN_FRAGMENT = 15


def _looks_like_clause(text: str) -> bool:
    padded = f" {text.strip().lower()} "
    return any(v in padded for v in _CLAUSE_VERBS)


def _split_enumeration(sentence: str) -> list[str]:
    """Split one sentence on enumeration separators, then re-join fragments that
    cannot stand alone."""
    # Drop a leading label such as "Constraints:" so it does not become a fact.
    body = sentence
    if ":" in sentence:
        head, _, tail = sentence.partition(":")
        if len(head.split()) <= 3 and tail.strip():
            body = tail.strip()

    parts = [p.strip() for p in _ENUM_SEP.split(body) if p.strip()]
    if len(parts) < 2:
        return [sentence.strip()]

    merged: list[str] = []
    for part in parts:
        # A fragment with no verb, or a very short one, is a continuation of the
        # previous clause rather than a claim of its own.
        if merged and (not _looks_like_clause(part) or len(part) < _MIN_FRAGMENT):
            sep = ", " if not merged[-1].endswith(",") else " "
            merged[-1] = f"{merged[-1]}{sep}{part}"
        else:
            merged.append(part)

    # If the split produced a single clause plus danglers, it was not really an
    # enumeration — keep the original sentence.
    if len(merged) < 2:
        return [sentence.strip()]
    return merged


def split_into_facts(text: str) -> list[str]:
    """Split `text` into verbatim fragments, each suitable as its own fact.

    Returns `[text]` when the text is a single claim, and `[]` for blank input.
    Every returned fragment is a substring of the input (modulo surrounding
    whitespace and a dropped enumeration label).
    """
    text = (text or "").strip()
    if not text:
        return []

    sentences = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
    if not sentences:
        return []

    facts: list[str] = []
    for sentence in sentences:
        # Only attempt clause-splitting when the sentence plausibly enumerates:
        # an explicit label, several separators, or one *strong* separator.
        # Ordinary prose joined by a single bare comma is left alone.
        separators = len(_ENUM_SEP.findall(sentence))
        strong = len(_STRONG_SEP.findall(sentence))
        labelled = ":" in sentence
        if separators >= 2 or strong >= 1 or (labelled and separators >= 1):
            facts.extend(_split_enumeration(sentence))
        else:
            facts.append(sentence)

    return [f for f in facts if f.strip()] or [text]


# --------------------------------------------------------------------------
# Claim vs directive
# --------------------------------------------------------------------------

STATEMENT = "statement"
DIRECTIVE = "directive"

# Verbs that open an instruction. Matched only at the start of a fragment, so
# "Compare them on cost" is a directive while "the comparison is done" is not.
_IMPERATIVE_OPENERS = (
    "please",
    "tell",
    "show",
    "give",
    "list",
    "find",
    "compare",
    "investigate",
    "research",
    "delegate",
    "explain",
    "describe",
    "summarize",
    "summarise",
    "remind",
    "check",
    "look",
    "consider",
    "be ",
    "make",
    "help",
    "let ",
    "recommend",
    "suggest",
    "pick",
    "choose",
    "evaluate",
    "assess",
    "review",
)

# Fragments opening with these are questions even without a question mark —
# small models drop terminal punctuation often enough to matter.
_INTERROGATIVE_OPENERS = (
    "which",
    "what",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "is ",
    "are ",
    "was ",
    "were ",
    "do ",
    "does ",
    "did ",
    "can ",
    "could ",
    "should ",
    "would ",
    "will ",
    "shall ",
    "may ",
    "might ",
    "and remind",
)

# First-person intent framing: "we need to X", "I want you to X" are goals, not
# claims about the world, even though they parse as declaratives.
_INTENT_PREFIXES = (
    "we need to",
    "we want to",
    "we should",
    "we must decide",
    "i need to",
    "i want to",
    "i would like",
    "we have to",
    "let's",
    "lets ",
)


def classify_fragment(text: str) -> str:
    """STATEMENT if the fragment asserts something, DIRECTIVE otherwise.

    Directives (questions, instructions, stated goals) are still captured and
    remain queryable — they are the provenance of every downstream choice, and
    dropping them would break the audit trail. They are classified apart so
    retrieval does not spend its top-K budget re-injecting text that is nearly
    the current query.

    Degenerate input classifies as DIRECTIVE: an empty fragment must never be
    treated as an assertable fact.
    """
    stripped = (text or "").strip()
    if not stripped:
        return DIRECTIVE
    lowered = stripped.lower()

    if stripped.endswith("?"):
        return DIRECTIVE
    if any(lowered.startswith(p) for p in _INTENT_PREFIXES):
        return DIRECTIVE
    if any(lowered.startswith(p) for p in _INTERROGATIVE_OPENERS):
        return DIRECTIVE
    if any(lowered.startswith(p) for p in _IMPERATIVE_OPENERS):
        return DIRECTIVE
    return STATEMENT
