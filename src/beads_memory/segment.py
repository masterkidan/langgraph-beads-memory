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

import os
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
    # Assertion and finding verbs. Added when splitting was extended to agent
    # conclusions: the list knew copulas and modals but not the verbs a
    # conclusion is actually written with, so "I recommend pgvector." — an
    # unambiguous conclusion — was discarded as insubstantial. The same gap hid
    # "Checkout p99 latency WENT from 180ms to 4.2s" on the capture side.
    " recommend ",
    " recommends ",
    " suggest ",
    " suggests ",
    " found ",
    " identified ",
    " concluded ",
    " concludes ",
    " shows ",
    " showed ",
    " indicates ",
    " indicated ",
    " confirms ",
    " confirmed ",
    " ruled ",
    " remains ",
    " stayed ",
    " went ",
    " deployed ",
    " introduced ",
    " caused ",
    " occurred ",
    " measured ",
    " increased ",
    " decreased ",
    " rose ",
    " fell ",
    " peaked ",
    " disable ",
    " roll ",
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


_HAS_DIGIT = re.compile(r"\d")

# A fragment that is only a number and punctuation: "4.", "3)", "1 -". These come
# from LLM numbered lists, which `_SENTENCE_END` splits on the "." after the
# digit, and they are the single worst thing in the store.
#
# MEASURED. On LongMemEval knowledge-update, for four numeric golds the
# BEST-matching fact in the entire store was one of these markers — '4.', '3.',
# '5.' — because a bare digit matches every numeric query while carrying no
# answer, and the sentence that actually stated the value lost its slot to it.
# Five of ten failures traced here.
_BARE_ORDINAL = re.compile(r"^\W*\d+\s*[.)\]:;,-]*\W*$")

# At least one real word. A digit-bearing fragment with no word is a measurement
# with nothing measured — "41ms" alone answers no question, because the subject
# it belonged to was in the clause the splitter cut it from.
_REAL_WORD = re.compile(r"[A-Za-z]{3,}")


def is_substantive(fragment: str) -> bool:
    """Is this fragment worth storing as a retrievable fact?

    Conversational frame is not memory. Measured on a real run: "New shift
    taking over." was the **top-ranked** fact injected for the question "what
    should we try next", closer to the query than anything in the store that
    actually answered it, with "Given everything we've established" second.
    Between them they consumed two of eight injection slots with no content.
    They rank highly precisely because they echo the phrasing of the question —
    similarity search rewards restating a question over answering it.

    The test is deliberately two-sided, because a stricter one was measured and
    rejected. Requiring a clause verb alone would have dropped "Checkout p99
    latency went from 180ms to 4.2s" and "Correction on the timeline: the
    deploy actually went out at 13:20 UTC" — the incident's central symptom and
    the correction the whole `supersedes` mechanism is meant to carry — because
    `_CLAUSE_VERBS` knows copulas and modals but not "went" or "deployed".

    So a fragment is substantive if it asserts (has a clause verb) OR carries a
    figure. That keeps every stated constraint ("it must be self-hostable",
    "I only trust primary benchmark data") and every measurement, while
    discarding pure discourse framing. Directives are exempt: they are held out
    of retrieval already and are kept for provenance.
    """
    stripped = fragment.strip()
    if _BARE_ORDINAL.match(stripped):
        return False
    if _looks_like_clause(stripped):
        return True
    # The digit branch keeps measurements ("Checkout p99 latency went from 180ms
    # to 4.2s") without keeping list scaffolding, by requiring the fragment to
    # name something as well as count it.
    return bool(_HAS_DIGIT.search(stripped)) and bool(_REAL_WORD.search(stripped))


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


# Master switch for regex splitting. OFF returns the text whole.
#
# MEASURED reason this exists. The splitter was written for human prose and is
# fed LLM markdown, where `_SENTENCE_END` treats "4." in a numbered list as a
# sentence and `is_substantive` passes it because _HAS_DIGIT matches any digit.
# The store fills with punctuation-only facts that match every numeric query. On
# LongMemEval knowledge-update, for four numeric golds the BEST-matching fact in
# the whole store was a bare list marker — '4.', '3.', '5.' — while the sentence
# that stated the answer had not survived as a retrievable unit. Five of ten
# failures traced to this, and no ranking change can reach any of them.
#
# It also destroys co-occurrence: a value split from its subject only matches a
# query on its own thin surface. Every arm that keeps turns whole (bm25,
# PostgresStore, full context) beat the fact graph on that benchmark.
#
# The counterweight is real and is why this is a knob rather than a deletion:
# splitting is what cut input tokens ~40%, the one result that has held up
# across every measurement. Turning it off trades cost for reach.
SPLIT_ENABLED = os.environ.get("BEADS_SPLIT", "1") not in ("0", "false", "off")


def split_into_facts(text: str) -> list[str]:
    """Split `text` into verbatim fragments, each suitable as its own fact.

    Returns `[text]` when the text is a single claim, and `[]` for blank input.
    Every returned fragment is a substring of the input (modulo surrounding
    whitespace and a dropped enumeration label).
    """
    text = (text or "").strip()
    if not text:
        return []
    if not SPLIT_ENABLED:
        return [text]

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


_WORD = re.compile(r"[a-z']+")


def _first_word(lowered: str) -> str:
    """The first alphabetic word, so punctuation and digits do not hide it."""
    m = _WORD.search(lowered)
    return m.group(0) if m else ""


# A turn often continues the previous one ("And list everything we ruled out").
# The conjunction is not part of the speech act, so it is stripped before
# classifying. This generalises what used to be a single hard-coded phrase,
# "and remind", added for one demo-1 question; without it "And list ..." and
# "So tell me ..." were captured as assertions.
_LEADING_CONJUNCTIONS = ("and", "but", "so", "then", "also", "plus")


def _strip_leading_conjunction(lowered: str) -> str:
    first = _first_word(lowered)
    if first in _LEADING_CONJUNCTIONS:
        return lowered[lowered.index(first) + len(first) :].lstrip(" ,;-—")
    return lowered


def _split_openers(openers: tuple[str, ...]) -> tuple[frozenset[str], tuple[str, ...]]:
    """Partition an opener list into whole-word tokens and multi-word phrases.

    The lists are written with trailing spaces on some entries ("is ", "be ",
    "let ") to stop a prefix match from firing inside a longer word. Whole-word
    matching makes that trick unnecessary, but it also makes those entries
    unmatchable unless they are stripped — which would silently disable
    detection of verb-initial questions like "Is the budget fixed", exactly the
    unpunctuated case those entries were added for. Phrases with an internal
    space ("and remind") still need a prefix test.
    """
    words, phrases = set(), []
    for opener in openers:
        token = opener.strip()
        (words.add(token) if " " not in token else phrases.append(token))
    return frozenset(words), tuple(phrases)


_IMPERATIVE_WORDS, _IMPERATIVE_PHRASES = _split_openers(_IMPERATIVE_OPENERS)
_INTERROGATIVE_WORDS, _INTERROGATIVE_PHRASES = _split_openers(_INTERROGATIVE_OPENERS)


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
    lowered = _strip_leading_conjunction(lowered)
    # Multi-word prefixes ("i want to", "we need to") are phrases, so a plain
    # prefix test is right for these.
    if any(lowered.startswith(p) for p in _INTENT_PREFIXES):
        return DIRECTIVE
    # Single-word openers must match a WHOLE word. Matching on a raw prefix
    # classified "Checkout p99 latency went from 180ms to 4.2s" as a directive,
    # because "checkout" starts with "check" — and directives are held out of
    # retrieval, so the central fact of an incident silently became
    # unretrievable. "Listing prices rose" hit the same bug via "list". A
    # statement misfiled as a directive is invisible to search, which is the
    # expensive direction to get wrong.
    first = _first_word(lowered)
    if first in _INTERROGATIVE_WORDS or first in _IMPERATIVE_WORDS:
        return DIRECTIVE
    if any(lowered.startswith(p) for p in _INTERROGATIVE_PHRASES + _IMPERATIVE_PHRASES):
        return DIRECTIVE
    return STATEMENT


# --------------------------------------------------------------------------
# Values, for invalidation
# --------------------------------------------------------------------------

_VALUE = re.compile(r"\b\d{1,2}:\d{2}\b|\b\d[\d,]*(?:\.\d+)?\s*[kmb]?\b", re.IGNORECASE)
_MULT = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}


def value_tokens(text: str) -> set[str]:
    """Normalised figures in a piece of text, for comparing what a fact asserts.

    "$100k" and "$100,000" are the same value written two ways; a stale
    restatement of a corrected fact almost always writes it the other way. Times
    are kept verbatim, since 13:50 is not a quantity.

    This exists because cosine distance cannot tell $100,000 from $50,000 — the
    two scored 0.002 apart in a real run and the model took the stale one. A
    normalised token comparison separates them exactly.
    """
    out: set[str] = set()
    for match in _VALUE.finditer(text.lower()):
        token = match.group(0).strip()
        if ":" in token:
            out.add(token)
            continue
        multiplier = 1
        if token and token[-1] in _MULT:
            multiplier = _MULT[token[-1]]
            token = token[:-1].strip()
        try:
            value = float(token.replace(",", "")) * multiplier
        except ValueError:
            continue
        out.add(f"{value:.10g}")
    return out


def _shape(token: str) -> str:
    return "time" if ":" in token else "number"


def contested_values(old: str, new: str) -> tuple[set[str], set[str]]:
    """Which values a correction actually changed: `(stale, replacement)`.

    Not every figure in a corrected fact is stale. "Release 2.14 was deployed at
    13:50 UTC", corrected to 13:20, leaves 2.14 perfectly current — so retiring
    every later fact that merely mentions a value from the old one would take
    "Release 2.14 introduced a caching layer" with it.

    A value is contested only when the correction asserts a *different* value of
    the same shape. Times and numbers are separate shapes, which is enough to
    keep a changed timestamp from implicating an unchanged version number.

    Note that a correction routinely names the old value while rejecting it
    ("went out at 13:20 UTC, not 13:50"), so the stale set cannot simply be
    "old minus new" — it is the old fact's values of any shape the correction
    disturbed.
    """
    old_tokens, new_tokens = value_tokens(old), value_tokens(new)
    introduced = new_tokens - old_tokens
    shapes = {_shape(t) for t in introduced}
    stale = {t for t in old_tokens if _shape(t) in shapes}
    return stale, introduced
