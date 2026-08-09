"""Deterministic fact ids (idempotency under checkpoint replay) and short ids."""

import secrets
import uuid

# Fixed namespace for uuid5 derivation; never change once data exists.
_FACT_NS = uuid.UUID("b3ad5000-0000-4000-8000-000000000000")


def _canonical(*parts: str) -> str:
    """Length-prefixed join, so the encoding is unambiguous.

    A plain `":".join(parts)` is NOT injective: ("remember", "x:y") and
    ("remember:x", "y") both render as "remember:x:y" and therefore derived the
    same id — two different facts, one row. Prefixing each part with its length
    removes the ambiguity, because the decoder never has to guess where a part
    ends. Any separator would otherwise be forgeable, since both source keys and
    fact bodies are arbitrary user/model text.
    """
    return "".join(f"{len(p)}:{p}" for p in parts)


def derive_fact_id(namespace_id: uuid.UUID, source: str, source_key: str, body: str) -> uuid.UUID:
    """Content-derived fact id: replaying the same capture no-ops on insert.

    `source` (the write path) is part of the derivation because source keys are
    not globally unique. Passive capture falls back to using the message body as
    its key when a message carries no id, so a user typing exactly "remember"
    would otherwise derive the same id as an agent calling
    `remember_fact(body="remember")` — silently collapsing two distinct facts
    into one. Including the write path keeps them apart.

    Identity is therefore (namespace, write path, source key, body): a true
    repeat within one namespace collapses, and nothing else does.
    """
    return uuid.uuid5(_FACT_NS, _canonical(str(namespace_id), source, source_key, body))


def short_id(fact_id: uuid.UUID) -> str:
    """Beads-style display id, e.g. 'fact-a3f8b2c1'."""
    return f"fact-{fact_id.hex[:8]}"


def random_fork_suffix() -> str:
    """Collision-safe child-namespace segment, e.g. 'sub-a1b2c3d4'."""
    return f"sub-{secrets.token_hex(4)}"
