"""Content-addressed identifiers.

Every id in this package is derived from what a thing *is*, never from a random
draw. That is what makes a LangGraph checkpoint replay a no-op instead of a
duplicate — and it only holds if the derivation is injective and rooted in
stable inputs, so both properties are pinned down by tests/test_collisions.py.
"""

import hashlib
import secrets
import uuid

# Fixed namespaces for uuid5 derivation; never change once data exists.
_FACT_NS = uuid.UUID("b3ad5000-0000-4000-8000-000000000000")
_NAMESPACE_NS = uuid.UUID("b3ad5000-0000-4000-8000-000000000001")


def fingerprint(text: str) -> str:
    """Stable SHA-256 content fingerprint.

    Used instead of raw text wherever content contributes to an id. Two reasons:
    it bounds the derivation input to 64 chars regardless of body size, and it
    keeps arbitrary user/model text out of the key space, where it could
    otherwise be mistaken for a structural component.
    """
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def content_key(text: str) -> str:
    """A source key standing in for a message that carries no id.

    Explicitly labelled, so it can never be confused with a real key like
    "remember". Using the raw body here (the previous behaviour) meant identity
    and content were the same string, which is how a user message reading
    "remember" once collided with `remember_fact(body="remember")`.
    """
    return f"sha256:{fingerprint(text)}"


def _canonical(*parts: str) -> str:
    """Length-prefixed join, so the encoding is unambiguous.

    A plain ":".join(parts) is NOT injective: ("remember", "x:y") and
    ("remember:x", "y") both render as "remember:x:y" and would derive the same
    id — two different facts, one row. Length-prefixing removes the ambiguity,
    because a decoder never has to guess where a part ends. Any separator would
    otherwise be forgeable, since keys and bodies are arbitrary text.
    """
    return "".join(f"{len(p)}:{p}" for p in parts)


def derive_namespace_id(session_id: str, extra_path: list[str]) -> uuid.UUID:
    """Namespace id derived from (session_id, extra_path).

    Deliberately not random. Fact ids are derived from the namespace id, so a
    random namespace id would make the whole chain irreproducible: tear the
    database down, replay the identical conversation, and every fact would get a
    new id. Deriving it keeps content-addressing true end to end, and makes the
    id agree with the table's UNIQUE (session_id, extra_path) constraint by
    construction rather than by convention.
    """
    return uuid.uuid5(_NAMESPACE_NS, _canonical(session_id, *extra_path))


def derive_fact_id(
    session_id: str,
    namespace_id: uuid.UUID,
    source: str,
    source_key: str,
    body: str,
) -> uuid.UUID:
    """Content-derived fact id: replaying the same capture no-ops on insert.

    Identity is (session, namespace, write path, source key, body fingerprint).

    `session_id` is included even though `namespace_id` already encodes it, so
    the derivation states its own scope rather than relying on that invariant
    holding elsewhere. `source` is included because source keys are not globally
    unique across write paths — passive capture keys on a message id, the tools
    key on a constant — and without it two different facts could derive one id.
    """
    return uuid.uuid5(
        _FACT_NS,
        _canonical(
            session_id,
            str(namespace_id),
            source,
            source_key,
            fingerprint(body),
        ),
    )


def short_id(fact_id: uuid.UUID) -> str:
    """Beads-style display id, e.g. 'fact-a3f8b2c1'."""
    return f"fact-{fact_id.hex[:8]}"


def random_fork_suffix() -> str:
    """Collision-safe child-namespace segment, e.g. 'sub-a1b2c3d4'.

    Deliberately random, unlike everything else here: two concurrent sub-agents
    spawned for the same task must get distinct namespaces, so there is nothing
    stable to derive from. Randomness is the point.
    """
    return f"sub-{secrets.token_hex(4)}"
