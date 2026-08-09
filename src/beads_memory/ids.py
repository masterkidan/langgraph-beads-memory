"""Deterministic fact ids (idempotency under checkpoint replay) and short ids."""

import secrets
import uuid

# Fixed namespace for uuid5 derivation; never change once data exists.
_FACT_NS = uuid.UUID("b3ad5000-0000-4000-8000-000000000000")


def derive_fact_id(namespace_id: uuid.UUID, source_key: str, body: str) -> uuid.UUID:
    """Content-derived fact id: replaying the same capture no-ops on insert."""
    return uuid.uuid5(_FACT_NS, f"{namespace_id}:{source_key}:{body}")


def short_id(fact_id: uuid.UUID) -> str:
    """Beads-style display id, e.g. 'fact-a3f8b2c1'."""
    return f"fact-{fact_id.hex[:8]}"


def random_fork_suffix() -> str:
    """Collision-safe child-namespace segment, e.g. 'sub-a1b2c3d4'."""
    return f"sub-{secrets.token_hex(4)}"
