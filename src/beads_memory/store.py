"""BeadsStore: Postgres persistence for namespaces, facts, and edges."""

from __future__ import annotations

import dataclasses
import importlib.resources
import uuid

import psycopg

from .ids import random_fork_suffix


@dataclasses.dataclass(frozen=True)
class Namespace:
    id: uuid.UUID
    session_id: str
    extra_path: list[str]
    parent_id: uuid.UUID | None


@dataclasses.dataclass(frozen=True)
class Fact:
    id: uuid.UUID
    namespace_id: uuid.UUID
    session_id: str
    kind: str
    body: str
    status: str
    source: str
    agent_id: str
    acting_on_behalf_of: str


class BeadsStore:
    def __init__(self, conn: psycopg.Connection):
        self._conn = conn

    def init_schema(self) -> None:
        ddl = importlib.resources.files("beads_memory").joinpath("schema.sql").read_text()
        self._conn.execute(ddl)

    def get_or_create_namespace(self, session_id: str) -> Namespace:
        return self._upsert_namespace(session_id, [], None)

    def fork_namespace(self, parent: Namespace) -> Namespace:
        extra = parent.extra_path + ["task", random_fork_suffix()]
        return self._upsert_namespace(parent.session_id, extra, parent.id)

    def _upsert_namespace(
        self, session_id: str, extra_path: list[str], parent_id: uuid.UUID | None
    ) -> Namespace:
        row = self._conn.execute(
            """
            INSERT INTO namespaces (id, session_id, extra_path, parent_id)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (session_id, extra_path) DO UPDATE SET session_id = EXCLUDED.session_id
            RETURNING id, session_id, extra_path, parent_id
            """,
            (uuid.uuid4(), session_id, extra_path, parent_id),
        ).fetchone()
        return Namespace(id=row[0], session_id=row[1], extra_path=list(row[2]), parent_id=row[3])

    def ancestor_chain(self, namespace_id: uuid.UUID) -> list[uuid.UUID]:
        rows = self._conn.execute(
            """
            WITH RECURSIVE chain AS (
                SELECT id, parent_id, 0 AS depth FROM namespaces WHERE id = %s
                UNION ALL
                SELECT n.id, n.parent_id, c.depth + 1
                FROM namespaces n JOIN chain c ON n.id = c.parent_id
            )
            SELECT id FROM chain ORDER BY depth
            """,
            (namespace_id,),
        ).fetchall()
        return [r[0] for r in rows]
