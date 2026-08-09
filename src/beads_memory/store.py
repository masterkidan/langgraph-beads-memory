"""BeadsStore: Postgres persistence for namespaces, facts, and edges."""

from __future__ import annotations

import dataclasses
import importlib.resources
import uuid

import psycopg

from .ids import derive_fact_id, random_fork_suffix


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

    def write_fact(
        self,
        namespace: Namespace,
        *,
        kind: str,
        body: str,
        source: str,
        source_key: str,
        agent_id: str,
        acting_on_behalf_of: str,
        embedding: list[float] | None = None,
    ) -> Fact:
        fid = derive_fact_id(namespace.id, source, source_key, body)
        self._conn.execute(
            """
            INSERT INTO facts (id, namespace_id, session_id, kind, body, embedding,
                               source, agent_id, acting_on_behalf_of)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                fid,
                namespace.id,
                namespace.session_id,
                kind,
                body,
                embedding,
                source,
                agent_id,
                acting_on_behalf_of,
            ),
        )
        row = self._conn.execute(
            "SELECT id, namespace_id, session_id, kind, body, status, source,"
            " agent_id, acting_on_behalf_of FROM facts WHERE id=%s",
            (fid,),
        ).fetchone()
        return Fact(*row)

    def add_edge(self, from_fact_id: uuid.UUID, to_fact_id: uuid.UUID, relation: str) -> None:
        self._conn.execute(
            """
            INSERT INTO fact_edges (id, from_fact_id, to_fact_id, relation)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (from_fact_id, to_fact_id, relation) DO NOTHING
            """,
            (uuid.uuid4(), from_fact_id, to_fact_id, relation),
        )
        if relation == "supersedes":
            # The only path that sets status='superseded' (design spec).
            self._conn.execute("UPDATE facts SET status='superseded' WHERE id=%s", (to_fact_id,))

    def facts_in_namespace(self, namespace_id: uuid.UUID) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT id, namespace_id, session_id, kind, body, status, source,"
            " agent_id, acting_on_behalf_of FROM facts WHERE namespace_id=%s"
            " ORDER BY created_at, id",
            (namespace_id,),
        ).fetchall()
        return [Fact(*r) for r in rows]

    def search(
        self,
        namespace_id: uuid.UUID,
        query_embedding: list[float],
        *,
        k: int = 8,
        exclude_ids: list[uuid.UUID] | None = None,
    ) -> list[Fact]:
        """Top-k active facts by cosine similarity across self + ancestor chain.
        Facts without embeddings are invisible to search."""
        chain = self.ancestor_chain(namespace_id)
        rows = self._conn.execute(
            """
            SELECT id, namespace_id, session_id, kind, body, status, source,
                   agent_id, acting_on_behalf_of
            FROM facts
            WHERE namespace_id = ANY(%s)
              AND status = 'active'
              AND embedding IS NOT NULL
              AND NOT (id = ANY(%s))
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (chain, exclude_ids or [], query_embedding, k),
        ).fetchall()
        return [Fact(*r) for r in rows]

    def resolve_short_id(self, prefix: str, readable_ns_ids: list[uuid.UUID]) -> Fact:
        """Resolve 'fact-a3f8b2c1' within readable scope; raise on miss/ambiguity."""
        hexpref = prefix.removeprefix("fact-")
        rows = self._conn.execute(
            "SELECT id, namespace_id, session_id, kind, body, status, source,"
            " agent_id, acting_on_behalf_of FROM facts"
            " WHERE namespace_id = ANY(%s) AND id::text LIKE %s",
            (readable_ns_ids, _hex_like(hexpref)),
        ).fetchall()
        if len(rows) != 1:
            raise LookupError(f"short id {prefix!r} matched {len(rows)} facts")
        return Fact(*rows[0])


def _hex_like(hexpref: str) -> str:
    """uuid::text is 8-4-4-4-12 with dashes; first 8 hex chars are the first group."""
    h = hexpref.lower()
    return f"{h[:8]}%"
