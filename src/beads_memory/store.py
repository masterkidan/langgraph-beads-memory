"""BeadsStore: Postgres persistence for namespaces, facts, and edges."""

from __future__ import annotations

import dataclasses
import importlib.resources
import os
import uuid

import psycopg

from .ids import derive_fact_id, derive_namespace_id, random_fork_suffix

# Cosine-similarity floor for a `supersedes` edge. Chosen from measurement, not
# taste: on real run data the legitimate budget revision scored 0.730 against
# its target while every spurious edge scored at most 0.472, so anything in that
# gap separates them. 0.55 sits near the middle.
SUPERSEDE_MIN_SIMILARITY = float(os.environ.get("BEADS_SUPERSEDE_MIN_SIMILARITY", "0.55"))

# Cosine-distance penalty added to facts from a descendant namespace, so a
# sub-agent's raw exploration is reachable but never competes with the parent's
# own facts on equal terms. Isolation was protecting the parent's working
# context from noise — a ranking concern, which gets a ranking answer rather
# than a wall. Large enough that loosely-related child chatter loses to a
# directly relevant parent fact; small enough that a pointed question ("what was
# that memory optimization?") still reaches a strongly-matching child fact.
DESCENDANT_RANK_PENALTY = float(os.environ.get("BEADS_DESCENDANT_PENALTY", "0.15"))


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
    def __init__(self, conn: psycopg.Connection, *, retire_superseded: bool = True):
        """`retire_superseded=False` is an ablation switch, not a feature.

        With it off, a `supersedes` edge is still recorded — the provenance is
        intact and the graph still says which fact replaced which — but the
        target keeps `status='active'` and stays retrievable. That isolates
        *typed invalidation* from everything else the fact graph does, so a
        benchmark arm can answer "is retiring the stale fact what carries the
        result, or is it the per-claim granularity?" Nothing in the library
        sets this; only the demo's ablation arm does.
        """
        self._conn = conn
        self._retire_superseded = retire_superseded

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
            (derive_namespace_id(session_id, extra_path), session_id, extra_path, parent_id),
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

    def descendant_scope(self, namespace_id: uuid.UUID) -> list[uuid.UUID]:
        """Namespaces below this one. Parents may read their whole subtree.

        Deliberately asymmetric with `ancestor_chain`: a child reads only itself
        and its ancestors, so siblings still cannot see each other. Descendant
        visibility is a parent privilege, not a general relaxation.
        """
        rows = self._conn.execute(
            """
            WITH RECURSIVE sub AS (
                SELECT id FROM namespaces WHERE parent_id = %s
                UNION ALL
                SELECT n.id FROM namespaces n JOIN sub ON n.parent_id = sub.id
            )
            SELECT id FROM sub
            """,
            (namespace_id,),
        ).fetchall()
        return [r[0] for r in rows]

    def children(self, namespace_id: uuid.UUID) -> list[Namespace]:
        """Direct sub-namespaces. One level, not the whole subtree.

        Lets an orchestrator enumerate what it delegated to, which demoted
        descendant *search* cannot do — that is similarity-driven and only
        surfaces a child fact when the query happens to match it.
        """
        rows = self._conn.execute(
            "SELECT id, session_id, extra_path, parent_id FROM namespaces"
            " WHERE parent_id = %s ORDER BY created_at",
            (namespace_id,),
        ).fetchall()
        return [
            Namespace(id=r[0], session_id=r[1], extra_path=list(r[2]), parent_id=r[3]) for r in rows
        ]

    def subtree_facts(
        self,
        namespace_id: uuid.UUID,
        *,
        agent_id: str | None = None,
        limit: int = 50,
    ) -> list[Fact]:
        """Active facts beneath this namespace, newest first. Excludes self.

        Deliberate drill-down rather than a similarity guess: an orchestrator
        that delegated Qdrant can ask what that researcher recorded, instead of
        hoping the embedding of "what was that memory optimization?" lands near
        it. The measured failure had exactly this shape — the fact existed in a
        child and the parent never reached it.

        Retired facts are excluded, matching `search`: a superseded value should
        not resurface through a side door.
        """
        scope = self.descendant_scope(namespace_id)
        if not scope:
            return []
        rows = self._conn.execute(
            "SELECT id, namespace_id, session_id, kind, body, status, source,"
            " agent_id, acting_on_behalf_of FROM facts"
            " WHERE namespace_id = ANY(%s) AND status = 'active'"
            "   AND (%s::text IS NULL OR agent_id = %s)"
            " ORDER BY created_at DESC, id LIMIT %s",
            (scope, agent_id, agent_id, limit),
        ).fetchall()
        return [Fact(*r) for r in rows]

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
        fid = derive_fact_id(namespace.session_id, namespace.id, source, source_key, body)
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

    def add_edge(self, from_fact_id: uuid.UUID, to_fact_id: uuid.UUID, relation: str) -> bool:
        """Create a typed edge. Returns False if a `supersedes` edge was refused.

        `supersedes` is guarded because it is the only relation that retires a
        fact from retrieval, and an agent can aim it at anything it can see. In
        a real run "The investigation into Weaviate has been completed" retired
        the user's stated constraints; nothing checked that the two facts were
        about the same thing.

        The guard requires cosine similarity above SUPERSEDE_MIN_SIMILARITY.
        This only became viable once messages were split into one fact per
        claim: against a bundled fact, measured spurious edges scored HIGHER
        (0.63-0.74) than legitimate revisions (0.45-0.68), so a threshold would
        have blocked the right edges. Post-split the separation is clean —
        legitimate 0.73, spurious at most 0.47.

        Fails open when either fact lacks an embedding, so a fact written before
        embeddings existed never becomes permanently un-supersedable.
        """
        if relation == "supersedes" and not self._may_supersede(from_fact_id, to_fact_id):
            return False
        self._conn.execute(
            """
            INSERT INTO fact_edges (id, from_fact_id, to_fact_id, relation)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (from_fact_id, to_fact_id, relation) DO NOTHING
            """,
            (uuid.uuid4(), from_fact_id, to_fact_id, relation),
        )
        if relation == "supersedes" and self._retire_superseded:
            # The only path that sets status='superseded' (design spec).
            self._conn.execute("UPDATE facts SET status='superseded' WHERE id=%s", (to_fact_id,))
        return True

    def _may_supersede(self, from_fact_id: uuid.UUID, to_fact_id: uuid.UUID) -> bool:
        row = self._conn.execute(
            """
            SELECT 1 - (a.embedding <=> b.embedding)
            FROM facts a, facts b
            WHERE a.id = %s AND b.id = %s
              AND a.embedding IS NOT NULL AND b.embedding IS NOT NULL
            """,
            (from_fact_id, to_fact_id),
        ).fetchone()
        if row is None:  # missing embedding on either side -> fail open
            return True
        return float(row[0]) >= SUPERSEDE_MIN_SIMILARITY

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
        include_directives: bool = False,
    ) -> list[Fact]:
        """Top-k active facts by cosine similarity across self + ancestor chain.

        Scope is self + ancestors + descendants. Descendant facts carry a rank
        penalty (DESCENDANT_RANK_PENALTY) so a sub-agent's raw exploration is
        reachable without competing on equal terms — a specific question can
        find a specific detail, but child chatter cannot displace the parent's
        own constraints. Siblings remain mutually invisible, since a child's
        scope is still only itself plus its ancestors.

        Facts without embeddings are invisible to search. `directive` facts —
        questions, instructions, stated goals — are excluded by default: they
        rank highly against a query precisely because they resemble it, and in a
        measured run four of eight injected slots were question fragments that
        displaced the constraint the answer needed. They remain stored and
        queryable; pass include_directives=True to retrieve them.
        """
        chain = self.ancestor_chain(namespace_id)
        descendants = self.descendant_scope(namespace_id)
        rows = self._conn.execute(
            """
            SELECT id, namespace_id, session_id, kind, body, status, source,
                   agent_id, acting_on_behalf_of
            FROM facts
            WHERE namespace_id = ANY(%s)
              AND status = 'active'
              AND (%s OR kind <> 'directive')
              AND embedding IS NOT NULL
              AND NOT (id = ANY(%s))
            ORDER BY (embedding <=> %s::vector)
                     + CASE WHEN namespace_id = ANY(%s) THEN %s ELSE 0 END
            LIMIT %s
            """,
            (
                chain + descendants,
                include_directives,
                exclude_ids or [],
                query_embedding,
                descendants,
                DESCENDANT_RANK_PENALTY,
                k,
            ),
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
