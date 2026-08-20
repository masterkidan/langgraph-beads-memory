"""BeadsStore: Postgres persistence for namespaces, facts, and edges."""

from __future__ import annotations

import dataclasses
import importlib.resources
import os
import uuid

import psycopg

from .ids import derive_fact_id, derive_namespace_id, random_fork_suffix
from .segment import contested_values, value_tokens

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

# Weight on the CLAIM's own embedding when ranking; the remainder goes to the
# embedding of the text it was carved from (`context_embedding`). 1.0 is the
# claim-only behaviour this library shipped with.
#
# Why it is not 1.0. Splitting a message into one fact per claim is what makes
# `supersedes` surgical, but it also shrinks what each embedding has to match
# on. Measured on the incident scenario at N=3: the root cause was captured
# nine times over, every copy active, and ranked 30th of ~90 — outside the
# top-8 injection in all three runs, while the baseline's un-split document
# ranked 7th of 10 and got used. Retrieval, not generation, was the whole gap.
#
# 0.5 rather than 0.0 deliberately. Pure context ranking makes every claim from
# one capture score identically, which destroys the ordering WITHIN a capture
# and is why `CONTEXT_MAX_PER_SOURCE` exists alongside it.
CLAIM_WEIGHT = float(os.environ.get("BEADS_CLAIM_WEIGHT", "0.5"))

# Ceiling on how many claims from a single capture may occupy the top-k. Blended
# ranking pulls a whole capture's claims up together — that is the point — but
# without a ceiling one verbose answer takes every slot, which is the failure
# the blend was introduced to fix, reintroduced from the other direction.
#
# DEFAULT IS NOW OFF (0), and the reason is measured. This knob was chosen by
# reasoning and never scored. On 3 fixtures per scenario at k=8, verbatim
# queries, turning it off moved incident aspect coverage 50% -> 61% and left
# vecdb bit-identical — it is a loss on one scenario and inert on the other.
# The failure it was written to prevent is real but is handled better by the
# derivation filter below, which drops a restatement because it RESTATES
# something already selected rather than because it shares a context vector.
#
# Kept rather than deleted because the failure mode is genuine: set it to a
# positive value if a deployment sees one capture monopolising the block.
CONTEXT_MAX_PER_SOURCE = int(os.environ.get("BEADS_CONTEXT_MAX_PER_SOURCE", "0"))

# Cosine-similarity floor for cascading a supersede to EARLIER restatements of
# the fact being retired. Paired with a created_at ordering constraint — see
# BeadsStore._cascade_supersede — because similarity alone cannot tell "$100k"
# from "$50k".
#
# DELIBERATELY CONSERVATIVE, and the reason is measured. Two real runs give
# incompatible answers, because a verbatim restatement and a paraphrased one
# score very differently while a merely-topical fact sits between them.
#
#   incident, superseding "deployed at 13:50 UTC":
#     0.973 / 0.972  "Release 2.14 was deployed at 13:50 UTC."      stale
#     0.689          "Synthesis: the issue is the application tier"  MUST KEEP
#
#   vecdb, superseding "the budget is $100k per year":
#     0.720          "The annual budget ... is $100,000."            stale
#     0.500          "Qdrant was evaluated as a ..."                 keep
#
# A threshold low enough to catch vecdb's paraphrase (0.72) would also retire
# incident's root-cause synthesis (0.689) — deleting the finding that lets the
# agent name the cause, which is far worse than leaving a stale value behind.
# 0.85 catches near-verbatim restatements safely and MISSES paraphrased ones.
#
# So this alone is a partial fix, and it is now the *fallback* path. The
# paraphrase gap is closed by asking a question cosine distance cannot answer —
# which value does this fact actually assert — see _is_stale_restatement.
CASCADE_MIN_SIMILARITY = float(os.environ.get("BEADS_CASCADE_MIN_SIMILARITY", "0.85"))

# Topical floor for the value-aware path. Much lower than the verbatim
# threshold, because the discriminating work is done by the contested-value test
# rather than by similarity: this only has to exclude a fact from an unrelated
# discussion that happens to mention the same figure. A fact carrying a
# `derived_from` edge to the superseded one skips this entirely — recorded
# provenance is better evidence than a distance.
CASCADE_TOPICAL_SIMILARITY = float(os.environ.get("BEADS_CASCADE_TOPICAL", "0.45"))


def _is_stale_restatement(
    body: str,
    similarity: float,
    derived: bool,
    stale_values: set[str],
    new_values: set[str],
) -> bool:
    """Does this earlier fact still assert the value that was just corrected?

    The measured problem: a stale "$100,000" and its own "$50,000" correction
    reached the model two ranks apart, 0.002 of cosine distance from each other.
    Similarity cannot separate them, because the two sentences differ in exactly
    the token embeddings are worst at — the number. So it is read directly.

    A fact is retired when it carries a contested stale value and does NOT carry
    the replacement. Carrying both means it is the corrected version restating
    what it supersedes ("deployed at 13:20 UTC, the 13:50 timestamp was the
    canary promotion"), which must survive.

    Verified against the two runs the cascade exists for:

        "Release 2.14 was deployed at 13:50 UTC."          13:50, no 13:20  RETIRE
        "Incident timeline: ... 13:50 UTC; p99 180ms ..."  13:50, no 13:20  RETIRE
        "... 13:20 UTC (13:50 was the canary promotion)"   has 13:20        keep
        "Synthesis: the issue is the application tier"     no time at all   keep
        "The annual budget ... is $100,000."               100k, no 50k     RETIRE
        "Qdrant was evaluated as a managed service"        no figure        keep

    The last two lines are what the old 0.85 threshold could not reach: 0.720
    and 0.500 are too close together to split, and they no longer have to be.
    """
    if similarity >= CASCADE_MIN_SIMILARITY:
        return True  # near-verbatim, no figures needed
    if not stale_values:
        return False
    tokens = value_tokens(body)
    if not tokens & stale_values or tokens & new_values:
        return False
    return derived or similarity >= CASCADE_TOPICAL_SIMILARITY


# Cycle guard for the supersedes walk in `search`. The relation is asserted by
# agents and by the cascade, and nothing forbids A superseding B superseding A;
# without a bound the recursive resolution would not terminate.
SUPERSEDE_MAX_DEPTH = int(os.environ.get("BEADS_SUPERSEDE_MAX_DEPTH", "10"))


# Most slots any ONE sub-agent may take in its parent's top-k, and how many
# extra rows to fetch so the cap has something to promote in their place.
#
# Delegation exists to buy breadth, and a purely similarity-ordered ranker does
# not protect it. This used to hold by accident: a sub-agent crossed into its
# parent as exactly one summary fact, so it could occupy exactly one slot.
# Splitting summaries per claim removed the accident and the loss was immediate
# — in the final synthesis turn of an incident run, the strongest-scoring
# researcher took two descendant slots in all four model calls and the other two
# researchers appeared in none, so the parent never saw the app-tier or network
# findings at all. Breadth went 3 subsystems to 2 and the surviving cause went
# unnamed.
#
# 2 rather than 1: a sub-agent should be able to contribute a finding plus the
# detail that supports it. The cap applies only to descendants — an agent's own
# facts and its ancestors' are not rationed.
DESCENDANT_MAX_PER_AGENT = int(os.environ.get("BEADS_DESCENDANT_MAX_PER_AGENT", "2"))
_OVERFETCH = 4


def _select_top_k(rows: list, k: int, derived_parents: dict | None = None) -> list:
    """Take the best k rows, subject to three constraints.

    Rows arrive in rank order, so this preserves ranking within the constraints:
    a skipped row's slot goes to the next-best row.

    The three ration different things. The per-agent cap rations VOICES, so
    three researchers each get a hearing. The per-source cap rations one
    CAPTURE (off by default — see CONTEXT_MAX_PER_SOURCE). The derivation
    filter rations RESTATEMENT, and is the only one of the three that reads
    recorded provenance rather than inferring redundancy from a vector.
    """
    derived_parents = derived_parents or {}
    kept, kept_ids, per_agent, per_source = [], set(), {}, {}
    for row in rows:
        fact_id, source, agent_id, demoted = row[0], row[6], row[7], row[10]
        # A fact DERIVED FROM something already selected adds nothing: it was
        # written by a model that had those facts in front of it. Measured on 3
        # fixtures per scenario — vecdb reached full aspect coverage at 888
        # chars against the baseline's 1338, and stopped growing past k=8
        # because the filter runs out of non-redundant candidates.
        #
        # ONE DIRECTION ONLY, and the other direction was measured and rejected.
        # Also skipping a candidate because a SELECTED fact derives from IT
        # reads as "the conclusion supersedes its premise", and it evicted
        # "The selected vector database must be self-hostable." — a user
        # constraint — because the agent's conclusion built on it ranked higher.
        # Forward is redundancy; backward is a preference for conclusions over
        # premises wearing a redundancy costume.
        if derived_parents.get(fact_id, ()) and derived_parents[fact_id] & kept_ids:
            continue
        # Keyed on the context vector: every claim from one capture shares it,
        # and it is null for facts with no wider context — which must NOT all
        # collide into a single group, so those bypass the cap entirely.
        ctx = row[11]
        if CONTEXT_MAX_PER_SOURCE and ctx is not None:
            seen_src = per_source.get(ctx, 0)
            if seen_src >= CONTEXT_MAX_PER_SOURCE:
                continue
            per_source[ctx] = seen_src + 1
        # Keyed on agent_id, and namespace is NOT usable here. `conclude_task`
        # writes a sub-agent's summary into the PARENT namespace tagged with the
        # child's agent_id, so a summary is not a descendant row at all — a
        # namespace-keyed cap misses exactly the facts that need rationing, and
        # measurably did: capping on namespace left three fragments from one
        # researcher in a single injection, unchanged.
        #
        # Restricted to delegated contributions so the agent's OWN conclusions
        # are never rationed. Those are the working context, not one voice among
        # several competing for the floor.
        if demoted or source in ("conclude_task", "fallback_conclude"):
            seen = per_agent.get(agent_id, 0)
            if seen >= DESCENDANT_MAX_PER_AGENT:
                continue
            per_agent[agent_id] = seen + 1
        kept.append(row)
        kept_ids.add(fact_id)
        if len(kept) == k:
            break
    return kept


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
            # Same reason as `search`: `id` is not a stable tiebreak, because a
            # sub-agent's namespace id — and therefore every fact id under it —
            # is reseeded on every run.
            " ORDER BY created_at DESC, md5(body) LIMIT %s",
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
        context_embedding: list[float] | None = None,
    ) -> Fact:
        """Write one claim.

        `context_embedding` is the embedding of the wider text this claim came
        from — the whole user message, the whole answer, the whole tool body —
        and is shared by every claim from that capture. Callers embed it once
        per capture, not once per claim; see middleware, where a message
        splitting into six facts costs two embed calls rather than seven.
        """
        fid = derive_fact_id(namespace.session_id, namespace.id, source, source_key, body)
        self._conn.execute(
            """
            INSERT INTO facts (id, namespace_id, session_id, kind, body, embedding,
                               context_embedding, source, agent_id, acting_on_behalf_of)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (
                fid,
                namespace.id,
                namespace.session_id,
                kind,
                body,
                embedding,
                context_embedding,
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
            self._cascade_supersede(from_fact_id, to_fact_id)
        return True

    def _cascade_supersede(self, from_fact_id: uuid.UUID, to_fact_id: uuid.UUID) -> int:
        """Retire earlier restatements of the fact just superseded.

        MEASURED FAILURE this exists for. A user corrected a budget from $100k
        to $50k. The correction fired and the original was retired — and the
        answer still cited $100k, because four *derived* claims were still
        active:

            [active] The annual budget for the vector database is $100,000.
            [active] I have recorded your requirements: a $100,000 annual budget...
            [active] pgvector ... fits well within the budget
            [active] All three ... fall within the $100,000 annual budget

        A `supersedes` edge retired the row it pointed at and nothing else, so
        both values were live at once and the ranker surfaced the stale one.

        Two conditions, and the second is what makes this safe. Similarity alone
        would also retire the *new* value: embeddings are poor at distinguishing
        $100k from $50k, so "the annual budget is $50,000" scores highly against
        "the budget is $100k per year". Requiring the candidate to predate the
        superseding fact separates them — a restatement of the stale value must
        have been written before the correction arrived.

        The threshold is deliberately stricter than the guard on `supersedes`
        itself (0.55): that guard only has to tell "related" from "unrelated",
        while this decides to retire a fact nobody pointed at.

        Scoped to the same namespace. Fails closed — no embedding, no cascade —
        because the cost of wrongly retiring a live fact is higher than the cost
        of leaving a stale one.
        """
        bodies = dict(
            self._conn.execute(
                "SELECT id, body FROM facts WHERE id IN (%s, %s)", (from_fact_id, to_fact_id)
            ).fetchall()
        )
        stale_values, new_values = contested_values(
            bodies.get(to_fact_id, ""), bodies.get(from_fact_id, "")
        )

        candidates = self._conn.execute(
            """
            WITH target AS (SELECT embedding, namespace_id FROM facts WHERE id = %s),
                 newer  AS (SELECT created_at FROM facts WHERE id = %s)
            SELECT f.id,
                   f.body,
                   1 - (f.embedding <=> (SELECT embedding FROM target)) AS similarity,
                   EXISTS (
                       SELECT 1 FROM fact_edges e
                       WHERE e.from_fact_id = f.id
                         AND e.to_fact_id = %s
                         AND e.relation = 'derived_from'
                   ) AS derived
            FROM facts f
            WHERE f.status = 'active'
              AND f.id <> %s
              AND f.namespace_id = (SELECT namespace_id FROM target)
              AND f.created_at < (SELECT created_at FROM newer)
              AND f.embedding IS NOT NULL
              AND (SELECT embedding FROM target) IS NOT NULL
            """,
            (to_fact_id, from_fact_id, to_fact_id, from_fact_id),
        ).fetchall()

        rows = [
            (fact_id,)
            for fact_id, body, similarity, derived in candidates
            if _is_stale_restatement(body, similarity, derived, stale_values, new_values)
        ]
        for (stale_id,) in rows:
            self._conn.execute("UPDATE facts SET status='superseded' WHERE id=%s", (stale_id,))
            # Record why it was retired, so the audit trail shows a cascade
            # rather than an unexplained status change.
            self._conn.execute(
                """
                INSERT INTO fact_edges (id, from_fact_id, to_fact_id, relation)
                VALUES (%s, %s, %s, 'supersedes')
                ON CONFLICT (from_fact_id, to_fact_id, relation) DO NOTHING
                """,
                (uuid.uuid4(), from_fact_id, stale_id),
            )
        return len(rows)

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
        with_scores: bool = False,
    ) -> list[Fact] | list[tuple[Fact, float, bool]]:
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
            -- Resolve every fact to the HEAD of its supersedes chain.
            --
            -- WHY RESOLUTION AND NOT A PENALTY. Demoting superseded facts was
            -- implemented, measured, and was a literal no-op: on 78 LongMemEval
            -- knowledge-update questions it produced byte-identical answers to
            -- excluding them, because a fact that has been corrected sits just
            -- behind its own correction and any penalty large enough to matter
            -- is indistinguishable from a filter. Removing a stale value leaves
            -- the slot EMPTY; the measured failure was answering "you currently
            -- have three bikes" when the user owns 4, and an empty slot does not
            -- fix that. Resolution REPLACES the match with the current value, so
            -- landing on any version of a claim returns the one that is true.
            --
            -- Ranking uses the BEST distance across all versions (min below), so
            -- an old phrasing that happens to match the query well still pulls
            -- its head in. Versions collapse to one row, which is also why this
            -- costs nothing in context: k slots hold k SUBJECTS, not k facts.
            WITH RECURSIVE walk(fact_id, cur_id, depth) AS (
                SELECT f.id, f.id, 0 FROM facts f
                UNION ALL
                SELECT w.fact_id, e.from_fact_id, w.depth + 1
                FROM walk w
                JOIN fact_edges e ON e.to_fact_id = w.cur_id
                                 AND e.relation = 'supersedes'
                -- Bounded because `supersedes` is asserted by agents and by the
                -- cascade, and nothing forbids A superseding B superseding A.
                WHERE w.depth < %s
            ),
            -- Deepest wins: the end of the longest correction chain is the
            -- current value. created_at breaks a DAG merge, where two facts
            -- independently supersede the same one.
            head_of AS (
                SELECT DISTINCT ON (w.fact_id) w.fact_id, w.cur_id AS head_id
                FROM walk w JOIN facts hf ON hf.id = w.cur_id
                ORDER BY w.fact_id, w.depth DESC, hf.created_at DESC, md5(hf.body)
            ),
            scored AS (
                SELECT h.head_id,
                       %s * (f.embedding <=> %s::vector)
                       + (1 - %s) * (COALESCE(f.context_embedding, f.embedding)
                                     <=> %s::vector)
                       + CASE WHEN f.namespace_id = ANY(%s) THEN %s ELSE 0 END
                       AS s,
                       (f.embedding <=> %s::vector) AS raw
                FROM facts f
                JOIN head_of h ON h.fact_id = f.id
                WHERE f.namespace_id = ANY(%s)
                  AND f.status <> 'archived'
                  AND (%s OR f.kind <> 'directive')
                  AND f.embedding IS NOT NULL
                  AND NOT (f.id = ANY(%s))
            ),
            best AS (
                SELECT head_id, min(s) AS s, min(raw) AS raw
                FROM scored GROUP BY head_id
            )
            SELECT hf.id, hf.namespace_id, hf.session_id, hf.kind, hf.body,
                   hf.status, hf.source, hf.agent_id, hf.acting_on_behalf_of,
                   b.raw AS distance,
                   (hf.namespace_id = ANY(%s)) AS demoted,
                   hf.context_embedding::text
            FROM best b
            JOIN facts hf ON hf.id = b.head_id
            -- The head must itself be readable and live: a chain may end in a
            -- namespace this caller cannot see, or in an archived fact, and
            -- neither may be surfaced just because an old version matched.
            WHERE hf.namespace_id = ANY(%s)
              AND hf.status <> 'archived'
              AND NOT (hf.id = ANY(%s))
            -- Deterministic tiebreak. Without one, near-ties resolved to
            -- whatever physical order Postgres returned, which follows INSERT
            -- order — and insert order is not stable across runs, because
            -- sub-agents race on ToolNode's thread pool and each fork draws a
            -- random suffix. Measured: at temperature 0 every BASELINE cell
            -- scored identically three times while every treatment cell had
            -- spread 1-2, and on one turn only 4 of 8 injected facts were common
            -- to all three runs. `md5(body)`, NOT `id`, because a fact's id
            -- derives from its namespace id and sub-agent namespaces are
            -- reseeded every run.
            ORDER BY b.s, md5(hf.body)
            LIMIT %s
            """,
            (
                SUPERSEDE_MAX_DEPTH,
                CLAIM_WEIGHT,
                query_embedding,
                CLAIM_WEIGHT,
                query_embedding,
                descendants,
                DESCENDANT_RANK_PENALTY,
                query_embedding,
                chain + descendants,
                include_directives,
                exclude_ids or [],
                descendants,
                chain + descendants,
                exclude_ids or [],
                # Over-fetch so the per-agent cap below has alternatives to
                # promote; without spare rows the cap would just shrink k.
                k * _OVERFETCH,
            ),
        ).fetchall()
        rows = _select_top_k(rows, k, self._derived_parents([r[0] for r in rows]))
        if with_scores:
            # (fact, raw cosine distance, whether the descendant penalty applied)
            # Returned so a caller can log WHY a fact was chosen. Reconstructing
            # this after the fact meant re-running the query by hand, which is
            # how an earlier analysis ranked across three runs' facts at once and
            # reported a rank that did not exist.
            return [(Fact(*r[:9]), float(r[9]), bool(r[10])) for r in rows]
        return [Fact(*r[:9]) for r in rows]

    def _derived_parents(self, fact_ids: list[uuid.UUID]) -> dict[uuid.UUID, set[uuid.UUID]]:
        """{fact_id: facts it was derived from}, restricted to the candidate set.

        Scoped to the candidates rather than fetched per fact: the filter only
        ever asks "is a parent of this row also in this row's competition?", so
        edges pointing outside the fetched window cannot change the answer. One
        query per search, over k*_OVERFETCH ids.
        """
        if not fact_ids:
            return {}
        out: dict[uuid.UUID, set[uuid.UUID]] = {}
        for frm, to in self._conn.execute(
            "SELECT from_fact_id, to_fact_id FROM fact_edges"
            " WHERE relation = 'derived_from' AND from_fact_id = ANY(%s)",
            (fact_ids,),
        ).fetchall():
            out.setdefault(frm, set()).add(to)
        return out

    def neighbours(
        self, fact_ids: list[uuid.UUID], readable_ns_ids: list[uuid.UUID]
    ) -> dict[uuid.UUID, list[tuple[Fact, str, str]]]:
        """Distance-1 graph neighbours of each fact, as {id: [(fact, relation, direction)]}.

        Edges are followed in BOTH directions and the direction is returned, not
        discarded: `supersedes` reads in exactly opposite senses depending on
        which end you are standing at, and a renderer that collapsed the two
        would show a retired value as though it were the correction.

        Scoped to namespaces the caller can already read, so expansion cannot
        become a way around the sibling isolation that `search` enforces —
        otherwise one hit on a shared ancestor would pull a sibling's private
        exploration in behind it.

        Superseded and archived facts are included, unlike in `search`. Here
        that is the point: "this replaced the $100k figure" is the most useful
        thing to say about a $50k fact, and the retired value is labelled by its
        relation rather than presented as current.
        """
        if not fact_ids:
            return {}
        rows = self._conn.execute(
            """
            SELECT e.from_fact_id, e.to_fact_id, e.relation,
                   f.id, f.namespace_id, f.session_id, f.kind, f.body, f.status,
                   f.source, f.agent_id, f.acting_on_behalf_of,
                   (e.from_fact_id = ANY(%s)) AS seed_is_source
            FROM fact_edges e
            JOIN facts f ON f.id = CASE WHEN e.from_fact_id = ANY(%s)
                                        THEN e.to_fact_id ELSE e.from_fact_id END
            WHERE (e.from_fact_id = ANY(%s) OR e.to_fact_id = ANY(%s))
              AND f.namespace_id = ANY(%s)
            """,
            (fact_ids, fact_ids, fact_ids, fact_ids, readable_ns_ids),
        ).fetchall()
        out: dict[uuid.UUID, list[tuple[Fact, str, str]]] = {}
        for r in rows:
            seed = r[0] if r[12] else r[1]
            fact = Fact(r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], r[11])
            out.setdefault(seed, []).append((fact, r[2], "out" if r[12] else "in"))
        return out

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
