"""Freeze both arms' stores — bodies AND embeddings — into one JSON fixture.

Why this exists. The end-to-end harness answers "did the agent get the right
answer?", which is a question about retrieval *and* generation at once. When
gemma4:12b scored 58% on treatment against 88% on baseline (N=3, incident), the
end-to-end numbers could not say which half failed. Reading one injection log
could: all eight slots at conv-3 held constraints, and none held the
fraud-scoring root cause the answer needed.

That is a ranking question, and ranking does not need an LLM. Embeddings are
already computed and sitting in Postgres; the corpus is already written. So
this captures both stores once, and `demo.retrieval_eval` scores ranking
against them offline, deterministically, in under a second — no generation, no
temperature, no wedged Ollama, no 30-minute run.

    uv run python -m demo.retrieval_fixture \
        --schema run_incident_treatment_1_fb6dbb \
        --baseline-prefix memories.baseline-run1-24535a \
        --out results/fixtures/incident-run1.json

The two arms store fundamentally different things, and the fixture keeps that
difference rather than flattening it — that asymmetry is the thing under test:

  baseline   N whole documents, each a self-contained paragraph
  treatment  M per-claim facts, each separately embedded, plus the namespace
             tree, because treatment ranking depends on ancestor/descendant
             scope and a flat fact list cannot reproduce it.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import psycopg

DSN = "postgresql://beads:beads@localhost:5433/beads"


def _vec(raw: str | None) -> list[float] | None:
    """pgvector renders as '[0.1,0.2,...]'. Parsed as text rather than via the
    pgvector adapter so this module has no dependency on the registration the
    production connection does — the fixture is meant to outlive the schema it
    came from."""
    if raw is None:
        return None
    return [float(x) for x in raw.strip("[]").split(",")]


def capture_treatment(conn: psycopg.Connection, schema: str) -> dict:
    """Every fact in a run schema, plus the namespace tree.

    Facts WITHOUT embeddings are captured too, flagged by a null embedding.
    They are invisible to search by design, and a fixture that silently dropped
    them would hide a whole class of failure — "the fact was never retrievable"
    reads identically to "the fact ranked badly" once it is missing.
    """
    ns = [
        {
            "id": str(r[0]),
            "extra_path": list(r[2]) if r[2] else [],
            "parent_id": str(r[3]) if r[3] else None,
        }
        for r in conn.execute(
            f'SELECT id, session_id, extra_path, parent_id FROM "{schema}".namespaces'
        ).fetchall()
    ]
    # Older run schemas predate `context_embedding`. Probing for it keeps this
    # tool able to read every run in results/, which is the whole point of a
    # fixture — a capture tool that only reads the current schema cannot
    # compare a change against the runs made before it.
    has_ctx = bool(
        conn.execute(
            "SELECT 1 FROM information_schema.columns WHERE table_schema = %s"
            " AND table_name = 'facts' AND column_name = 'context_embedding'",
            (schema,),
        ).fetchone()
    )
    ctx_col = "context_embedding::text" if has_ctx else "NULL::text"
    facts = [
        {
            "id": str(r[0]),
            "namespace_id": str(r[1]),
            "kind": r[2],
            "body": r[3],
            "status": r[4],
            "source": r[5],
            "agent_id": r[6],
            "embedding": _vec(r[7]),
            # Null for a capture that produced a single claim, where the wider
            # context IS the claim. Kept as null rather than back-filled with a
            # copy so the per-source cap can tell "no context" from "shares a
            # context with others" — collapsing the two would put every
            # context-less fact into one group and ration them against each
            # other for no reason.
            "context_embedding": _vec(r[8]) if len(r) > 8 else None,
        }
        for r in conn.execute(
            f'SELECT id, namespace_id, kind, body, status, source, agent_id,'
            f' embedding::text, {ctx_col}'
            f' FROM "{schema}".facts ORDER BY created_at, id'
        ).fetchall()
    ]
    # Edges, so neighbour expansion can be evaluated offline like everything
    # else. Captured for both directions; the traversal decides which to follow.
    edges = [
        {"from": str(r[0]), "to": str(r[1]), "relation": r[2]}
        for r in conn.execute(
            f'SELECT from_fact_id, to_fact_id, relation FROM "{schema}".fact_edges'
        ).fetchall()
    ]
    return {"schema": schema, "namespaces": ns, "facts": facts, "edges": edges,
            "has_context_embedding": has_ctx}


def capture_baseline(conn: psycopg.Connection, prefix: str) -> dict:
    """Every LangMem document under one run's namespace, with its vector.

    LEFT JOIN, not INNER: a document whose embedding never landed is still a
    document the baseline stored, and counting it as absent would flatter the
    baseline's recall by shrinking its denominator.
    """
    docs = [
        {
            "key": r[0],
            "content": (r[1] or {}).get("content") if isinstance(r[1], dict) else str(r[1]),
            "embedding": _vec(r[2]),
        }
        for r in conn.execute(
            "SELECT s.key, s.value, v.embedding::text FROM public.store s"
            " LEFT JOIN public.store_vectors v ON v.prefix = s.prefix AND v.key = s.key"
            " WHERE s.prefix = %s ORDER BY s.created_at",
            (prefix,),
        ).fetchall()
    ]
    return {"prefix": prefix, "documents": docs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True, help="treatment run schema in Postgres")
    ap.add_argument("--baseline-prefix", required=True, help="LangMem store prefix for the pair")
    ap.add_argument("--scenario", default="incident")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conn = psycopg.connect(DSN, autocommit=True)
    fixture = {
        "scenario": args.scenario,
        "treatment": capture_treatment(conn, args.schema),
        "baseline": capture_baseline(conn, args.baseline_prefix),
    }

    t, b = fixture["treatment"], fixture["baseline"]
    t_emb = sum(1 for f in t["facts"] if f["embedding"])
    b_emb = sum(1 for d in b["documents"] if d["embedding"])
    print(f"treatment: {len(t['facts'])} facts ({t_emb} embedded),"
          f" {len(t['namespaces'])} namespaces")
    print(f"baseline:  {len(b['documents'])} documents ({b_emb} embedded)")
    if not t_emb or not b_emb:
        raise SystemExit("refusing to write a fixture with an unembedded arm")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(fixture))
    print(f"wrote {out}  ({out.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
