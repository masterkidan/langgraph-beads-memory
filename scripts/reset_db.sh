#!/usr/bin/env bash
# Drop every run schema and the LangMem tables, so a measurement starts clean.
#
# Run schemas are unique per run (see demo/harness.run_once), so runs cannot
# contaminate each other any more — but they accumulate, and a fresh start
# before a benchmark makes "what is in here?" answerable at a glance.
#
# Destructive. Everything the harness needs is rebuilt on the next run:
# BeadsStore.init_schema() and PostgresStore.setup() are both idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."
uv run python - <<'PY'
import psycopg
c = psycopg.connect("postgresql://beads:beads@localhost:5433/beads", autocommit=True)
schemas = [r[0] for r in c.execute("""
    SELECT schema_name FROM information_schema.schemata
    WHERE schema_name NOT IN ('pg_catalog','information_schema','public')
      AND schema_name NOT LIKE 'pg_%'""").fetchall()]
for s in schemas:
    c.execute(f'DROP SCHEMA "{s}" CASCADE')
for t in ("store_vectors", "store", "store_migrations", "vector_migrations"):
    c.execute(f"DROP TABLE IF EXISTS public.{t} CASCADE")
print(f"dropped {len(schemas)} run schema(s) and the LangMem tables")
PY
