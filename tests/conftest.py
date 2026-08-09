import os
import uuid

import psycopg
import pytest
from pgvector.psycopg import register_vector

DSN = os.environ.get("BEADS_PG_DSN", "postgresql://beads:beads@localhost:5433/beads")


@pytest.fixture()
def conn():
    schema = f"test_{uuid.uuid4().hex[:12]}"
    try:
        c = psycopg.connect(DSN, autocommit=True)
    except psycopg.OperationalError as e:
        pytest.fail(f"Postgres not reachable at {DSN} — run `docker compose up -d`. ({e})")
    c.execute(f'CREATE SCHEMA "{schema}"')
    c.execute(f'SET search_path TO "{schema}", public')
    register_vector(c)
    yield c
    c.execute(f'DROP SCHEMA "{schema}" CASCADE')
    c.close()
