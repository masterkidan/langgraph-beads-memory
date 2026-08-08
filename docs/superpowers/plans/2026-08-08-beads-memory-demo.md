# langgraph-beads-memory Package + Demo Harness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the demo-scoped `langgraph-beads-memory` package (Postgres fact graph + LangGraph middleware + sub-agent fork/rollup) and the comparison harness that measures it against a LangMem baseline on the scripted 3-conversation scenario.

**Architecture:** A `beads_memory` Python package: `BeadsStore` (psycopg3 + pgvector, three tables), `BeadsMemoryMiddleware` (`AgentMiddleware` subclass — passive capture in `before_model`, window-trim + fact injection in `wrap_model_call`, final-answer capture in `after_model`), tool factories (`remember_fact`, `conclude_task`), and `make_subagent_tool` (fork + enforced rollup). A `demo/` harness runs both conditions N times against a local corpus, captures transcripts, computes objective metrics, and runs a blinded LLM judge.

**Tech Stack:** Python 3.12, psycopg3 + pgvector, LangChain/LangGraph 1.x (`create_agent` + middleware), LangMem (baseline), ChatOllama + `nomic-embed-text` (local), pytest, Docker (Postgres).

**Specs:** `docs/superpowers/specs/2026-07-31-beads-memory-design.md` (architecture), `docs/superpowers/specs/2026-08-08-beads-memory-demo-design.md` (demo scope/measurement). Read both before starting.

---

## File Structure

```
pyproject.toml                     # package langgraph-beads-memory, module beads_memory
docker-compose.yml                 # pgvector/pgvector:pg16 on port 5433
src/beads_memory/__init__.py       # public exports
src/beads_memory/ids.py            # deterministic fact ids + short-id render/resolve
src/beads_memory/schema.sql        # DDL: namespaces, facts, fact_edges
src/beads_memory/store.py          # BeadsStore: namespaces, facts, edges, search
src/beads_memory/embeddings.py     # Embedder protocol, OllamaEmbedder, FakeEmbedder
src/beads_memory/middleware.py     # BeadsMemoryMiddleware
src/beads_memory/tools.py          # make_remember_fact / make_conclude_task
src/beads_memory/subagent.py       # make_subagent_tool (fork + enforced rollup)
tests/conftest.py                  # fresh-schema Postgres fixture, FakeEmbedder
tests/test_ids.py
tests/test_store.py
tests/test_search.py
tests/test_middleware.py
tests/test_tools.py
tests/test_subagent.py
demo/corpus/pgvector.md            # planted research corpus (3 docs)
demo/corpus/qdrant.md
demo/corpus/weaviate.md
demo/scenario.py                   # scripted turns + planted constraints
demo/conditions.py                 # build_baseline() / build_treatment()
demo/harness.py                    # N runs x 2 conditions -> transcripts JSON
demo/judge.py                      # blinded rubric judge
demo/metrics.py                    # token accounting + constraint-carry
demo/smoke_test.py                 # pre-flight model checks
results/                           # run outputs (committed)
```

Conventions used throughout: DB connection string env var `BEADS_PG_DSN` (default `postgresql://beads:beads@localhost:5433/beads`), Ollama at default `http://localhost:11434`, embedding dim 768 (`nomic-embed-text`).

---

### Task 1: Project scaffolding

**Files:**
- Create: `pyproject.toml`, `docker-compose.yml`
- Modify: `.gitignore` (add `results/raw/`)

- [ ] **Step 1: Write `pyproject.toml`**

```toml
[project]
name = "langgraph-beads-memory"
version = "0.1.0"
description = "Beads-style durable memory for LangGraph agents on Postgres"
requires-python = ">=3.12"
license = { text = "MIT" }
dependencies = [
    "langchain>=1.0",
    "langgraph>=1.0",
    "psycopg[binary]>=3.2",
    "pgvector>=0.3",
]

[project.optional-dependencies]
demo = [
    "langchain-ollama>=0.3",
    "langgraph-checkpoint-postgres>=2.0",
    "langmem>=0.0.29",
]
dev = ["pytest>=8.0"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/beads_memory"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write `docker-compose.yml`**

```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: beads
      POSTGRES_PASSWORD: beads
      POSTGRES_DB: beads
    ports:
      - "5433:5432"
```

- [ ] **Step 3: Create venv, install, pin versions**

Run:
```bash
python3.12 -m venv .venv && .venv/bin/pip install -e '.[demo,dev]' && .venv/bin/pip freeze > requirements.lock
```
Expected: install succeeds; `requirements.lock` created (this is the version pin required by demo spec §5). If `langmem` or `langchain>=1.0` fail to resolve, record the actual resolvable versions in `pyproject.toml` rather than loosening the lock.

- [ ] **Step 4: Start Postgres and verify**

Run: `docker compose up -d && sleep 3 && docker compose exec postgres psql -U beads -c "CREATE EXTENSION IF NOT EXISTS vector; SELECT extversion FROM pg_extension WHERE extname='vector';"`
Expected: one row with a version like `0.7.x`.

- [ ] **Step 5: Append to `.gitignore` and commit**

Add line `results/raw/` to `.gitignore`.

```bash
git add pyproject.toml docker-compose.yml requirements.lock .gitignore
git commit -m "chore: scaffold package, docker postgres, pinned deps"
```

---

### Task 2: Deterministic ids (`ids.py`)

**Files:**
- Create: `src/beads_memory/__init__.py` (empty for now), `src/beads_memory/ids.py`
- Test: `tests/test_ids.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_ids.py
import uuid
from beads_memory.ids import derive_fact_id, short_id, random_fork_suffix

NS = uuid.UUID("00000000-0000-0000-0000-000000000001")

def test_derive_fact_id_is_deterministic():
    a = derive_fact_id(NS, "msg-1", "hello world")
    b = derive_fact_id(NS, "msg-1", "hello world")
    assert isinstance(a, uuid.UUID) and a == b

def test_derive_fact_id_varies_by_inputs():
    base = derive_fact_id(NS, "msg-1", "hello")
    assert base != derive_fact_id(NS, "msg-2", "hello")
    assert base != derive_fact_id(NS, "msg-1", "bye")
    assert base != derive_fact_id(uuid.uuid4(), "msg-1", "hello")

def test_short_id_format():
    fid = derive_fact_id(NS, "msg-1", "hello")
    s = short_id(fid)
    assert s == f"fact-{fid.hex[:8]}"

def test_random_fork_suffix_shape_and_uniqueness():
    a, b = random_fork_suffix(), random_fork_suffix()
    assert a.startswith("sub-") and len(a) == 12  # "sub-" + 8 hex
    assert a != b
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_ids.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'beads_memory.ids'`

- [ ] **Step 3: Implement `src/beads_memory/ids.py`**

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_ids.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/beads_memory tests/test_ids.py
git commit -m "feat: deterministic fact ids, short ids, fork suffixes"
```

---

### Task 3: Schema + store core (namespaces)

**Files:**
- Create: `src/beads_memory/schema.sql`, `src/beads_memory/store.py`, `tests/conftest.py`
- Test: `tests/test_store.py`

- [ ] **Step 1: Write `src/beads_memory/schema.sql`**

```sql
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS namespaces (
    id          uuid PRIMARY KEY,
    session_id  text NOT NULL,
    extra_path  text[] NOT NULL DEFAULT '{}',
    parent_id   uuid REFERENCES namespaces(id),
    created_at  timestamptz NOT NULL DEFAULT now(),
    UNIQUE (session_id, extra_path)
);

CREATE TABLE IF NOT EXISTS facts (
    id                  uuid PRIMARY KEY,
    namespace_id        uuid NOT NULL REFERENCES namespaces(id),
    session_id          text NOT NULL,
    kind                text NOT NULL CHECK (kind IN ('user_input','conclusion','summary')),
    body                text NOT NULL,
    embedding           vector(768),
    status              text NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active','superseded','archived')),
    source              text NOT NULL CHECK (source IN
                        ('passive_capture','remember_tool','conclude_task',
                         'fallback_conclude','compaction')),
    agent_id            text NOT NULL,
    acting_on_behalf_of text NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS facts_session_idx ON facts (session_id);
CREATE INDEX IF NOT EXISTS facts_ns_status_idx ON facts (namespace_id, status);

CREATE TABLE IF NOT EXISTS fact_edges (
    id           uuid PRIMARY KEY,
    from_fact_id uuid NOT NULL REFERENCES facts(id),
    to_fact_id   uuid NOT NULL REFERENCES facts(id),
    relation     text NOT NULL CHECK (relation IN
                 ('supersedes','contradicts','relates_to','derived_from','rollup_of')),
    created_at   timestamptz NOT NULL DEFAULT now(),
    UNIQUE (from_fact_id, to_fact_id, relation)
);
```

- [ ] **Step 2: Write `tests/conftest.py`** (fresh schema per test — parallel-safe, no cross-test bleed)

```python
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
```

Note: `CREATE EXTENSION vector` lands in `public`; the `search_path` includes `public` so the `vector` type resolves inside test schemas.

- [ ] **Step 3: Write the failing namespace tests**

```python
# tests/test_store.py
import uuid

from beads_memory.store import BeadsStore


def test_init_schema_idempotent(conn):
    store = BeadsStore(conn)
    store.init_schema()
    store.init_schema()  # must not raise


def test_get_or_create_root_namespace(conn):
    store = BeadsStore(conn)
    store.init_schema()
    ns1 = store.get_or_create_namespace("sess-1")
    ns2 = store.get_or_create_namespace("sess-1")
    assert ns1.id == ns2.id
    assert ns1.session_id == "sess-1"
    assert ns1.extra_path == []
    assert ns1.parent_id is None


def test_fork_namespace_creates_child(conn):
    store = BeadsStore(conn)
    store.init_schema()
    root = store.get_or_create_namespace("sess-1")
    child = store.fork_namespace(root)
    assert child.parent_id == root.id
    assert child.session_id == "sess-1"
    assert child.extra_path[0] == "task" and child.extra_path[1].startswith("sub-")


def test_ancestor_chain(conn):
    store = BeadsStore(conn)
    store.init_schema()
    root = store.get_or_create_namespace("sess-1")
    child = store.fork_namespace(root)
    grand = store.fork_namespace(child)
    chain = store.ancestor_chain(grand.id)
    assert chain == [grand.id, child.id, root.id]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_store.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'beads_memory.store'`

- [ ] **Step 5: Implement `src/beads_memory/store.py` (namespace part)**

```python
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
```

Also make the DDL ship with the wheel — add to `pyproject.toml`:

```toml
[tool.hatch.build.targets.wheel.force-include]
"src/beads_memory/schema.sql" = "beads_memory/schema.sql"
```

(If `schema.sql` already ends up inside the package dir via `packages = ["src/beads_memory"]`, this is redundant but harmless.)

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_store.py -v`
Expected: 4 passed

- [ ] **Step 7: Commit**

```bash
git add src/beads_memory tests/conftest.py tests/test_store.py pyproject.toml
git commit -m "feat: schema DDL + BeadsStore namespaces (create/fork/ancestors)"
```

---

### Task 4: Fact writes (idempotent) + supersedes edges

**Files:**
- Modify: `src/beads_memory/store.py`
- Test: `tests/test_store.py` (append)

- [ ] **Step 1: Write the failing tests (append to `tests/test_store.py`)**

```python
def _mkstore(conn):
    s = BeadsStore(conn)
    s.init_schema()
    return s


def test_write_fact_idempotent(conn):
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("sess-1")
    f1 = store.write_fact(ns, kind="user_input", body="budget is 100k",
                          source="passive_capture", source_key="msg-1",
                          agent_id="root", acting_on_behalf_of="user")
    f2 = store.write_fact(ns, kind="user_input", body="budget is 100k",
                          source="passive_capture", source_key="msg-1",
                          agent_id="root", acting_on_behalf_of="user")
    assert f1.id == f2.id
    count = conn.execute("SELECT count(*) FROM facts").fetchone()[0]
    assert count == 1


def test_supersedes_edge_flips_status(conn):
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("sess-1")
    old = store.write_fact(ns, kind="user_input", body="budget is 100k",
                           source="passive_capture", source_key="m1",
                           agent_id="root", acting_on_behalf_of="user")
    new = store.write_fact(ns, kind="conclusion", body="budget is 50k",
                           source="remember_tool", source_key="tc-1",
                           agent_id="root", acting_on_behalf_of="user")
    store.add_edge(new.id, old.id, "supersedes")
    status = conn.execute("SELECT status FROM facts WHERE id=%s", (old.id,)).fetchone()[0]
    assert status == "superseded"


def test_relates_to_edge_does_not_touch_status(conn):
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("sess-1")
    a = store.write_fact(ns, kind="conclusion", body="A", source="remember_tool",
                         source_key="t1", agent_id="root", acting_on_behalf_of="user")
    b = store.write_fact(ns, kind="conclusion", body="B", source="remember_tool",
                         source_key="t2", agent_id="root", acting_on_behalf_of="user")
    store.add_edge(a.id, b.id, "relates_to")
    status = conn.execute("SELECT status FROM facts WHERE id=%s", (b.id,)).fetchone()[0]
    assert status == "active"


def test_facts_in_namespace(conn):
    store = _mkstore(conn)
    ns = store.get_or_create_namespace("sess-1")
    other = store.fork_namespace(ns)
    store.write_fact(ns, kind="user_input", body="root fact", source="passive_capture",
                     source_key="m1", agent_id="root", acting_on_behalf_of="user")
    store.write_fact(other, kind="conclusion", body="child fact", source="remember_tool",
                     source_key="t1", agent_id="sub", acting_on_behalf_of="root")
    bodies = [f.body for f in store.facts_in_namespace(other.id)]
    assert bodies == ["child fact"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_store.py -v -k "fact or edge"`
Expected: FAIL with `AttributeError: 'BeadsStore' object has no attribute 'write_fact'`

- [ ] **Step 3: Implement — append methods to `BeadsStore`**

```python
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
        from .ids import derive_fact_id

        fid = derive_fact_id(namespace.id, source_key, body)
        self._conn.execute(
            """
            INSERT INTO facts (id, namespace_id, session_id, kind, body, embedding,
                               source, agent_id, acting_on_behalf_of)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id) DO NOTHING
            """,
            (fid, namespace.id, namespace.session_id, kind, body, embedding,
             source, agent_id, acting_on_behalf_of),
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
            # The only path that sets status='superseded' (design spec §4).
            self._conn.execute(
                "UPDATE facts SET status='superseded' WHERE id=%s", (to_fact_id,)
            )

    def facts_in_namespace(self, namespace_id: uuid.UUID) -> list[Fact]:
        rows = self._conn.execute(
            "SELECT id, namespace_id, session_id, kind, body, status, source,"
            " agent_id, acting_on_behalf_of FROM facts WHERE namespace_id=%s"
            " ORDER BY created_at",
            (namespace_id,),
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
```

Note for the implementer: `id::text LIKE 'a3f8b2c1%'` works because a UUID's first text group is exactly the first 8 hex chars.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_store.py -v`
Expected: 8 passed

- [ ] **Step 5: Commit**

```bash
git add src/beads_memory/store.py tests/test_store.py
git commit -m "feat: idempotent fact writes, typed edges with supersedes status flip"
```

---

### Task 5: Embeddings + semantic search with ancestor read-through

**Files:**
- Create: `src/beads_memory/embeddings.py`
- Modify: `src/beads_memory/store.py`, `tests/conftest.py`
- Test: `tests/test_search.py`

- [ ] **Step 1: Implement `src/beads_memory/embeddings.py`** (no test needed for the protocol itself; FakeEmbedder is test infrastructure)

```python
"""Embedding providers. Demo embeds synchronously (demo spec §4)."""
from __future__ import annotations

from typing import Protocol


class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> list[float]: ...


class OllamaEmbedder:
    """nomic-embed-text via langchain-ollama; dim must match schema vector(768)."""

    dim = 768

    def __init__(self, model: str = "nomic-embed-text"):
        from langchain_ollama import OllamaEmbeddings

        self._emb = OllamaEmbeddings(model=model)

    def embed(self, text: str) -> list[float]:
        return self._emb.embed_query(text)


class FakeEmbedder:
    """Deterministic embeddings for tests: same text -> same vector; token overlap
    -> higher cosine similarity. Not semantically meaningful, but ranking-testable."""

    dim = 768

    def embed(self, text: str) -> list[float]:
        vec = [0.0] * self.dim
        for tok in text.lower().split():
            vec[hash(tok) % self.dim] += 1.0
        norm = sum(v * v for v in vec) ** 0.5 or 1.0
        return [v / norm for v in vec]
```

- [ ] **Step 2: Add `fake_embedder` fixture to `tests/conftest.py`**

```python
from beads_memory.embeddings import FakeEmbedder


@pytest.fixture()
def embedder():
    return FakeEmbedder()
```

- [ ] **Step 3: Write the failing search tests**

```python
# tests/test_search.py
from beads_memory.store import BeadsStore


def _store(conn):
    s = BeadsStore(conn)
    s.init_schema()
    return s


def _write(store, ns, body, embedder, *, kind="conclusion", source="remember_tool", key=None):
    return store.write_fact(ns, kind=kind, body=body, source=source,
                            source_key=key or body, agent_id="root",
                            acting_on_behalf_of="user", embedding=embedder.embed(body))


def test_search_ranks_relevant_first(conn, embedder):
    store = _store(conn)
    ns = store.get_or_create_namespace("s1")
    _write(store, ns, "the budget for the project is 100k dollars", embedder)
    _write(store, ns, "the team mascot is a heron", embedder)
    hits = store.search(ns.id, embedder.embed("what is the project budget"), k=2)
    assert hits[0].body.startswith("the budget")


def test_search_excludes_superseded(conn, embedder):
    store = _store(conn)
    ns = store.get_or_create_namespace("s1")
    old = _write(store, ns, "budget is 100k", embedder)
    new = _write(store, ns, "budget is 50k", embedder)
    store.add_edge(new.id, old.id, "supersedes")
    hits = store.search(ns.id, embedder.embed("budget"), k=10)
    bodies = [h.body for h in hits]
    assert "budget is 50k" in bodies and "budget is 100k" not in bodies


def test_search_reads_ancestors_not_siblings(conn, embedder):
    store = _store(conn)
    root = store.get_or_create_namespace("s1")
    child_a = store.fork_namespace(root)
    child_b = store.fork_namespace(root)
    _write(store, root, "root knows the goal", embedder)
    _write(store, child_a, "sibling A secret finding", embedder)
    hits = store.search(child_b.id, embedder.embed("secret finding goal"), k=10)
    bodies = [h.body for h in hits]
    assert "root knows the goal" in bodies
    assert "sibling A secret finding" not in bodies


def test_search_excludes_explicit_ids(conn, embedder):
    store = _store(conn)
    ns = store.get_or_create_namespace("s1")
    f = _write(store, ns, "budget is 100k", embedder)
    hits = store.search(ns.id, embedder.embed("budget"), k=10, exclude_ids=[f.id])
    assert all(h.id != f.id for h in hits)


def test_search_skips_unembedded_facts(conn, embedder):
    store = _store(conn)
    ns = store.get_or_create_namespace("s1")
    store.write_fact(ns, kind="user_input", body="unembedded budget note",
                     source="passive_capture", source_key="m9",
                     agent_id="root", acting_on_behalf_of="user")  # no embedding
    hits = store.search(ns.id, embedder.embed("budget"), k=10)
    assert hits == []
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_search.py -v`
Expected: FAIL with `AttributeError: 'BeadsStore' object has no attribute 'search'`

- [ ] **Step 5: Implement — append `search` to `BeadsStore`**

```python
    def search(
        self,
        namespace_id: uuid.UUID,
        query_embedding: list[float],
        *,
        k: int = 8,
        exclude_ids: list[uuid.UUID] | None = None,
    ) -> list[Fact]:
        """Top-k active facts by cosine similarity across self + ancestor chain.
        Facts without embeddings are invisible to search (design spec §7)."""
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
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_search.py -v`
Expected: 5 passed

- [ ] **Step 7: Commit**

```bash
git add src/beads_memory/embeddings.py src/beads_memory/store.py tests/
git commit -m "feat: embedders + pgvector search with ancestor read-through"
```

---

### Task 6: `remember_fact` and `conclude_task` tools

**Files:**
- Create: `src/beads_memory/tools.py`
- Test: `tests/test_tools.py`

The tools are closures over `(store, namespace, embedder, agent_id)` so each agent instance gets namespace-bound tools. `conclude_task` additionally closes over the *parent* namespace and a mutable `concluded` holder the sub-agent wrapper (Task 8) inspects.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_tools.py
from beads_memory.store import BeadsStore
from beads_memory.tools import make_remember_fact, make_conclude_task


def _setup(conn, embedder):
    store = BeadsStore(conn)
    store.init_schema()
    root = store.get_or_create_namespace("s1")
    return store, root


def test_remember_fact_writes_conclusion(conn, embedder):
    store, root = _setup(conn, embedder)
    tool = make_remember_fact(store, root, embedder, agent_id="root",
                              acting_on_behalf_of="user")
    out = tool.invoke({"body": "pgvector fits the 50k budget"})
    assert out.startswith("Remembered [fact-")
    facts = store.facts_in_namespace(root.id)
    assert facts[0].kind == "conclusion" and facts[0].source == "remember_tool"


def test_remember_fact_supersedes_by_short_id(conn, embedder):
    store, root = _setup(conn, embedder)
    old = store.write_fact(root, kind="user_input", body="budget is 100k",
                           source="passive_capture", source_key="m1",
                           agent_id="root", acting_on_behalf_of="user")
    from beads_memory.ids import short_id
    tool = make_remember_fact(store, root, embedder, agent_id="root",
                              acting_on_behalf_of="user")
    tool.invoke({"body": "budget is 50k", "relates_to": short_id(old.id),
                 "relation": "supersedes"})
    status = conn.execute("SELECT status FROM facts WHERE id=%s", (old.id,)).fetchone()[0]
    assert status == "superseded"


def test_remember_fact_bad_short_id_returns_error_string(conn, embedder):
    store, root = _setup(conn, embedder)
    tool = make_remember_fact(store, root, embedder, agent_id="root",
                              acting_on_behalf_of="user")
    out = tool.invoke({"body": "x", "relates_to": "fact-zzzzzzzz", "relation": "relates_to"})
    assert "Error" in out
    assert store.facts_in_namespace(root.id) == []  # no fact written on bad ref


def test_conclude_task_writes_summary_to_parent_with_rollups(conn, embedder):
    store, root = _setup(conn, embedder)
    child = store.fork_namespace(root)
    exploration = store.write_fact(child, kind="conclusion", body="checked qdrant docs",
                                   source="remember_tool", source_key="t1",
                                   agent_id="sub-1", acting_on_behalf_of="root")
    holder = {}
    tool = make_conclude_task(store, child, root, embedder, agent_id="sub-1",
                              acting_on_behalf_of="root", concluded=holder)
    tool.invoke({"summary": "qdrant needs 32GB RAM minimum"})
    assert "fact_id" in holder
    parent_facts = store.facts_in_namespace(root.id)
    assert parent_facts[0].kind == "summary" and parent_facts[0].source == "conclude_task"
    edge = conn.execute(
        "SELECT relation FROM fact_edges WHERE to_fact_id=%s", (exploration.id,)
    ).fetchone()
    assert edge[0] == "rollup_of"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'beads_memory.tools'`

- [ ] **Step 3: Implement `src/beads_memory/tools.py`**

```python
"""Agent-facing tools, bound per-namespace as closures."""
from __future__ import annotations

import uuid

from langchain_core.tools import tool

from .embeddings import Embedder
from .ids import short_id
from .store import BeadsStore, Namespace

_RELATIONS = ("supersedes", "contradicts", "relates_to")


def make_remember_fact(
    store: BeadsStore,
    namespace: Namespace,
    embedder: Embedder,
    *,
    agent_id: str,
    acting_on_behalf_of: str,
):
    @tool
    def remember_fact(body: str, relates_to: str | None = None,
                      relation: str | None = None) -> str:
        """Durably remember a conclusion you have reached. Use relates_to (a short
        fact id like 'fact-a3f8b2c1' from your Memory context) with relation
        'supersedes' when this conclusion replaces an earlier fact, 'contradicts'
        or 'relates_to' otherwise."""
        target = None
        if relates_to is not None:
            if relation not in _RELATIONS:
                return f"Error: relation must be one of {_RELATIONS}"
            try:
                readable = store.ancestor_chain(namespace.id)
                target = store.resolve_short_id(relates_to, readable)
            except LookupError as e:
                return f"Error: {e}"
        fact = store.write_fact(
            namespace, kind="conclusion", body=body, source="remember_tool",
            source_key=f"remember:{uuid.uuid4()}", agent_id=agent_id,
            acting_on_behalf_of=acting_on_behalf_of, embedding=embedder.embed(body),
        )
        if target is not None:
            store.add_edge(fact.id, target.id, relation)
        return f"Remembered [{short_id(fact.id)}]"

    return remember_fact


def make_conclude_task(
    store: BeadsStore,
    child_namespace: Namespace,
    parent_namespace: Namespace,
    embedder: Embedder,
    *,
    agent_id: str,
    acting_on_behalf_of: str,
    concluded: dict,
):
    @tool
    def conclude_task(summary: str, supersedes: str | None = None) -> str:
        """REQUIRED before you finish: report your task's conclusion. The summary
        is written to your parent's memory; your raw exploration stays in your own.
        Optionally pass supersedes=<short fact id> if your conclusion replaces an
        earlier fact."""
        target = None
        if supersedes is not None:
            try:
                readable = store.ancestor_chain(child_namespace.id)
                target = store.resolve_short_id(supersedes, readable)
            except LookupError as e:
                return f"Error: {e}"
        fact = store.write_fact(
            parent_namespace, kind="summary", body=summary, source="conclude_task",
            source_key=f"conclude:{child_namespace.id}", agent_id=agent_id,
            acting_on_behalf_of=acting_on_behalf_of, embedding=embedder.embed(summary),
        )
        for child_fact in store.facts_in_namespace(child_namespace.id):
            store.add_edge(fact.id, child_fact.id, "rollup_of")
        if target is not None:
            store.add_edge(fact.id, target.id, "supersedes")
        concluded["fact_id"] = fact.id
        return f"Task concluded [{short_id(fact.id)}]"

    return conclude_task
```

Note: `source_key=f"conclude:{child_namespace.id}"` makes a repeated/replayed conclude for the same child namespace idempotent (same fact id) as long as the summary text matches; a *different* summary writes a second fact — acceptable for demo scope.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_tools.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/beads_memory/tools.py tests/test_tools.py
git commit -m "feat: remember_fact and conclude_task tools with short-id resolution"
```

---

### Task 7: `BeadsMemoryMiddleware`

**Files:**
- Create: `src/beads_memory/middleware.py`
- Test: `tests/test_middleware.py`

Hook mapping (verified against langchain 1.x docs):
- `before_model(state, runtime) -> dict | None` — passive user-input capture (write path 1).
- `wrap_model_call(request, handler)` — view-only window trim + fact injection into the system message (write nothing; never mutates state).
- `after_model(state, runtime) -> dict | None` — passive final-answer capture (write path 4, root only: only when the last message is an `AIMessage` **without** tool calls).
- `tools` instance attribute — `remember_fact` (+ `conclude_task` when forked).

- [ ] **Step 1: Write the failing tests**

Tests drive hooks directly with synthetic `state` dicts — no LLM, no compiled agent. For `wrap_model_call` we call the hook with a stub request object and a `handler` that records what it receives.

```python
# tests/test_middleware.py
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from beads_memory.middleware import BeadsMemoryMiddleware
from beads_memory.store import BeadsStore


def _mw(conn, embedder, window=3):
    store = BeadsStore(conn)
    store.init_schema()
    ns = store.get_or_create_namespace("s1")
    mw = BeadsMemoryMiddleware(store=store, namespace=ns, embedder=embedder,
                               agent_id="root", acting_on_behalf_of="user",
                               window=window)
    return store, ns, mw


def test_before_model_captures_new_human_messages_idempotently(conn, embedder):
    store, ns, mw = _mw(conn, embedder)
    state = {"messages": [HumanMessage("budget is 100k", id="m1")]}
    mw.before_model(state, None)
    mw.before_model(state, None)  # replay — must not duplicate
    facts = store.facts_in_namespace(ns.id)
    assert len(facts) == 1
    assert facts[0].kind == "user_input" and facts[0].source == "passive_capture"


def test_after_model_captures_final_answer_not_toolcalls(conn, embedder):
    store, ns, mw = _mw(conn, embedder)
    toolcall_msg = AIMessage("", tool_calls=[{"name": "x", "args": {}, "id": "tc1"}], id="a1")
    mw.after_model({"messages": [toolcall_msg]}, None)
    assert store.facts_in_namespace(ns.id) == []  # tool-call turns are not conclusions
    final = AIMessage("I recommend pgvector.", id="a2")
    mw.after_model({"messages": [final]}, None)
    facts = store.facts_in_namespace(ns.id)
    assert len(facts) == 1 and facts[0].kind == "conclusion"


def test_after_model_disabled_for_non_root(conn, embedder):
    store, ns, mw = _mw(conn, embedder)
    mw.capture_final = False
    mw.after_model({"messages": [AIMessage("done", id="a1")]}, None)
    assert store.facts_in_namespace(ns.id) == []


def test_wrap_model_call_trims_window_and_injects_facts(conn, embedder):
    store, ns, mw = _mw(conn, embedder, window=3)
    old = store.write_fact(ns, kind="user_input", body="the budget is 100k dollars",
                           source="passive_capture", source_key="old-1",
                           agent_id="root", acting_on_behalf_of="user",
                           embedding=embedder.embed("the budget is 100k dollars"))
    msgs = [HumanMessage(f"filler {i}", id=f"m{i}") for i in range(5)]
    msgs.append(HumanMessage("what is the budget?", id="m-q"))
    captured = {}

    def handler(req):
        captured["messages"] = req.messages
        captured["system"] = req.system_message
        return "RESPONSE"

    req = SimpleNamespace(
        messages=msgs,
        system_message=SystemMessage("You are helpful."),
        override=lambda **kw: SimpleNamespace(
            messages=kw.get("messages", msgs),
            system_message=kw.get("system_message", SystemMessage("You are helpful.")),
        ),
    )
    result = mw.wrap_model_call(req, handler)
    assert result == "RESPONSE"
    assert len(captured["messages"]) == 3  # window trim, view-only
    sys_text = str(captured["system"].content)
    assert "fact-" in sys_text and "budget is 100k" in sys_text


def test_wrap_model_call_dedups_facts_still_in_window(conn, embedder):
    store, ns, mw = _mw(conn, embedder, window=5)
    msg = HumanMessage("the budget is 100k dollars", id="m1")
    mw.before_model({"messages": [msg]}, None)  # captured AND still in window
    captured = {}

    def handler(req):
        captured["system"] = req.system_message
        return "R"

    req = SimpleNamespace(
        messages=[msg],
        system_message=SystemMessage("sys"),
        override=lambda **kw: SimpleNamespace(
            messages=kw.get("messages", [msg]),
            system_message=kw.get("system_message", SystemMessage("sys")),
        ),
    )
    mw.wrap_model_call(req, handler)
    assert "budget is 100k" not in str(captured["system"].content)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_middleware.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'beads_memory.middleware'`

- [ ] **Step 3: Implement `src/beads_memory/middleware.py`**

```python
"""LangGraph agent middleware: the integration surface of beads-memory."""
from __future__ import annotations

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from .embeddings import Embedder
from .ids import derive_fact_id, short_id
from .store import BeadsStore, Namespace


class BeadsMemoryMiddleware(AgentMiddleware):
    def __init__(
        self,
        *,
        store: BeadsStore,
        namespace: Namespace,
        embedder: Embedder,
        agent_id: str,
        acting_on_behalf_of: str,
        window: int = 10,
        k: int = 8,
        capture_final: bool = True,
        extra_tools: list | None = None,
    ):
        super().__init__()
        self.store = store
        self.namespace = namespace
        self.embedder = embedder
        self.agent_id = agent_id
        self.acting_on_behalf_of = acting_on_behalf_of
        self.window = window
        self.k = k
        self.capture_final = capture_final
        from .tools import make_remember_fact

        self.tools = [
            make_remember_fact(store, namespace, embedder, agent_id=agent_id,
                               acting_on_behalf_of=acting_on_behalf_of)
        ] + (extra_tools or [])

    # -- write path 1: passive user-input capture ---------------------------
    def before_model(self, state, runtime):
        for msg in state["messages"]:
            if isinstance(msg, HumanMessage):
                self.store.write_fact(
                    self.namespace, kind="user_input", body=str(msg.content),
                    source="passive_capture", source_key=msg.id or str(msg.content),
                    agent_id=self.agent_id,
                    acting_on_behalf_of=self.acting_on_behalf_of,
                    embedding=self.embedder.embed(str(msg.content)),
                )
        return None

    # -- write path 4: passive final-answer capture (root only) -------------
    def after_model(self, state, runtime):
        if not self.capture_final:
            return None
        last = state["messages"][-1] if state["messages"] else None
        if isinstance(last, AIMessage) and not last.tool_calls and str(last.content).strip():
            self.store.write_fact(
                self.namespace, kind="conclusion", body=str(last.content),
                source="passive_capture", source_key=last.id or str(last.content),
                agent_id=self.agent_id,
                acting_on_behalf_of=self.acting_on_behalf_of,
                embedding=self.embedder.embed(str(last.content)),
            )
        return None

    # -- view-only window trim + fact injection -----------------------------
    def wrap_model_call(self, request, handler):
        msgs = list(request.messages)
        windowed = msgs[-self.window:]

        # Dedup: facts derived from messages still visible raw must not re-inject.
        exclude = [
            derive_fact_id(self.namespace.id, m.id or str(m.content), str(m.content))
            for m in windowed
            if isinstance(m, HumanMessage)
        ]
        query_text = next(
            (str(m.content) for m in reversed(windowed) if isinstance(m, HumanMessage)),
            None,
        )
        facts = []
        if query_text:
            facts = self.store.search(
                self.namespace.id, self.embedder.embed(query_text),
                k=self.k, exclude_ids=exclude,
            )
        system = request.system_message or SystemMessage("")
        if facts:
            lines = [f"- [{short_id(f.id)}] ({f.kind}) {f.body}" for f in facts]
            memory_block = (
                "\n\n## Memory (beads)\n"
                "Durable facts from this session. Cite short ids in remember_fact"
                " when a new conclusion supersedes/contradicts/relates to one.\n"
                + "\n".join(lines)
            )
            system = SystemMessage(str(system.content) + memory_block)
        return handler(request.override(messages=windowed, system_message=system))
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_middleware.py -v`
Expected: 5 passed

- [ ] **Step 5: Run the whole suite**

Run: `.venv/bin/pytest -v`
Expected: all pass (ids 4, store 8, search 5, tools 4, middleware 5 = 26)

- [ ] **Step 6: Commit**

```bash
git add src/beads_memory/middleware.py tests/test_middleware.py
git commit -m "feat: BeadsMemoryMiddleware (passive capture, window trim, fact injection)"
```

**Integration risk note for the implementer:** the `request.override(...)` /
`request.system_message` shapes were taken from the langchain 1.x custom-
middleware docs. If the installed version differs (e.g. system prompt lives in
`request.system_prompt`, or `override` is named `replace`), adapt the
middleware AND the stub in the test the same way — the test's stub is the
contract mirror, not an independent spec. Verify against the real API in
Task 9's smoke test before declaring this task truly done.

---

### Task 8: `make_subagent_tool` — fork + enforced rollup

**Files:**
- Create: `src/beads_memory/subagent.py`, `src/beads_memory/__init__.py` (exports)
- Test: `tests/test_subagent.py`

- [ ] **Step 1: Write the failing tests**

The sub-agent is faked as a callable so no LLM is involved: `make_subagent_tool` takes a `build_agent(middleware, tools) -> Callable[[str], str]` factory; tests supply factories that do/don't call `conclude_task`, or raise.

```python
# tests/test_subagent.py
import pytest

from beads_memory.store import BeadsStore
from beads_memory.subagent import make_subagent_tool


def _setup(conn, embedder):
    store = BeadsStore(conn)
    store.init_schema()
    root = store.get_or_create_namespace("s1")
    return store, root


def test_subagent_conclusion_lands_in_parent(conn, embedder):
    store, root = _setup(conn, embedder)

    def build_agent(middleware, tools):
        def run(task: str) -> str:
            conclude = next(t for t in tools if t.name == "conclude_task")
            conclude.invoke({"summary": f"finished: {task}"})
            return "done"
        return run

    tool = make_subagent_tool("researcher", "desc", store=store, parent_namespace=root,
                              embedder=embedder, build_agent=build_agent)
    out = tool.invoke({"task": "check qdrant"})
    assert "finished: check qdrant" in out
    parent = store.facts_in_namespace(root.id)
    assert len(parent) == 1 and parent[0].source == "conclude_task"


def test_lazy_subagent_gets_fallback_summary(conn, embedder):
    store, root = _setup(conn, embedder)

    def build_agent(middleware, tools):
        return lambda task: "I looked at things but forgot to conclude"

    tool = make_subagent_tool("lazy", "desc", store=store, parent_namespace=root,
                              embedder=embedder, build_agent=build_agent)
    tool.invoke({"task": "t"})
    parent = store.facts_in_namespace(root.id)
    assert len(parent) == 1
    assert parent[0].source == "fallback_conclude"
    assert "forgot to conclude" in parent[0].body  # last output used as fallback


def test_crashed_subagent_leaves_unresolved_fact(conn, embedder):
    store, root = _setup(conn, embedder)

    def build_agent(middleware, tools):
        def run(task):
            raise RuntimeError("boom")
        return run

    tool = make_subagent_tool("crashy", "desc", store=store, parent_namespace=root,
                              embedder=embedder, build_agent=build_agent)
    out = tool.invoke({"task": "t"})
    assert "did not complete" in out
    parent = store.facts_in_namespace(root.id)
    assert len(parent) == 1
    assert parent[0].source == "fallback_conclude"
    assert "did not complete" in parent[0].body


def test_parallel_forks_get_distinct_namespaces(conn, embedder):
    store, root = _setup(conn, embedder)
    seen = []

    def build_agent(middleware, tools):
        def run(task):
            seen.append(middleware.namespace.id)
            conclude = next(t for t in tools if t.name == "conclude_task")
            conclude.invoke({"summary": "ok"})
            return "ok"
        return run

    tool = make_subagent_tool("r", "d", store=store, parent_namespace=root,
                              embedder=embedder, build_agent=build_agent)
    tool.invoke({"task": "a"})
    tool.invoke({"task": "b"})
    assert len(set(seen)) == 2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/pytest tests/test_subagent.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'beads_memory.subagent'`

- [ ] **Step 3: Implement `src/beads_memory/subagent.py`**

```python
"""Sub-agent fork/rollup wrapper: fork namespace, run, enforce a conclusion."""
from __future__ import annotations

from typing import Callable

from langchain_core.tools import tool as tool_decorator

from .embeddings import Embedder
from .middleware import BeadsMemoryMiddleware
from .store import BeadsStore, Namespace
from .tools import make_conclude_task


def make_subagent_tool(
    name: str,
    description: str,
    *,
    store: BeadsStore,
    parent_namespace: Namespace,
    embedder: Embedder,
    build_agent: Callable,  # (middleware, tools) -> Callable[[str], str]
    parent_agent_id: str = "root",
):
    def _run(task: str) -> str:
        child = store.fork_namespace(parent_namespace)
        concluded: dict = {}
        middleware = BeadsMemoryMiddleware(
            store=store, namespace=child, embedder=embedder,
            agent_id=name, acting_on_behalf_of=parent_agent_id,
            capture_final=False,  # sub-agents conclude via conclude_task, not passively
        )
        conclude = make_conclude_task(
            store, child, parent_namespace, embedder,
            agent_id=name, acting_on_behalf_of=parent_agent_id, concluded=concluded,
        )
        tools = list(middleware.tools) + [conclude]
        agent = build_agent(middleware, tools)
        try:
            output = agent(task)
        except Exception as e:  # crashed sub-agent must not vanish silently
            output = None
            error = str(e)
        else:
            error = None

        if "fact_id" not in concluded:
            # Enforced rollup: synthesize the summary the sub-agent failed to write.
            if error is not None:
                body = f"Task did not complete: sub-agent '{name}' failed with: {error}"
            else:
                body = f"(auto-summary, agent did not conclude) {output}"
            fact = store.write_fact(
                parent_namespace, kind="summary", body=body,
                source="fallback_conclude", source_key=f"fallback:{child.id}",
                agent_id=name, acting_on_behalf_of=parent_agent_id,
                embedding=embedder.embed(body),
            )
            for child_fact in store.facts_in_namespace(child.id):
                store.add_edge(fact.id, child_fact.id, "rollup_of")
            if error is not None:
                return f"Sub-agent '{name}' did not complete: {error}"
        return output if output is not None else f"Sub-agent '{name}' did not complete."

    _run.__name__ = name
    _run.__doc__ = description
    return tool_decorator(_run)


# v1 scope (design spec §5.1): tool-invoked sub-agents only. Handoff-style
# delegation (Command(goto=...)) is documented future work.
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/pytest tests/test_subagent.py -v`
Expected: 4 passed

- [ ] **Step 5: Write `src/beads_memory/__init__.py` exports**

```python
from .embeddings import Embedder, FakeEmbedder, OllamaEmbedder
from .middleware import BeadsMemoryMiddleware
from .store import BeadsStore, Fact, Namespace
from .subagent import make_subagent_tool
from .tools import make_conclude_task, make_remember_fact

__all__ = [
    "BeadsMemoryMiddleware", "BeadsStore", "Fact", "Namespace",
    "Embedder", "FakeEmbedder", "OllamaEmbedder",
    "make_subagent_tool", "make_remember_fact", "make_conclude_task",
]
```

- [ ] **Step 6: Run full suite and commit**

Run: `.venv/bin/pytest -v` — Expected: 30 passed

```bash
git add src/beads_memory tests/test_subagent.py
git commit -m "feat: make_subagent_tool with enforced rollup and crash fallback"
```

---

### Task 9: Demo corpus + smoke test

**Files:**
- Create: `demo/corpus/pgvector.md`, `demo/corpus/qdrant.md`, `demo/corpus/weaviate.md`, `demo/smoke_test.py`, `demo/__init__.py` (empty), `demo/llm.py`

- [ ] **Step 1: Write the three corpus docs** (planted facts the scenario references; each ~30 lines so sub-agent reading is genuinely verbose; the **buried detail** is Qdrant's binary quantization)

`demo/corpus/pgvector.md`:
```markdown
# pgvector Evaluation Notes

pgvector is a Postgres extension providing vector similarity search.

## Deployment
Self-hosting is trivial if you already run Postgres: install the extension,
no separate service. Fully open source (PostgreSQL license). Annual
infrastructure cost for our workload (10M vectors, 768-d): approximately
$18,000/year on managed Postgres, or $12,000/year self-hosted on EC2.

## Performance
HNSW index: ~40ms p95 at 10M vectors with 95% recall in our load test.
Write throughput degrades ~20% while the HNSW index builds.

## Operational notes
Backups ride the existing Postgres backup pipeline. No new on-call surface.
Index rebuild after bulk load takes ~50 minutes at 10M vectors.
Team already knows Postgres; zero new operational skills required.

## Limits
Single-node vertical scaling only, without Citus. Metadata filtering is
just SQL WHERE clauses (a strength). No built-in quantization in the
version we tested; memory footprint is ~6GB for 10M 768-d vectors.
```

`demo/corpus/qdrant.md`:
```markdown
# Qdrant Evaluation Notes

Qdrant is a dedicated vector database written in Rust.

## Deployment
Self-hostable (Apache 2.0) as a single binary or via Docker/K8s. Managed
cloud also available. Self-hosted cluster for our workload (10M vectors,
768-d, HA pair): approximately $30,000/year including the extra on-call
burden we priced at half an SRE-week per quarter.

## Performance
p95 ~12ms at 10M vectors with 97% recall. Strong filtered-search
performance with payload indexes.

## Memory optimization
Binary quantization reduces RAM usage up to 32x with a modest recall hit
(recovered via oversampling + rescoring). In our test it cut a 24GB
deployment to under 2GB of hot RAM. This was the single largest memory
saving we measured across all candidates.

## Operational notes
New service to run: monitoring, upgrades, snapshots are all new surface.
Snapshot/restore tooling is solid. Team has no prior Rust-service
operational experience, though none is strictly required.
```

`demo/corpus/weaviate.md`:
```markdown
# Weaviate Evaluation Notes

Weaviate is an open-source vector database with a GraphQL-flavored API.

## Deployment
Self-hostable (BSD-3) via Docker/K8s; managed cloud available. Self-hosted
HA deployment for our workload: approximately $60,000/year — the K8s
operator effectively requires a small dedicated cluster, and we'd need the
commercial tier for the module ecosystem we want.

## Performance
p95 ~15ms at 10M vectors, 96% recall. Built-in hybrid (BM25+vector) search
is the standout feature; it noticeably improved our relevance on short
queries.

## Operational notes
Heaviest operational footprint of the three candidates. Module system
(rerankers, vectorizers) is powerful but adds upgrade complexity. GraphQL
API is a new paradigm for the team.
```

- [ ] **Step 2: Write `demo/llm.py`** (single place the model is chosen/configured)

```python
"""Model configuration for the demo. One model for agents AND judge."""
import os

from langchain_ollama import ChatOllama

MODEL = os.environ.get("BEADS_DEMO_MODEL", "qwen2.5:14b")


def make_llm(temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=MODEL, temperature=temperature)
```

- [ ] **Step 3: Write `demo/smoke_test.py`** (pre-flight required by demo spec §5 — run before any scored run)

```python
"""Pre-flight: verify the local model can do what BOTH conditions need.
Failing this means 'pick a better model', not 'the memory layer lost'."""
import sys

from langchain_core.messages import HumanMessage

from demo.llm import make_llm


def check_tool_calling() -> bool:
    """(b) treatment needs reliable tool calls with structured args."""
    from langchain_core.tools import tool

    calls = []

    @tool
    def remember_fact(body: str, relates_to: str | None = None,
                      relation: str | None = None) -> str:
        """Durably remember a conclusion."""
        calls.append({"body": body, "relates_to": relates_to, "relation": relation})
        return "Remembered [fact-12345678]"

    llm = make_llm().bind_tools([remember_fact])
    prompt = (
        "Context fact: [fact-aaaa1111] (user_input) budget is 100k.\n"
        "The user just said the budget is now 50k, not 100k. Record this "
        "revision using the remember_fact tool with relation supersedes."
    )
    resp = llm.invoke([HumanMessage(prompt)])
    ok = bool(resp.tool_calls)
    if ok:
        args = resp.tool_calls[0]["args"]
        ok = args.get("relation") == "supersedes" and "fact-aaaa1111" in str(args.get("relates_to"))
    print(f"tool-calling + supersedes: {'PASS' if ok else 'FAIL'} ({resp.tool_calls})")
    return ok


def check_extraction_prereq() -> bool:
    """(a) baseline needs the model to make LangMem-style extraction calls work;
    proxy check: structured JSON extraction from a transcript."""
    llm = make_llm()
    resp = llm.invoke([HumanMessage(
        'Extract durable facts as a JSON list of strings from: '
        '"My budget is 100k and I only trust primary sources." '
        "Answer with ONLY the JSON list."
    )])
    text = str(resp.content).strip()
    ok = text.startswith("[") and "100k" in text and "primary sources" in text
    print(f"extraction-shaped output: {'PASS' if ok else 'FAIL'} ({text[:120]})")
    return ok


def check_embeddings() -> bool:
    from beads_memory.embeddings import OllamaEmbedder

    emb = OllamaEmbedder()
    v = emb.embed("hello world")
    ok = len(v) == 768
    print(f"nomic-embed-text 768-d: {'PASS' if ok else 'FAIL'} (len={len(v)})")
    return ok


if __name__ == "__main__":
    results = [check_tool_calling(), check_extraction_prereq(), check_embeddings()]
    sys.exit(0 if all(results) else 1)
```

- [ ] **Step 4: Pull models and run the smoke test**

Run:
```bash
ollama pull qwen2.5:14b && ollama pull nomic-embed-text && .venv/bin/python -m demo.smoke_test
```
Expected: three PASS lines, exit 0. If tool-calling FAILs, try `BEADS_DEMO_MODEL=llama3.1:8b` (then re-run); record the winning model in `demo/llm.py`'s default. Do not proceed to Task 10 until this passes — that's the point of the gate.

- [ ] **Step 5: Commit**

```bash
git add demo/
git commit -m "feat: demo corpus with planted facts, model config, pre-flight smoke test"
```

---

### Task 10: Scenario script + conditions

**Files:**
- Create: `demo/scenario.py`, `demo/conditions.py`

- [ ] **Step 1: Write `demo/scenario.py`** (the fixed script + planted answers the metrics check against)

```python
"""The scripted 3-conversation scenario (demo spec §2). Identical for both
conditions; the only variable is the memory layer."""

RESEARCH_SYSTEM_PROMPT = (
    "You are a research analyst. You have memory tools: when you reach a "
    "conclusion, record it with remember_fact. When the user revises an "
    "earlier constraint, record the new value with remember_fact using "
    "relation='supersedes' and the old fact's short id from your Memory "
    "context. Use the read_document tool to research; delegate sub-topics "
    "to your researcher sub-agents when asked to investigate in depth."
)

SUBAGENT_SYSTEM_PROMPT = (
    "You are a focused researcher. Read the relevant documents thoroughly "
    "with read_document, record notable findings with remember_fact, and "
    "you MUST finish by calling conclude_task with a summary of your "
    "conclusion for your supervisor."
)

# conversation_id -> list of scripted user turns
CONVERSATIONS = [
    ("conv-1", [
        "We need to pick a vector database for our product. Constraints: "
        "the budget is $100k per year, it must be self-hostable, and I only "
        "trust primary benchmark data we measured ourselves, not vendor "
        "marketing numbers.",
    ]),
    ("conv-2", [
        "Please investigate our vector database options in depth. Delegate "
        "pgvector, Qdrant, and Weaviate to your researchers, then give me "
        "your synthesis.",
        # The revision beat (demo spec §2): supersedes gets exercised here.
        "One correction before we wrap up: the budget is $50k per year, "
        "not $100k.",
    ]),
    ("conv-3", [
        "Given everything we've established, which vector database should "
        "we pick and why? Be specific about how it fits our constraints.",
        # Buried-detail question: lives only in qdrant.md's memory section.
        "And remind me — what was that big memory optimization for the "
        "strongest runner-up, and roughly how much did it save?",
    ]),
]

# Ground truth for objective metrics (metrics.py checks these substrings).
PLANTED = {
    "revised_budget": "50k",
    "stale_budget": "100k",          # must NOT be presented as current
    "constraint_selfhost": "self-host",
    "constraint_primary_sources": "primary",
    "buried_detail_terms": ["binary quantization", "32x"],
    # $50k budget: pgvector ($12-18k) fits; qdrant ($30k) fits; weaviate ($60k) does not.
    "expected_pick_one_of": ["pgvector", "qdrant"],
}
```

- [ ] **Step 2: Write `demo/conditions.py`**

```python
"""Builds the two conditions. Both: same LLM, same corpus tool, same prompts,
same sub-agent split. Only the memory layer differs."""
from __future__ import annotations

import pathlib
import uuid

import psycopg
from langchain.agents import create_agent
from langchain_core.tools import tool
from pgvector.psycopg import register_vector

from demo.llm import make_llm
from demo.scenario import RESEARCH_SYSTEM_PROMPT, SUBAGENT_SYSTEM_PROMPT

CORPUS = pathlib.Path(__file__).parent / "corpus"
DSN = "postgresql://beads:beads@localhost:5433/beads"


@tool
def read_document(name: str) -> str:
    """Read an evaluation document. Available: pgvector, qdrant, weaviate."""
    path = CORPUS / f"{name.lower().strip()}.md"
    if not path.exists():
        return f"No document named {name}. Available: pgvector, qdrant, weaviate"
    return path.read_text()


SUBTOPICS = ["pgvector", "qdrant", "weaviate"]


# ---------------------------------------------------------------- treatment
def build_treatment(session_id: str, run_schema: str):
    """beads-memory condition. Returns (invoke_fn, cleanup_fn) where
    invoke_fn(thread_id, user_text) -> AIMessage-final-text, and internally
    constructs a fresh agent per conversation (new thread) but the SAME
    session_id namespace — that is the cross-conversation memory claim."""
    from beads_memory import (BeadsMemoryMiddleware, BeadsStore, OllamaEmbedder,
                              make_subagent_tool)

    conn = psycopg.connect(DSN, autocommit=True)
    conn.execute(f'CREATE SCHEMA IF NOT EXISTS "{run_schema}"')
    conn.execute(f'SET search_path TO "{run_schema}", public')
    register_vector(conn)
    store = BeadsStore(conn)
    store.init_schema()
    embedder = OllamaEmbedder()
    root_ns = store.get_or_create_namespace(session_id)

    def make_researcher(topic: str):
        def build_agent(middleware, tools):
            agent = create_agent(
                model=make_llm(),
                tools=[read_document] + tools,
                system_prompt=SUBAGENT_SYSTEM_PROMPT
                + f" Your assigned topic is: {topic}.",
                middleware=[middleware],
            )

            def run(task: str) -> str:
                result = agent.invoke(
                    {"messages": [("user", task)]},
                    {"configurable": {"thread_id": f"sub-{uuid.uuid4()}"}},
                )
                return str(result["messages"][-1].content)

            return run

        return make_subagent_tool(
            f"researcher_{topic}",
            f"Delegate in-depth research on {topic} to a focused researcher.",
            store=store, parent_namespace=root_ns, embedder=embedder,
            build_agent=build_agent,
        )

    def invoke(thread_id: str, user_text: str) -> dict:
        middleware = BeadsMemoryMiddleware(
            store=store, namespace=root_ns, embedder=embedder,
            agent_id="root", acting_on_behalf_of="user",
        )
        agent = create_agent(
            model=make_llm(),
            tools=[read_document] + [make_researcher(t) for t in SUBTOPICS],
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            middleware=[middleware],
        )
        result = agent.invoke(
            {"messages": [("user", user_text)]},
            {"configurable": {"thread_id": thread_id}},
        )
        return result

    return invoke, conn.close


# ----------------------------------------------------------------- baseline
def build_baseline(session_id: str, run_schema: str):
    """LangMem + PostgresStore condition (demo spec §3): idiomatic supervisor,
    sub-agent results return as tool messages, LangMem store shared by all."""
    from langgraph.store.postgres import PostgresStore
    from langmem import create_manage_memory_tool, create_search_memory_tool

    store_cm = PostgresStore.from_conn_string(DSN)
    store = store_cm.__enter__()
    store.setup()
    ns = ("memories", session_id)
    mem_tools = [create_manage_memory_tool(namespace=ns, store=store),
                 create_search_memory_tool(namespace=ns, store=store)]

    def make_researcher(topic: str):
        researcher = create_agent(
            model=make_llm(),
            tools=[read_document] + mem_tools,
            system_prompt=SUBAGENT_SYSTEM_PROMPT
            + f" Your assigned topic is: {topic}."
            + " Save important findings with the memory tool.",
        )

        @tool(f"researcher_{topic}")
        def _run(task: str) -> str:
            result = researcher.invoke({"messages": [("user", task)]})
            return str(result["messages"][-1].content)

        _run.description = f"Delegate in-depth research on {topic}."
        return _run

    def invoke(thread_id: str, user_text: str) -> dict:
        agent = create_agent(
            model=make_llm(),
            tools=[read_document] + mem_tools
            + [make_researcher(t) for t in SUBTOPICS],
            system_prompt=RESEARCH_SYSTEM_PROMPT.replace(
                "remember_fact", "the manage_memory tool"
            ).replace(
                "relation='supersedes' and the old fact's short id from your "
                "Memory context",
                "an update to the existing memory",
            )
            + " Before answering, search your memory for relevant context.",
        )
        result = agent.invoke(
            {"messages": [("user", user_text)]},
            {"configurable": {"thread_id": thread_id}},
        )
        return result

    def cleanup():
        store_cm.__exit__(None, None, None)

    return invoke, cleanup
```

Implementation notes for this task (verify at build time, adjust minimally):
- `create_agent` signature (`model=`, `tools=`, `system_prompt=`, `middleware=`) is langchain 1.x; if the installed version wants `prompt=` instead of `system_prompt=`, follow the installed API.
- `langmem` tool factory names (`create_manage_memory_tool(namespace=..., store=...)`) — if the installed langmem passes the store via `create_agent(store=...)`/runtime instead of the factory, wire it that way; the *behavioral* requirement is fixed: baseline agents can save/search memories in a session-scoped namespace shared across all three conversations and both agent tiers.
- Baseline has no checkpointer-carryover between conversations (fresh thread each time) — matching the treatment, which also starts fresh threads. Cross-conversation memory must come from each condition's memory layer. That is the comparison.

- [ ] **Step 3: Quick manual sanity run (one conversation, treatment only)**

Run:
```bash
.venv/bin/python -c "
from demo.conditions import build_treatment
inv, close = build_treatment('sanity-sess', 'sanity_run')
r = inv('t1', 'The budget is 100k per year. Remember the key constraint.')
print(r['messages'][-1].content)
close()
"
```
Expected: a coherent reply; no tracebacks. Then verify a fact landed:
```bash
docker compose exec postgres psql -U beads -c 'SET search_path TO sanity_run,public; SELECT kind, source, left(body,60) FROM facts;'
```
Expected: at least a `user_input/passive_capture` row and a `conclusion/passive_capture` row (final answer). Drop the schema afterwards: `docker compose exec postgres psql -U beads -c 'DROP SCHEMA sanity_run CASCADE;'`

- [ ] **Step 4: Commit**

```bash
git add demo/scenario.py demo/conditions.py
git commit -m "feat: scripted scenario + baseline/treatment condition builders"
```

---

### Task 11: Harness (N runs, transcripts) + objective metrics

**Files:**
- Create: `demo/harness.py`, `demo/metrics.py`
- Create dir: `results/raw/` (gitignored), `results/` (committed summaries)

- [ ] **Step 1: Write `demo/metrics.py`**

```python
"""Objective metrics: no judge involved (demo spec §6)."""
from __future__ import annotations

from demo.scenario import PLANTED


def token_usage(messages: list) -> dict:
    """Sum usage_metadata across AI messages in a transcript."""
    tin = tout = 0
    for m in messages:
        usage = getattr(m, "usage_metadata", None)
        if usage:
            tin += usage.get("input_tokens", 0)
            tout += usage.get("output_tokens", 0)
    return {"input_tokens": tin, "output_tokens": tout}


def constraint_carry(final_answer: str, buried_answer: str) -> dict:
    """Did planted constraints survive to conversation 3, with the REVISED
    budget (not the stale one) and the buried detail?"""
    fa = final_answer.lower()
    ba = buried_answer.lower()
    return {
        "uses_revised_budget": PLANTED["revised_budget"] in fa,
        "avoids_stale_budget_as_current": not (
            PLANTED["stale_budget"] in fa and PLANTED["revised_budget"] not in fa
        ),
        "mentions_selfhost": PLANTED["constraint_selfhost"] in fa,
        "mentions_primary_sources": PLANTED["constraint_primary_sources"] in fa,
        "pick_is_feasible": any(p in fa for p in PLANTED["expected_pick_one_of"]),
        "buried_detail_recalled": all(
            t in ba for t in [x.lower() for x in PLANTED["buried_detail_terms"]]
        ),
    }
```

- [ ] **Step 2: Write `demo/harness.py`**

```python
"""Runs the full scenario N times per condition; saves transcripts + metrics."""
from __future__ import annotations

import argparse
import json
import pathlib
import time
import uuid

from langchain_core.messages import message_to_dict

from demo import metrics
from demo.conditions import build_baseline, build_treatment
from demo.scenario import CONVERSATIONS

RAW = pathlib.Path(__file__).parent.parent / "results" / "raw"


def run_once(condition: str, run_idx: int) -> dict:
    session_id = f"{condition}-run{run_idx}-{uuid.uuid4().hex[:6]}"
    build = build_treatment if condition == "treatment" else build_baseline
    invoke, cleanup = build(session_id, run_schema=f"run_{condition}_{run_idx}")
    transcript, all_msgs = [], []
    try:
        for conv_id, turns in CONVERSATIONS:
            thread_id = f"{session_id}-{conv_id}"
            for user_text in turns:
                result = invoke(thread_id, user_text)
                msgs = result["messages"]
                all_msgs.extend(msgs)
                transcript.append({
                    "conversation": conv_id,
                    "user": user_text,
                    "messages": [message_to_dict(m) for m in msgs],
                    "final": str(msgs[-1].content),
                })
    finally:
        cleanup()

    conv3 = [t for t in transcript if t["conversation"] == "conv-3"]
    final_answer = conv3[0]["final"] if conv3 else ""
    buried_answer = conv3[1]["final"] if len(conv3) > 1 else ""
    return {
        "condition": condition,
        "run": run_idx,
        "session_id": session_id,
        "transcript": transcript,
        "tokens": metrics.token_usage(all_msgs),
        "constraint_carry": metrics.constraint_carry(final_answer, buried_answer),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--conditions", nargs="+",
                    default=["baseline", "treatment"])
    args = ap.parse_args()
    RAW.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for condition in args.conditions:
        for i in range(args.runs):
            print(f"=== {condition} run {i} ===")
            record = run_once(condition, i)
            out = RAW / f"{stamp}-{condition}-{i}.json"
            out.write_text(json.dumps(record, indent=2, default=str))
            print(f"  tokens={record['tokens']}  carry={record['constraint_carry']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Unit-test the metrics (no LLM needed)**

Append to a new file `tests/test_metrics.py`:

```python
from demo.metrics import constraint_carry


def test_constraint_carry_happy_path():
    final = ("I recommend pgvector: it fits the revised $50k budget, is "
             "self-hostable, and our primary benchmark data supports it.")
    buried = "Qdrant's binary quantization cut RAM up to 32x."
    c = constraint_carry(final, buried)
    assert all(c.values())


def test_constraint_carry_stale_budget_detected():
    final = "I recommend Weaviate; it fits the $100k budget."
    c = constraint_carry(final, "no idea")
    assert not c["uses_revised_budget"]
    assert not c["avoids_stale_budget_as_current"]
    assert not c["buried_detail_recalled"]
```

Run: `.venv/bin/pytest tests/test_metrics.py -v`
Expected: 2 passed

- [ ] **Step 4: Commit**

```bash
git add demo/harness.py demo/metrics.py tests/test_metrics.py
git commit -m "feat: N-run harness with transcripts, token accounting, constraint-carry metrics"
```

---

### Task 12: Blinded judge

**Files:**
- Create: `demo/judge.py`
- Test: `tests/test_judge.py`

- [ ] **Step 1: Write the failing test for blinding logic** (judge scoring itself needs the LLM; the *blinding* is pure logic and must be tested)

```python
# tests/test_judge.py
from demo.judge import blind_pair


def test_blind_pair_strips_and_randomizes(monkeypatch):
    a = {"condition": "baseline", "final": "answer A"}
    b = {"condition": "treatment", "final": "answer B"}
    monkeypatch.setattr("random.random", lambda: 0.9)  # force swap branch
    pair, mapping = blind_pair(a, b)
    assert set(pair.keys()) == {"X", "Y"}
    assert "condition" not in str(pair)
    assert mapping in ({"X": "baseline", "Y": "treatment"},
                       {"X": "treatment", "Y": "baseline"})
```

Run: `.venv/bin/pytest tests/test_judge.py -v` — Expected: FAIL (module missing)

- [ ] **Step 2: Implement `demo/judge.py`**

```python
"""Blinded rubric judge (demo spec §6): labels stripped, order randomized."""
from __future__ import annotations

import json
import pathlib
import random
import sys

from langchain_core.messages import HumanMessage

from demo.llm import make_llm

RUBRIC = """Score each transcript excerpt 1-5 on each dimension. Respond with
ONLY a JSON object: {"X": {"recall": n, "delegation": n, "final": n},
"Y": {"recall": n, "delegation": n, "final": n}}.

Dimensions:
- recall: does the conversation-3 answer correctly reflect the constraints
  stated in conversation 1 (self-hostable; only primary benchmark data) and
  use the REVISED budget from the correction, without being re-told any of it?
- delegation: does the synthesis correctly incorporate every researcher's
  findings? Are any findings lost, contradicted, or double-counted?
- final: is the final recommendation substantively correct and well-supported
  given the constraints and the research?
"""


def blind_pair(rec_a: dict, rec_b: dict) -> tuple[dict, dict]:
    """Strip condition labels; randomize which is X and which is Y."""
    pair = [rec_a, rec_b]
    if random.random() >= 0.5:
        pair.reverse()
    blinded = {
        "X": {k: v for k, v in pair[0].items() if k != "condition"},
        "Y": {k: v for k, v in pair[1].items() if k != "condition"},
    }
    mapping = {"X": pair[0]["condition"], "Y": pair[1]["condition"]}
    return blinded, mapping


def _excerpt(record: dict) -> str:
    convs = {}
    for t in record["transcript"]:
        convs.setdefault(t["conversation"], []).append(
            f"USER: {t['user']}\nAGENT: {t['final']}"
        )
    return "\n\n".join(f"[{c}]\n" + "\n".join(v) for c, v in convs.items())


def judge_pair(rec_a: dict, rec_b: dict) -> dict:
    blinded, mapping = blind_pair(
        {"condition": rec_a["condition"], "transcript": rec_a["transcript"]},
        {"condition": rec_b["condition"], "transcript": rec_b["transcript"]},
    )
    prompt = (
        RUBRIC
        + "\n\n=== Transcript X ===\n" + _excerpt(blinded["X"])
        + "\n\n=== Transcript Y ===\n" + _excerpt(blinded["Y"])
    )
    resp = make_llm().invoke([HumanMessage(prompt)])
    text = str(resp.content)
    scores = json.loads(text[text.index("{"): text.rindex("}") + 1])
    return {mapping[label]: scores[label] for label in ("X", "Y")}


def main(raw_dir: str):
    raw = pathlib.Path(raw_dir)
    records = [json.loads(p.read_text()) for p in sorted(raw.glob("*.json"))]
    by_run: dict[int, dict[str, dict]] = {}
    for r in records:
        by_run.setdefault(r["run"], {})[r["condition"]] = r
    all_scores = []
    for run, conds in sorted(by_run.items()):
        if {"baseline", "treatment"} <= conds.keys():
            s = judge_pair(conds["baseline"], conds["treatment"])
            print(f"run {run}: {s}")
            all_scores.append(s)
    # means
    for cond in ("baseline", "treatment"):
        for dim in ("recall", "delegation", "final"):
            vals = [s[cond][dim] for s in all_scores]
            print(f"mean {cond}.{dim}: {sum(vals)/len(vals):.2f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/raw")
```

- [ ] **Step 3: Run the blinding test**

Run: `.venv/bin/pytest tests/test_judge.py -v`
Expected: 1 passed

- [ ] **Step 4: Commit**

```bash
git add demo/judge.py tests/test_judge.py
git commit -m "feat: blinded LLM judge with per-run pairing and mean reporting"
```

---

### Task 13: Full scored run + results write-up

**Files:**
- Create: `results/2026-XX-XX-results.md` (dated at run time)
- Modify: `README.md` (status checkboxes)

- [ ] **Step 1: Pre-flight**

Run: `.venv/bin/python -m demo.smoke_test`
Expected: all PASS. If not, fix the model choice first (Task 9 Step 4).

- [ ] **Step 2: Full harness run**

Run: `.venv/bin/python -m demo.harness --runs 3`
Expected: 6 JSON files in `results/raw/` (3 runs x 2 conditions), each printing token + carry summaries. Budget ~30-60 min on local hardware.

- [ ] **Step 3: Judge run**

Run: `.venv/bin/python -m demo.judge results/raw`
Expected: per-run scores + per-dimension means printed.

- [ ] **Step 4: Write `results/<date>-results.md`**

Structure (fill with the actual measured numbers — every cell from the real output, no invented values):

```markdown
# Demo results — <date>, model <model>, N=3

## Objective metrics (no judge)
| metric | baseline | treatment |
|---|---|---|
| uses_revised_budget (runs passing) | x/3 | x/3 |
| avoids_stale_budget_as_current | x/3 | x/3 |
| mentions_selfhost | x/3 | x/3 |
| mentions_primary_sources | x/3 | x/3 |
| pick_is_feasible | x/3 | x/3 |
| buried_detail_recalled | x/3 | x/3 |
| mean input tokens | n | n |
| mean output tokens | n | n |

## Blinded judge means (1-5)
| dimension | baseline | treatment |
|---|---|---|
| recall | n.nn | n.nn |
| delegation | n.nn | n.nn |
| final | n.nn | n.nn |

## Divergence moments
(2-4 annotated transcript excerpts, quoted verbatim with run/conversation ids)

## Honest caveats
(model used; N; scenario is a designed demonstration per demo spec §8;
anything that favored the baseline or went wrong)
```

Per demo spec §8: if results are mixed, iterate the *scenario* (scripts/corpus, not the metrics or the judge) and re-run; record each iteration's changes in the results file under an "Iterations" heading.

- [ ] **Step 5: Update README status checkboxes** (`- [x]` for package, harness, results) and commit

```bash
git add results/*.md README.md
git commit -m "docs: first scored demo results (N=3, blinded judge)"
git push
```

---

## Explicitly deferred (separate plans)

- **Explainer animation** — needs real results first; will be planned once this plan's results exist.
- **Medium post + publish sign-off** — requires your explicit go-ahead; not part of any implementation plan.
- **Full-design features not in demo scope** — compaction, async embedding worker, fail-open error handling, handoff-style sub-agent support.

## Self-Review Notes

- Spec coverage: schema/§4 → Tasks 3-4; capture paths 1/2/3/4 (§5) → Tasks 6-8; fork/rollup §5.1 → Task 8; retrieval rules §5.2 → Tasks 5, 7; idempotency §5.4 → Tasks 2, 4, 7; demo conditions §3 → Task 10; measurement §6 → Tasks 11-12; publication stance §8 → Task 13 Step 4. Compaction §6 and async embeddings §7 of the core design: deferred by demo spec §4 — intentionally no tasks.
- Known API-drift risks are flagged inline (Task 7 note, Task 10 notes) with the rule: adapt to the installed API, keep the behavioral contract.
- Type consistency: `BeadsStore` method names (`write_fact`, `add_edge`, `search`, `fork_namespace`, `ancestor_chain`, `facts_in_namespace`, `resolve_short_id`) are used identically across Tasks 4-12.
```
