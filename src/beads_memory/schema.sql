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
    -- 'directive': a question, instruction or stated goal. Captured and kept
    -- queryable because it is the provenance of downstream choices, but held
    -- out of default retrieval: injecting text that is nearly the current
    -- query wastes a top-K slot that a claim would use.
    kind                text NOT NULL CHECK (kind IN
                        ('user_input','conclusion','summary','directive')),
    body                text NOT NULL,
    embedding           vector(768),
    -- The embedding of the text this claim was carved OUT of, shared by every
    -- claim from the same capture. Storage granularity and retrieval
    -- granularity are not the same problem: a claim is the right unit to store
    -- (it is what `supersedes` acts on) and the wrong unit to embed. Measured
    -- on the incident scenario — the baseline's 518-char document sat at cosine
    -- distance 0.538 from the conv-3 query while the best 128-char claim carved
    -- from the same material sat at 0.562, putting the root cause at rank 30 of
    -- ~90 and outside every top-8 injection in three runs.
    --
    -- Nullable: facts written before this column existed, and any capture with
    -- no wider context than the claim itself, rank on `embedding` alone via the
    -- COALESCE in search().
    context_embedding   vector(768),
    status              text NOT NULL DEFAULT 'active'
                        CHECK (status IN ('active','superseded','archived')),
    -- 'tool_result': what a tool returned, captured into the namespace of the
    -- agent that called it. Memory tools are excluded at the middleware, since
    -- capturing retrieval output would feed the store back into itself.
    source              text NOT NULL CHECK (source IN
                        ('passive_capture','remember_tool','conclude_task',
                         'fallback_conclude','compaction','tool_result')),
    agent_id            text NOT NULL,
    acting_on_behalf_of text NOT NULL,
    created_at          timestamptz NOT NULL DEFAULT now()
);
-- Additive, so an existing database picks the column up without a migration
-- step. init_schema() runs this whole file and is documented as idempotent.
ALTER TABLE facts ADD COLUMN IF NOT EXISTS context_embedding vector(768);

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
