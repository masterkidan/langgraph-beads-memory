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
