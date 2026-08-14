-- OxiGate initial schema: pgcrypto + spend_records with split cache-write columns.
-- Squashed from 0001_initial + 0002_spend_records.
-- org_id included from day one for multi-tenancy isolation.
-- Granular token columns: split 5m/1h Anthropic cache-write tiers.
-- All monetary values in nano-USD. All sentinel defaults = 'default'.
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE spend_records (
    id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    org_id                  TEXT        NOT NULL DEFAULT 'default',
    identity_id             TEXT        NOT NULL DEFAULT 'default',
    model                   TEXT        NOT NULL DEFAULT '',
    provider                TEXT        NOT NULL DEFAULT '',
    prompt_tokens           BIGINT      NOT NULL DEFAULT 0,
    completion_tokens       BIGINT      NOT NULL DEFAULT 0,
    cache_read_tokens       BIGINT      NOT NULL DEFAULT 0,
    cache_write_5m_tokens   BIGINT      NOT NULL DEFAULT 0,
    cache_write_1h_tokens   BIGINT      NOT NULL DEFAULT 0,
    thinking_tokens         BIGINT      NOT NULL DEFAULT 0,
    cost_nano_usd           BIGINT      NOT NULL DEFAULT 0,
    latency_ms              INTEGER     NOT NULL DEFAULT 0,
    tags                    JSONB       NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Primary query patterns: per-org + per-identity spend window queries.
CREATE INDEX idx_spend_records_org_identity_created
    ON spend_records (org_id, identity_id, created_at DESC);
