-- OxiGate initial schema: pgcrypto + spend_records with generalized cache-write accounting.
-- Squashed from 0001_initial + 0002_spend_records.
-- org_id included from day one for multi-tenancy isolation.
-- Cache-write tokens are accounted by class in the application layer, not split into fixed
-- SQL columns; the row carries a cost-confidence status and optional evidence document instead.
-- All monetary values in nano-USD. All sentinel defaults = 'default'.
-- NOTE: this file was revised in place after v0.1.0. A database created by v0.1.0 fails
-- the startup migration on a checksum mismatch and must be recreated — see README,
-- "Upgrading from v0.1.0".
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
    thinking_tokens         BIGINT      NOT NULL DEFAULT 0,
    cost_nano_usd           BIGINT      NOT NULL DEFAULT 0,
    cost_status             TEXT        NOT NULL,
    usage_evidence          JSONB       NULL,
    latency_ms              INTEGER     NOT NULL DEFAULT 0,
    tags                    JSONB       NOT NULL DEFAULT '{}',
    created_at              TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Primary query patterns: per-org + per-identity spend window queries.
CREATE INDEX idx_spend_records_org_identity_created
    ON spend_records (org_id, identity_id, created_at DESC);
