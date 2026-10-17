-- Customer accounts for authentication. This is additive to migration 001;
-- complaints.customer_id remains TEXT and stores the UUID's canonical string.
CREATE TABLE IF NOT EXISTS customers (
    customer_id UUID PRIMARY KEY,
    normalized_email VARCHAR(254) NOT NULL,
    password_hash TEXT NOT NULL,
    account_status VARCHAR(16) NOT NULL DEFAULT 'ACTIVE'
        CHECK (account_status IN ('ACTIVE', 'INACTIVE', 'SUSPENDED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT customers_normalized_email_key UNIQUE (normalized_email),
    CONSTRAINT customers_email_normalized
        CHECK (normalized_email = lower(btrim(normalized_email)))
);

CREATE INDEX IF NOT EXISTS customers_status_created_idx
    ON customers (account_status, created_at DESC);

CREATE OR REPLACE FUNCTION customers_update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS customers_set_updated_at ON customers;
CREATE TRIGGER customers_set_updated_at
    BEFORE UPDATE ON customers
    FOR EACH ROW
    EXECUTE FUNCTION customers_update_timestamp();
