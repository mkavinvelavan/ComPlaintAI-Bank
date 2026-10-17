-- Initial complaint-registration schema. No complaint data is seeded here.
CREATE TABLE IF NOT EXISTS complaints (
    id UUID PRIMARY KEY,
    case_reference TEXT NOT NULL,
    customer_id TEXT NOT NULL
        CHECK (char_length(btrim(customer_id)) BETWEEN 1 AND 256),
    complaint_text TEXT NOT NULL
        CHECK (char_length(btrim(complaint_text)) BETWEEN 1 AND 50000),
    predicted_intent VARCHAR(128),
    priority VARCHAR(8)
        CHECK (priority IS NULL OR priority IN ('LOW', 'MEDIUM', 'HIGH')),
    department_suggestion VARCHAR(128),
    credential_disclosure_detected BOOLEAN NOT NULL DEFAULT FALSE,
    status VARCHAR(16) NOT NULL DEFAULT 'RECEIVED'
        CHECK (status IN ('RECEIVED', 'IN_REVIEW', 'RESOLVED', 'CLOSED')),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT complaints_case_reference_key UNIQUE (case_reference),
    CONSTRAINT complaints_case_reference_format
        CHECK (case_reference ~ '^CPB-[0-9]{8}-[A-F0-9]{10}$')
);

CREATE INDEX IF NOT EXISTS complaints_customer_created_idx
    ON complaints (customer_id, created_at DESC);

CREATE INDEX IF NOT EXISTS complaints_status_created_idx
    ON complaints (status, created_at DESC);

CREATE OR REPLACE FUNCTION complaints_update_timestamp()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS complaints_set_updated_at ON complaints;
CREATE TRIGGER complaints_set_updated_at
    BEFORE UPDATE ON complaints
    FOR EACH ROW
    EXECUTE FUNCTION complaints_update_timestamp();
