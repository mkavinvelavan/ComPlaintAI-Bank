-- Allow public guest submissions and retain a requested follow-up channel.
-- Apply after 001_create_complaints.sql and 002_create_customers.sql.
ALTER TABLE complaints
    ALTER COLUMN customer_id DROP NOT NULL;

ALTER TABLE complaints
    ADD COLUMN contact_email VARCHAR(254),
    ADD COLUMN contact_phone VARCHAR(16);

ALTER TABLE complaints
    ADD CONSTRAINT complaints_contact_method_required
        CHECK (customer_id IS NOT NULL OR contact_email IS NOT NULL OR contact_phone IS NOT NULL),
    ADD CONSTRAINT complaints_contact_email_normalized
        CHECK (contact_email IS NULL OR contact_email = lower(btrim(contact_email))),
    ADD CONSTRAINT complaints_contact_phone_normalized
        CHECK (contact_phone IS NULL OR contact_phone ~ '^\+?[0-9]{7,15}$');

CREATE INDEX complaints_contact_email_created_idx
    ON complaints (contact_email, created_at DESC)
    WHERE contact_email IS NOT NULL;

CREATE INDEX complaints_contact_phone_created_idx
    ON complaints (contact_phone, created_at DESC)
    WHERE contact_phone IS NOT NULL;
