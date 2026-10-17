"""Transactional PostgreSQL boundary for complaint registration.

This module does not run complaint analysis, expose administrator operations,
or retain extracted entity values. Original complaint text is stored as
sensitive data. A case reference is returned only after PostgreSQL confirms
the transaction commit.
"""

from __future__ import annotations

import re
import secrets
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from src.credential_redaction import CredentialRedactionError, redact_credential_values
from src.contact_validation import normalize_contact_details
from src.runtime_config import get_setting

_MAX_COMPLAINT_LENGTH = 50_000
_MAX_CASE_REFERENCE_ATTEMPTS = 3
_CASE_REFERENCE_PATTERN = re.compile(r"CPB-[0-9]{8}-[A-F0-9]{10}\Z")
_ALLOWED_PRIORITIES = {"LOW", "MEDIUM", "HIGH"}

_INSERT_COMPLAINT_SQL = """
INSERT INTO complaints (
    id,
    case_reference,
    customer_id,
    contact_email,
    contact_phone,
    complaint_text,
    predicted_intent,
    priority,
    department_suggestion,
    credential_disclosure_detected
)
VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
RETURNING id
"""


class ComplaintPersistenceError(RuntimeError):
    """Raised when registration cannot be confirmed without exposing DB data."""


class _CaseReferenceCollision(Exception):
    """Internal signal to retry only the named case-reference constraint."""


def _default_connection_factory(database_url: str) -> Any:
    """Connect lazily so importing and testing this module needs no live DB."""
    try:
        import psycopg
    except ImportError:
        raise ComplaintPersistenceError(
            "The psycopg PostgreSQL driver is required for database registration."
        ) from None
    return psycopg.connect(database_url)


def _is_case_reference_collision(error: Exception) -> bool:
    """Recognize only a unique violation of this table's reference constraint."""
    diagnostic = getattr(error, "diag", None)
    return (
        getattr(error, "sqlstate", None) == "23505"
        and getattr(diagnostic, "constraint_name", None) == "complaints_case_reference_key"
    )


class ComplaintRepository:
    """Validate and transactionally persist a complaint registration.

    Guest complaints may use ``customer_id=None`` and must provide at least one
    validated contact method. Contact information is not authentication.
    ``assessment`` accepts the existing ``ComplaintWorkflow`` result. Only its
    predicted intent, internal priority/routing suggestion, and Boolean
    credential-disclosure indicator are persisted. Confidence scores,
    extracted entity values, credential categories, and raw model output are
    deliberately omitted. Explicitly labelled credential values are redacted
    before persistence. Pattern-based redaction is not exhaustive; ambiguous
    labelled values fail closed rather than saving the original text.

    The customer identifier must come from a trusted caller. This repository
    does not authenticate customers or authorize access to complaint records.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        connection_factory: Callable[[str], Any] | None = None,
        case_reference_factory: Callable[[], str] | None = None,
    ) -> None:
        """Configure a database URL and injectable connection/reference factories."""
        configured_url = database_url if database_url is not None else get_setting("DATABASE_URL")
        if not isinstance(configured_url, str) or not configured_url.strip():
            raise ValueError("DATABASE_URL must be configured before database registration.")
        self._database_url = configured_url.strip()
        self._connection_factory = connection_factory or _default_connection_factory
        self._case_reference_factory = case_reference_factory or self._new_case_reference

    @staticmethod
    def _new_case_reference() -> str:
        date_part = datetime.now(timezone.utc).strftime("%Y%m%d")
        return f"CPB-{date_part}-{secrets.token_hex(5).upper()}"

    @staticmethod
    def _validate_registration(
        customer_id: str | None,
        complaint_text: str,
        assessment: Mapping[str, Any] | None,
        contact_email: str | None,
        contact_phone: str | None,
    ) -> tuple[str | None, str | None, str | None, bool, str | None, str | None]:
        if customer_id is not None:
            if not isinstance(customer_id, str):
                raise TypeError("customer_id must be a string or None.")
            if not customer_id.strip():
                raise ValueError("customer_id must not be empty or whitespace only.")
            if len(customer_id) > 256 or any(ord(char) < 32 for char in customer_id):
                raise ValueError("customer_id must be at most 256 characters without control characters.")

        normalized_email, normalized_phone = normalize_contact_details(
            contact_email, contact_phone, require_one=customer_id is None
        )

        if not isinstance(complaint_text, str):
            raise TypeError("complaint_text must be a string.")
        if not complaint_text.strip():
            raise ValueError("complaint_text must not be empty or whitespace only.")
        if len(complaint_text) > _MAX_COMPLAINT_LENGTH:
            raise ValueError(f"complaint_text must not exceed {_MAX_COMPLAINT_LENGTH} characters.")

        if assessment is None:
            return None, None, None, False, normalized_email, normalized_phone
        if not isinstance(assessment, Mapping):
            raise TypeError("assessment must be a mapping or None.")

        classification = assessment.get("classification", assessment)
        priority_assessment = assessment.get("priority_assessment", {})
        security_disclosure = assessment.get("security_disclosure", {})
        if not isinstance(classification, Mapping):
            raise TypeError("assessment classification must be a mapping.")
        if not isinstance(priority_assessment, Mapping):
            raise TypeError("assessment priority_assessment must be a mapping.")
        if not isinstance(security_disclosure, Mapping):
            raise TypeError("assessment security_disclosure must be a mapping.")

        intent = classification.get("predicted_intent")
        if intent is not None and (
            not isinstance(intent, str)
            or not intent.strip()
            or len(intent) > 128
            or any(ord(char) < 32 for char in intent)
        ):
            raise ValueError("predicted_intent must be a non-empty string of at most 128 characters.")

        priority = priority_assessment.get("priority")
        if priority is not None:
            if not isinstance(priority, str) or priority.upper() not in _ALLOWED_PRIORITIES:
                raise ValueError("priority must be LOW, MEDIUM, HIGH, or None.")
            priority = priority.upper()

        department = priority_assessment.get("department_suggestion")
        if department is not None and (
            not isinstance(department, str)
            or not department.strip()
            or len(department) > 128
            or any(ord(char) < 32 for char in department)
        ):
            raise ValueError("department_suggestion must be a non-empty string of at most 128 characters.")

        disclosure = security_disclosure.get(
            "detected", assessment.get("sensitive_disclosure_detected", False)
        )
        if not isinstance(disclosure, bool):
            raise ValueError("credential-disclosure indicator must be a Boolean.")

        return intent, priority, department, disclosure, normalized_email, normalized_phone

    def register_complaint(
        self,
        customer_id: str | None,
        complaint_text: str,
        assessment: Mapping[str, Any] | None = None,
        *,
        contact_email: str | None = None,
        contact_phone: str | None = None,
    ) -> str:
        """Persist one complaint and return its reference only after commit.

        Validation happens before opening a connection. Database and commit
        exceptions are intentionally replaced with a fixed safe error message;
        no complaint text, credentials, URL, or exception content is logged.
        """
        (
            intent,
            priority,
            department,
            disclosure,
            normalized_email,
            normalized_phone,
        ) = self._validate_registration(
            customer_id, complaint_text, assessment, contact_email, contact_phone
        )
        try:
            redaction = redact_credential_values(complaint_text)
        except CredentialRedactionError:
            # Do not include the input or secret value in this safe rejection.
            raise ValueError(
                "A labelled credential value could not be safely separated; no registration was attempted."
            ) from None
        safe_complaint_text = redaction.sanitized_text
        disclosure = disclosure or bool(redaction.categories)
        internal_id: UUID = uuid4()

        for _ in range(_MAX_CASE_REFERENCE_ATTEMPTS):
            case_reference = self._case_reference_factory()
            if not isinstance(case_reference, str) or not _CASE_REFERENCE_PATTERN.fullmatch(
                case_reference
            ):
                raise ValueError("case_reference_factory returned an invalid reference format.")
            try:
                self._insert_once(
                    internal_id=internal_id,
                    case_reference=case_reference,
                    customer_id=customer_id,
                    contact_email=normalized_email,
                    contact_phone=normalized_phone,
                    complaint_text=safe_complaint_text,
                    predicted_intent=intent,
                    priority=priority,
                    department_suggestion=department,
                    credential_disclosure_detected=disclosure,
                )
            except _CaseReferenceCollision:
                continue
            return case_reference

        raise ComplaintPersistenceError(
            "Complaint registration could not be confirmed; no case reference was returned."
        ) from None

    def _insert_once(
        self,
        *,
        internal_id: UUID,
        case_reference: str,
        customer_id: str | None,
        contact_email: str | None,
        contact_phone: str | None,
        complaint_text: str,
        predicted_intent: str | None,
        priority: str | None,
        department_suggestion: str | None,
        credential_disclosure_detected: bool,
    ) -> None:
        connection: Any | None = None
        cursor: Any | None = None
        try:
            connection = self._connection_factory(self._database_url)
            cursor = connection.cursor()
            cursor.execute(
                _INSERT_COMPLAINT_SQL,
                (
                    internal_id,
                    case_reference,
                    customer_id,
                    contact_email,
                    contact_phone,
                    complaint_text,
                    predicted_intent,
                    priority,
                    department_suggestion,
                    credential_disclosure_detected,
                ),
            )
            inserted = cursor.fetchone()
            if not inserted:
                raise RuntimeError("insert returned no record")
            cursor.close()
            cursor = None
            connection.commit()
        except Exception as exc:
            collision = _is_case_reference_collision(exc)
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    pass
            if collision:
                raise _CaseReferenceCollision() from None
            raise ComplaintPersistenceError(
                "Complaint registration could not be confirmed; no case reference was returned."
            ) from None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass
