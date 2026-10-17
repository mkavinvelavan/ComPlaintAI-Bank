"""Coordinate complaint analysis with transactional complaint persistence.

The customer identifier is optional for public guest complaints. If supplied,
it is accepted only as data from a trusted server-side caller; this service
does not authenticate customers. The workflow's predicted
intent and internal priority/routing assessment are sent to the repository but
are not returned as customer-facing information. A predicted intent is not a
verified complaint finding.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import TypedDict

from src.complaint_repository import ComplaintRepository
from src.complaint_workflow import ComplaintWorkflow
from src.contact_validation import normalize_contact_details


_MAX_COMPLAINT_LENGTH = 50_000


class ComplaintRegistrationResult(TypedDict):
    """Minimal success result returned only after persistence confirms commit."""

    registered: bool
    case_reference: str


class ComplaintRegistrationError(RuntimeError):
    """Safe service-level error that never includes input or dependency details."""


class ComplaintRegistrationService:
    """Run the existing analysis workflow and persist its result transactionally.

    ``customer_id`` is optional for guests. If supplied by another trusted
    caller, this class still does not authenticate or authorize that caller.
    Public guest complaints require at least one valid contact method. When no
    repository is injected, its constructor is deferred until
    registration is attempted, so constructing this service does not require a
    configured database or open a database connection.
    """

    def __init__(
        self,
        workflow: ComplaintWorkflow | None = None,
        repository: ComplaintRepository | None = None,
        *,
        repository_factory: Callable[[], ComplaintRepository] | None = None,
    ) -> None:
        """Accept reusable dependencies or defer their default construction."""
        if repository is not None and repository_factory is not None:
            raise ValueError("Provide either repository or repository_factory, not both.")
        self.workflow = workflow if workflow is not None else ComplaintWorkflow()
        self.repository = repository
        self._repository_factory = repository_factory or ComplaintRepository

    @staticmethod
    def _validate_inputs(
        customer_id: str | None,
        complaint_text: str,
        contact_email: str | None,
        contact_phone: str | None,
    ) -> tuple[str | None, str | None]:
        if customer_id is not None:
            if not isinstance(customer_id, str):
                raise TypeError("customer_id must be a string or None.")
            if not customer_id.strip():
                raise ValueError("customer_id must not be empty or whitespace only.")
            if len(customer_id) > 256 or any(ord(char) < 32 for char in customer_id):
                raise ValueError("customer_id must be at most 256 characters without control characters.")

        if not isinstance(complaint_text, str):
            raise TypeError("complaint_text must be a string.")
        if not complaint_text.strip():
            raise ValueError("complaint_text must not be empty or whitespace only.")
        if len(complaint_text) > _MAX_COMPLAINT_LENGTH:
            raise ValueError(f"complaint_text must not exceed {_MAX_COMPLAINT_LENGTH} characters.")
        return normalize_contact_details(
            contact_email,
            contact_phone,
            require_one=customer_id is None,
        )

    def _get_repository(self) -> ComplaintRepository:
        if self.repository is None:
            self.repository = self._repository_factory()
        return self.repository

    def register_complaint(
        self,
        customer_id: str | None,
        complaint_text: str,
        *,
        contact_email: str | None = None,
        contact_phone: str | None = None,
    ) -> ComplaintRegistrationResult:
        """Analyze and persist a complaint, returning a reference after commit.

        Guest submissions require an email or phone contact channel. Contact
        information is not proof of identity. Input validation occurs before
        analysis. Analysis or persistence errors
        are replaced with safe service-level messages. No complaint text,
        credentials, model output, or raw dependency exception is logged or
        included in an error result.

        Raises:
            TypeError: If either input is not a string.
            ValueError: If either input is invalid.
            ComplaintRegistrationError: If analysis or persistence fails.
        """
        normalized_email, normalized_phone = self._validate_inputs(
            customer_id, complaint_text, contact_email, contact_phone
        )

        try:
            assessment = self.workflow.assess_complaint(complaint_text)
        except Exception:
            raise ComplaintRegistrationError(
                "Complaint analysis could not be completed; registration was not attempted."
            ) from None

        if not isinstance(assessment, Mapping):
            raise ComplaintRegistrationError(
                "Complaint analysis could not be completed; registration was not attempted."
            ) from None

        try:
            repository = self._get_repository()
            case_reference = repository.register_complaint(
                customer_id,
                complaint_text,
                assessment,
                contact_email=normalized_email,
                contact_phone=normalized_phone,
            )
        except Exception:
            raise ComplaintRegistrationError(
                "Complaint registration could not be confirmed; no case reference is available."
            ) from None

        if not isinstance(case_reference, str) or not case_reference:
            # A repository contract violation is not confirmation of a valid
            # reference, even if the call returned normally.
            raise ComplaintRegistrationError(
                "Complaint registration could not be confirmed; no case reference is available."
            ) from None

        return {"registered": True, "case_reference": case_reference}
