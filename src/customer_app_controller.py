"""UI-independent controller for public guest complaint intake and policy FAQs."""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from src.contact_validation import normalize_contact_details
from src.sensitive_query_guard import (
    SENSITIVE_QUERY_RESPONSE,
    is_sensitive_authentication_request,
)

_CASE_REFERENCE_RE = re.compile(r"CPB-[0-9]{8}-[A-F0-9]{10}\Z")


def _new_registration_service() -> Any:
    from src.complaint_registration_service import ComplaintRegistrationService

    return ComplaintRegistrationService()


def _new_policy_assistant() -> Any:
    from src.deployment_resources import get_policy_retriever
    from src.policy_assistant import PolicyAssistant

    return PolicyAssistant(retriever=get_policy_retriever())


class CustomerAppController:
    """Coordinate public complaint intake and the separate policy FAQ path."""

    def __init__(
        self,
        *,
        registration_service: Any | None = None,
        policy_assistant: Any | None = None,
        registration_service_factory: Callable[[], Any] | None = None,
        policy_assistant_factory: Callable[[], Any] | None = None,
    ) -> None:
        self._registration_service = registration_service
        self._policy_assistant = policy_assistant
        self._registration_factory = registration_service_factory or _new_registration_service
        self._policy_factory = policy_assistant_factory or _new_policy_assistant
        self._previous_policy_answer: str | None = None

    def _registration(self) -> Any:
        if self._registration_service is None:
            self._registration_service = self._registration_factory()
        return self._registration_service

    def _policy(self) -> Any:
        if self._policy_assistant is None:
            self._policy_assistant = self._policy_factory()
        return self._policy_assistant

    def clear_policy_context(self) -> None:
        """Forget the prior cited answer when a new FAQ conversation starts."""
        self._previous_policy_answer = None

    @staticmethod
    def _result(status: str, message: str, **extra: Any) -> dict[str, Any]:
        return {"status": status, "message": message, **extra}

    def submit_complaint(
        self,
        complaint_text: str,
        email: str | None = None,
        phone: str | None = None,
    ) -> dict[str, Any]:
        """Analyze and submit a guest complaint with a requested contact method."""
        if not isinstance(complaint_text, str):
            return self._result("validation_error", "Please describe the banking issue you experienced.")
        if not complaint_text.strip():
            return self._result("validation_error", "Please tell me what happened so I can submit your complaint.")
        if len(complaint_text) > 50_000:
            return self._result("validation_error", "Please shorten the description to 50,000 characters or fewer.")

        try:
            normalized_email, normalized_phone = normalize_contact_details(email, phone)
        except (TypeError, ValueError) as exc:
            detail = str(exc)
            if detail.startswith("provide an email"):
                message = "Please provide an email address, a phone number, or both so we can follow up."
            elif "email" in detail:
                message = "That email address does not look valid. Please check it or provide a phone number."
            elif "phone" in detail:
                message = "That phone number does not look valid. Please check it or provide an email address."
            else:
                message = "Please provide a valid email address, a phone number, or both so we can contact you."
            return self._result("validation_error", message)

        try:
            # Guest records have no authenticated customer ID. Contact details
            # are follow-up channels only and are never treated as identity.
            result = self._registration().register_complaint(
                None,
                complaint_text,
                contact_email=normalized_email,
                contact_phone=normalized_phone,
            )
        except Exception:
            return self._result(
                "error",
                "I’m sorry, but I couldn’t confirm that your complaint was registered. Please try again later.",
            )

        if (
            not isinstance(result, dict)
            or result.get("registered") is not True
            or not isinstance(result.get("case_reference"), str)
            or not _CASE_REFERENCE_RE.fullmatch(result["case_reference"])
        ):
            return self._result(
                "error",
                "I’m sorry, but I couldn’t confirm that your complaint was registered. Please try again later.",
            )
        return self._result(
            "success",
            "Thank you for explaining what happened. Your complaint has been recorded.",
            case_reference=result["case_reference"],
        )

    def answer_policy_question(self, question: str) -> dict[str, Any]:
        """Answer a policy query, optionally using the prior grounded answer as context."""
        if not isinstance(question, str) or not question.strip():
            return self._result("validation_error", "What would you like to know about the policies?")
        if is_sensitive_authentication_request(question):
            return self._result(
                "success",
                "Policy answer",
                answer=SENSITIVE_QUERY_RESPONSE,
                sources=[],
            )
        contextual_question = question
        previous_answer = None
        # Only a prior cited assistant answer is used for follow-up context.
        # Prior customer messages and complaint/contact data are never included.
        if isinstance(question, str):
            previous_answer = self._previous_policy_answer
        if isinstance(previous_answer, str) and previous_answer.strip():
            contextual_question = (
                "Previous cited policy answer for context:\n"
                + previous_answer[-2_000:]
                + "\n\nCurrent policy question:\n"
                + question
            )
        try:
            result = self._policy().answer_question(contextual_question)
        except Exception:
            return self._result(
                "error",
                "I’m sorry, the policy assistant is temporarily unavailable. Please try again later.",
            )

        answer = result.get("answer") if isinstance(result, dict) else None
        raw_sources = result.get("sources", []) if isinstance(result, dict) else []
        if not isinstance(answer, str) or not answer.strip() or not isinstance(raw_sources, list):
            return self._result(
                "error",
                "I couldn’t prepare a supported policy answer just now. Please try again later.",
            )

        sources: list[dict[str, Any]] = []
        for source in raw_sources:
            if not isinstance(source, dict):
                continue
            policy_id = source.get("policy_id")
            filename = source.get("source_file")
            page = source.get("page")
            if (
                isinstance(policy_id, str)
                and re.fullmatch(r"CPB-POL-[0-9]{3}", policy_id)
                and isinstance(filename, str)
                and filename.strip()
                and isinstance(page, int)
                and not isinstance(page, bool)
                and page > 0
            ):
                sources.append(
                    {"policy_id": policy_id, "source_file": filename, "page": page}
                )
        self._previous_policy_answer = answer.strip() if sources else None
        return self._result("success", "Policy answer", answer=answer.strip(), sources=sources)
