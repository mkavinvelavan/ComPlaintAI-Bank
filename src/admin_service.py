"""Authorization boundary for administrator complaint operations."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from src.admin_repository import ALLOWED_PRIORITIES, ALLOWED_STATUSES


class AdminAuthorizationError(PermissionError):
    """Raised when the current server-side session is not authenticated."""


class AdminOperationError(RuntimeError):
    """Safe error exposed to the admin interface."""


class AdminComplaintService:
    """Require authenticated server-side session state for every DB operation."""

    def __init__(self, repository: Any, *, is_authenticated: Callable[[], bool]) -> None:
        if not callable(is_authenticated):
            raise TypeError("is_authenticated must be callable.")
        self._repository = repository
        self._is_authenticated = is_authenticated

    def _authorize(self) -> None:
        try:
            authenticated = self._is_authenticated() is True
        except Exception:
            authenticated = False
        if not authenticated:
            raise AdminAuthorizationError("Administrator authentication is required.")

    def get_overview(self) -> dict[str, Any]:
        self._authorize()
        try:
            return self._repository.get_overview()
        except Exception:
            raise AdminOperationError("The complaint overview could not be loaded.") from None

    def list_complaints(
        self,
        *,
        case_reference: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        self._authorize()
        if status is not None and status not in ALLOWED_STATUSES:
            raise ValueError("Select a status permitted by the complaints schema.")
        if priority is not None and priority not in ALLOWED_PRIORITIES:
            raise ValueError("Select a priority permitted by the complaints schema.")
        try:
            return self._repository.list_complaints(
                case_reference=case_reference,
                status=status,
                priority=priority,
                limit=limit,
            )
        except ValueError:
            raise
        except Exception:
            raise AdminOperationError("The complaint list could not be loaded.") from None

    def get_complaint(self, case_reference: str) -> dict[str, Any] | None:
        self._authorize()
        try:
            return self._repository.get_complaint(case_reference)
        except Exception:
            raise AdminOperationError("The complaint detail could not be loaded.") from None

    def update_status(self, case_reference: str, status: str) -> dict[str, Any] | None:
        self._authorize()
        if status not in ALLOWED_STATUSES:
            raise ValueError("Select a status permitted by the complaints schema.")
        try:
            return self._repository.update_status(case_reference, status)
        except ValueError:
            raise
        except Exception:
            raise AdminOperationError("The complaint status could not be updated.") from None
