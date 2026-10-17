"""Local administrator credential verification for the separate admin app."""

from __future__ import annotations

import hmac
from collections.abc import MutableMapping
from typing import Any

from src.runtime_config import get_setting

_MIN_ADMIN_PASSWORD_LENGTH = 16


class AdminAuthenticator:
    """Compare submitted credentials with server-side environment settings.

    This demo verifier does not provide production session hardening, rate
    limiting, MFA, or deployment transport controls.
    """

    def __init__(
        self,
        *,
        username: str | None = None,
        password: str | None = None,
    ) -> None:
        configured_username = username if username is not None else get_setting("ADMIN_USERNAME")
        configured_password = password if password is not None else get_setting("ADMIN_PASSWORD")
        self._username = configured_username if isinstance(configured_username, str) else ""
        self._password = configured_password if isinstance(configured_password, str) else ""

    @property
    def is_configured(self) -> bool:
        """Whether both server-side credential values are present."""
        return bool(
            self._username
            and len(self._password) >= _MIN_ADMIN_PASSWORD_LENGTH
            and len(self._password) <= 1024
        )

    def authenticate(self, username: object, password: object) -> bool:
        """Constant-time compare both fields and return no credential details."""
        submitted_username = username if isinstance(username, str) else ""
        submitted_password = password if isinstance(password, str) else ""
        username_matches = hmac.compare_digest(submitted_username, self._username)
        password_matches = hmac.compare_digest(submitted_password, self._password)
        return self.is_configured and username_matches and password_matches


def clear_admin_session(session_state: MutableMapping[str, Any]) -> None:
    """Remove authenticated identity and dashboard-held complaint data on logout."""
    for key in list(session_state):
        if key == "_admin_authenticated" or key.startswith(("admin_", "_admin_")):
            session_state.pop(key, None)
