"""Validation and normalization for guest complaint contact details.

Contact information is used only to provide a follow-up channel. It is not an
account identifier, proof of identity, or authentication credential.
"""

from __future__ import annotations

import re

_LOCAL_PART = re.compile(r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+$")
_DOMAIN_LABEL = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
_PHONE_CHARS = re.compile(r"\+?[0-9\s().-]+\Z")


def _normalize_email(email: str) -> str:
    """Apply the same trimming, case, IDNA, and dotted-domain rules as auth."""
    candidate = email.strip()
    if not candidate or len(candidate) > 254 or candidate.count("@") != 1:
        raise ValueError("contact email is invalid.")
    local, domain = candidate.rsplit("@", 1)
    if (
        not local
        or len(local) > 64
        or not _LOCAL_PART.fullmatch(local)
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
        or not domain
    ):
        raise ValueError("contact email is invalid.")
    try:
        ascii_domain = domain.encode("idna").decode("ascii").casefold()
    except UnicodeError:
        raise ValueError("contact email is invalid.") from None
    labels = ascii_domain.split(".")
    if (
        len(labels) < 2
        or any(not _DOMAIN_LABEL.fullmatch(label) for label in labels)
        or len(ascii_domain) > 253
    ):
        raise ValueError("contact email is invalid.")
    normalized = f"{local.casefold()}@{ascii_domain}"
    if len(normalized) > 254:
        raise ValueError("contact email is invalid.")
    return normalized


def normalize_contact_details(
    email: str | None, phone: str | None, *, require_one: bool = True
) -> tuple[str | None, str | None]:
    """Normalize optional email/phone and require at least one valid method.

    Phone numbers retain an optional leading plus and otherwise contain only
    digits. Separators commonly used for readability are removed. The accepted
    format is intentionally limited to 7–15 digits; extensions are unsupported.
    """
    if email is not None and not isinstance(email, str):
        raise TypeError("contact email must be a string or None.")
    if phone is not None and not isinstance(phone, str):
        raise TypeError("contact phone must be a string or None.")

    normalized_email: str | None = None
    if email is not None and email.strip():
        normalized_email = _normalize_email(email)

    normalized_phone: str | None = None
    if phone is not None and phone.strip():
        candidate = phone.strip()
        if not _PHONE_CHARS.fullmatch(candidate):
            raise ValueError("contact phone is invalid.")
        has_plus = candidate.startswith("+")
        digits = re.sub(r"\D", "", candidate)
        if not 7 <= len(digits) <= 15:
            raise ValueError("contact phone must contain 7 to 15 digits.")
        normalized_phone = ("+" if has_plus else "") + digits

    if require_one and normalized_email is None and normalized_phone is None:
        raise ValueError("provide an email address, a phone number, or both.")
    return normalized_email, normalized_phone
