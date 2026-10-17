"""Local guard for policy questions requesting authentication secrets."""

from __future__ import annotations

import re

_SECRET_TYPE = (
    r"(?:passwords?|passcodes?|PINs?|CVVs?|CVCs?|OTPs?|one[- ]time\s+(?:passwords?|codes?)|"
    r"verification\s+codes?|security\s+codes?|access\s+tokens?|recovery\s+codes?|"
    r"authentication\s+secrets?)"
)
_SECRET_DISCLOSURE = re.compile(
    rf"\b{_SECRET_TYPE}\b\s*(?:(?:is|was)\s+|[:=#]\s*|\s+)"
    r"(?:['\"]?[A-Za-z0-9][A-Za-z0-9._-]{2,}['\"]?)",
    re.IGNORECASE,
)
_SECRET_REQUEST_PATTERNS = (
    re.compile(
        rf"\b(?:repeat|reveal|recite|read\s+back|echo|display|print|show|provide|retrieve|return)"
        rf"\b[\w\s'’-]{{0,100}}\b{_SECRET_TYPE}\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b{_SECRET_TYPE}\b[\w\s'’-]{{0,80}}"
        r"\b(?:repeat|reveal|recite|read\s+back|echo|display|print|show|provide|retrieve|return)\b",
        re.IGNORECASE,
    ),
    re.compile(
        rf"\b(?:what\s+(?:is|are)|tell\s+me|give\s+me)\b[\w\s'’-]{{0,60}}\b{_SECRET_TYPE}\b",
        re.IGNORECASE,
    ),
)

SENSITIVE_QUERY_RESPONSE = (
    "I can’t retrieve or repeat passwords, PINs, CVVs, one-time codes, passcodes, "
    "or other authentication secrets."
)


def is_sensitive_authentication_request(question: str) -> bool:
    """Return true for labelled secret disclosures or requests to repeat secrets."""
    if _SECRET_DISCLOSURE.search(question):
        return True
    return any(pattern.search(question) for pattern in _SECRET_REQUEST_PATTERNS)
