"""Best-effort redaction for explicitly labelled authentication secrets.

This standard-library-only helper recognizes common labelled password, PIN,
CVV/CVC, OTP, passcode, and verification-code forms. It intentionally fails
closed when a labelled value cannot be separated safely from nearby prose.
Pattern-based detection is not exhaustive and cannot guarantee detection of
every secret disclosure.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


class CredentialRedactionError(ValueError):
    """Raised when an explicitly labelled credential cannot be safely split."""


@dataclass(frozen=True)
class CredentialRedaction:
    """Sanitized text and value-free credential disclosure metadata."""

    sanitized_text: str
    value_spans: tuple[tuple[int, int], ...]
    categories: tuple[str, ...]


_LABEL_RE = re.compile(
    r"\b(?P<label>one[- ]time\s+(?:password|passcode)|verification\s+code|"
    r"security\s+code|password|passcode|pin|cvv|cvc|otp)\b",
    re.IGNORECASE,
)
_CONNECTOR_RE = re.compile(r"(?:(?:is|was)\b\s*|[:=]\s*)", re.IGNORECASE)
_QUOTED_VALUE_RE = re.compile(r"(?P<quote>[\"'])(?P<value>[^\"']+)(?P=quote)")
_UNQUOTED_VALUE_RE = re.compile(r"[^\s,;]+")
_SAFE_CONTEXT_RE = re.compile(
    r"(?:and|but)\s+(?:i|my|we|our|the|a|an|it|this|that|they|then|"
    r"one[- ]time|verification|security|password|passcode|pin|cvv|cvc|otp)\b|"
    r"(?:i|my|we|our|the|a|an|it|this|that|they|because|so|when|while|which)\b",
    re.IGNORECASE,
)
_CATEGORY_MAP = {
    "password": "PASSWORD",
    "passcode": "PASSCODE",
    "pin": "PIN",
    "cvv": "CVV",
    "cvc": "CVV",
    "otp": "OTP",
    "one-time password": "PASSWORD",
    "one time password": "PASSWORD",
    "one-time passcode": "PASSCODE",
    "one time passcode": "PASSCODE",
    "verification code": "OTP",
    "security code": "CVV",
}


def redact_credential_values(text: str) -> CredentialRedaction:
    """Redact values after recognizable credential labels.

    Supported forms include ``PIN: 1234``, ``OTP is 739184``,
    ``password=word123!``, quoted values with spaces, and whitespace-separated
    numeric PIN/CVV/OTP/verification-code values. A value is considered safely
    separated when it is quoted, ends at punctuation/end-of-text, or is
    followed by a recognizable prose boundary. Ambiguous unquoted multiword
    values raise :class:`CredentialRedactionError`; callers must not persist
    the original text in that case.

    The returned metadata contains character spans and category names only; it
    never contains the credential value.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")

    spans: list[tuple[int, int]] = []
    categories: set[str] = set()
    occupied_until = -1

    for label_match in _LABEL_RE.finditer(text):
        if label_match.start() < occupied_until:
            continue

        label = re.sub(r"\s+", " ", label_match.group("label").lower())
        category = _CATEGORY_MAP[label]
        cursor = label_match.end()
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1

        connector = _CONNECTOR_RE.match(text, cursor)
        if connector:
            cursor = connector.end()
            while cursor < len(text) and text[cursor].isspace():
                cursor += 1
        elif category not in {"PIN", "CVV", "OTP", "PASSCODE"}:
            # A label mentioned without an explicit value is not itself a
            # disclosure. Numeric credential categories also commonly use
            # the compact form "OTP 123456".
            continue

        if cursor >= len(text) or text[cursor] in ",;.!?":
            if connector:
                raise CredentialRedactionError(
                    "A labelled credential value could not be safely separated; no registration was attempted."
                )
            continue

        quoted = _QUOTED_VALUE_RE.match(text, cursor)
        if text[cursor] in "\"'":
            if quoted is None:
                raise CredentialRedactionError(
                    "A labelled credential value could not be safely separated; no registration was attempted."
                )
            start, end = quoted.span()
            next_index = end
        else:
            value_match = _UNQUOTED_VALUE_RE.match(text, cursor)
            if value_match is None:
                if connector:
                    raise CredentialRedactionError(
                        "A labelled credential value could not be safely separated; no registration was attempted."
                    )
                continue
            start, end = value_match.span()
            next_index = end
            remainder = text[next_index:]
            terminal_punctuation = text[end - 1] in ".!?"
            if remainder and not remainder[0].isspace() and remainder[0] not in ",;.!?":
                raise CredentialRedactionError(
                    "A labelled credential value could not be safely separated; no registration was attempted."
                )
            if remainder and remainder[0].isspace() and not terminal_punctuation:
                following = remainder.lstrip()
                if following and not following[0] in ",;.!?":
                    if _SAFE_CONTEXT_RE.match(following) is None:
                        raise CredentialRedactionError(
                            "A labelled credential value could not be safely separated; no registration was attempted."
                        )

        spans.append((start, end))
        categories.add(category)
        occupied_until = end

    sanitized = text
    for start, end in reversed(spans):
        sanitized = sanitized[:start] + "[REDACTED]" + sanitized[end:]

    return CredentialRedaction(
        sanitized_text=sanitized,
        value_spans=tuple(spans),
        categories=tuple(sorted(categories)),
    )
