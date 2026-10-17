"""Deterministic extraction of selected entities from complaint text.

This module uses a blank English spaCy pipeline with explicit ``Matcher`` cue
patterns and regular expressions. It does not load or download a pretrained
language model, log complaint text, infer complaint priority, or retain input.

Pattern limitations: only supported currency codes/symbols and common numeric
or English month-name dates are recognized. Numeric dates are locale-ambiguous.
Transaction references require an explicit transaction label and a value with
at least one digit. Card/account suffixes require explicit "ending" wording.
Credential disclosure detection is a conservative safety filter, not a full
secret scanner; it reports categories only and removes overlapping values.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import TypedDict

import spacy
from spacy.matcher import Matcher
from spacy.tokens import Doc

from src.credential_redaction import redact_credential_values


class ExtractedEntity(TypedDict):
    """A recognized entity and its half-open character range in the input."""

    text: str
    entity_type: str
    start: int
    end: int


class SecurityFlags(TypedDict):
    """Credential disclosure indicators that deliberately omit secret values."""

    credential_disclosed: bool
    credential_categories: list[str]


class ExtractionResult(TypedDict):
    """Result returned by :meth:`SpacyEntityExtractor.extract`."""

    entities: list[ExtractedEntity]
    security_flags: SecurityFlags


_DATE_PATTERN = re.compile(
    r"(?<!\w)(?:"
    r"\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|"
    r"\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4}|"
    r"\d{1,2}(?:st|nd|rd|th)?\s+"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{4}|"
    r"(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2}(?:st|nd|rd|th)?,?\s+\d{4}"
    r")(?!\w)",
    re.IGNORECASE,
)
_MONEY_PATTERN = re.compile(
    r"(?<!\w)(?:(?:AED|USD|EUR|GBP|INR|CAD|AUD)\s*|[₹$€£])"
    r"\d[\d,]*(?:\.\d{1,2})?(?!\w)",
    re.IGNORECASE,
)
_VALUE_AFTER_CUE = re.compile(
    r"\s*(?:(?:[:#=])\s*|(?:is\s+))?(?P<value>[A-Za-z0-9][A-Za-z0-9-]{2,})",
    re.IGNORECASE,
)
_LAST_FOUR_AFTER_CUE = re.compile(
    r"\s*(?:(?:in|with)\s+)?(?P<value>\d{4})(?!\d)", re.IGNORECASE
)
_ORDINAL_SUFFIX = re.compile(r"(?<=\d)(?:st|nd|rd|th)\b", re.IGNORECASE)


def _is_valid_date(candidate: str) -> bool:
    """Return whether a recognized date-shaped string is a calendar date."""
    normalized = _ORDINAL_SUFFIX.sub("", candidate).replace(",", "")
    formats = (
        "%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d",
        "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y", "%m-%d-%Y",
        "%d.%m.%Y", "%m.%d.%Y", "%d/%m/%y", "%m/%d/%y",
        "%d-%m-%y", "%m-%d-%y", "%d.%m.%y", "%m.%d.%y",
        "%d %B %Y", "%d %b %Y", "%B %d %Y", "%b %d %Y",
    )
    for date_format in formats:
        try:
            datetime.strptime(normalized, date_format)
            return True
        except ValueError:
            continue
    return False


def _credential_matches(text: str) -> tuple[list[tuple[int, int]], list[str]]:
    """Find redaction-safe credential spans and return category names only."""
    result = redact_credential_values(text)
    return list(result.value_spans), list(result.categories)


class SpacyEntityExtractor:
    """Extract explicitly patterned references, amounts, dates, and suffixes.

    Each call returns ``entities`` with original-input offsets, plus
    ``security_flags`` containing only a disclosure Boolean and credential
    category names. Sensitive credential values are never returned.
    """

    def __init__(self) -> None:
        """Build a blank English pipeline and deterministic cue matcher."""
        self.nlp = spacy.blank("en")
        self.matcher = Matcher(self.nlp.vocab)
        self.matcher.add(
            "TRANSACTION_LABEL",
            [[{"LOWER": "transaction"}, {"LOWER": {"IN": ["id", "reference", "ref"]}}]],
        )
        self.matcher.add(
            "CARD_ENDING_LABEL",
            [[{"LOWER": "card"}, {"LOWER": {"IN": ["ending", "ends"]}}]],
        )
        self.matcher.add(
            "ACCOUNT_ENDING_LABEL",
            [[{"LOWER": "account"}, {"LOWER": {"IN": ["ending", "ends"]}}]],
        )

    @staticmethod
    def _resolve_overlaps(candidates: list[ExtractedEntity]) -> list[ExtractedEntity]:
        """Deduplicate and select non-overlapping matches deterministically."""
        priority = {
            "CARD_LAST4": 0,
            "ACCOUNT_LAST4": 0,
            "TRANSACTION_REFERENCE": 1,
            "MONEY_AMOUNT": 2,
            "DATE": 3,
        }
        unique: dict[tuple[int, int, str], ExtractedEntity] = {}
        for entity in candidates:
            key = (entity["start"], entity["end"], entity["entity_type"])
            unique.setdefault(key, entity)

        ordered = sorted(
            unique.values(),
            key=lambda entity: (
                priority.get(entity["entity_type"], 99),
                -(entity["end"] - entity["start"]),
                entity["start"],
            ),
        )
        selected: list[ExtractedEntity] = []
        for candidate in ordered:
            if any(
                candidate["start"] < existing["end"]
                and existing["start"] < candidate["end"]
                for existing in selected
            ):
                continue
            selected.append(candidate)
        return sorted(selected, key=lambda entity: (entity["start"], entity["end"]))

    def extract(self, text: str) -> ExtractionResult:
        """Extract supported entities while preserving source character offsets.

        Args:
            text: Complaint text to inspect. The value is not logged or stored.

        Raises:
            TypeError: If ``text`` is not a string.
            ValueError: If ``text`` is empty or contains only whitespace.
        """
        if not isinstance(text, str):
            raise TypeError("text must be a string.")
        if not text.strip():
            raise ValueError("text must not be empty or whitespace only.")

        credential_spans, credential_categories = _credential_matches(text)
        doc: Doc = self.nlp.make_doc(text)
        candidates: list[ExtractedEntity] = []

        for match_id, start_token, end_token in self.matcher(doc):
            label = self.nlp.vocab.strings[match_id]
            cue = doc[start_token:end_token]
            if label == "TRANSACTION_LABEL":
                value_match = _VALUE_AFTER_CUE.match(text, cue.end_char)
                if value_match:
                    value = value_match.group("value")
                    # Requiring a digit avoids treating prose such as "was" as an ID.
                    if any(char.isdigit() for char in value):
                        start, end = value_match.span("value")
                        candidates.append(
                            {
                                "text": text[start:end],
                                "entity_type": "TRANSACTION_REFERENCE",
                                "start": start,
                                "end": end,
                            }
                        )
            elif label in {"CARD_ENDING_LABEL", "ACCOUNT_ENDING_LABEL"}:
                value_match = _LAST_FOUR_AFTER_CUE.match(text, cue.end_char)
                if value_match:
                    start, end = value_match.span("value")
                    candidates.append(
                        {
                            "text": text[start:end],
                            "entity_type": (
                                "CARD_LAST4" if label == "CARD_ENDING_LABEL" else "ACCOUNT_LAST4"
                            ),
                            "start": start,
                            "end": end,
                        }
                    )

        for match in _MONEY_PATTERN.finditer(text):
            start, end = match.span()
            candidates.append(
                {
                    "text": text[start:end],
                    "entity_type": "MONEY_AMOUNT",
                    "start": start,
                    "end": end,
                }
            )

        for match in _DATE_PATTERN.finditer(text):
            if not _is_valid_date(match.group()):
                continue
            start, end = match.span()
            candidates.append(
                {"text": text[start:end], "entity_type": "DATE", "start": start, "end": end}
            )

        # A credential-shaped span always takes precedence over an entity match.
        safe_candidates = [
            candidate
            for candidate in candidates
            if not any(
                candidate["start"] < secret_end and secret_start < candidate["end"]
                for secret_start, secret_end in credential_spans
            )
        ]

        return {
            "entities": self._resolve_overlaps(safe_candidates),
            "security_flags": {
                "credential_disclosed": bool(credential_categories),
                "credential_categories": credential_categories,
            },
        }
