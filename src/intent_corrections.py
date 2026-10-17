"""Narrow, deterministic corrections for clear card-delivery intent signals."""

from __future__ import annotations

import re


_CARD = re.compile(r"\b(?:debit|credit|physical|replacement|new)?\s*card\b", re.IGNORECASE)
_NOT_ARRIVED_AFTER_CARD = re.compile(
    r"\bcard\b.{0,50}\b(?:has not|hasn't|have not|haven't|is not|isn't|"
    r"was not|wasn't|did not|didn't|not yet|never)\b.{0,30}\b"
    r"(?:arriv\w*|deliver\w*|receiv\w*|come)\b",
    re.IGNORECASE,
)
_NOT_ARRIVED_BEFORE_CARD = re.compile(
    r"\b(?:has not|hasn't|have not|haven't|did not|didn't|not yet|never)\b"
    r".{0,45}\b(?:arriv\w*|deliver\w*|receiv\w*|get|got|come)\b.{0,35}\bcard\b",
    re.IGNORECASE,
)
_DID_NOT_GET_CARD_DELIVERED = re.compile(
    r"\b(?:did not|didn't)\s+(?:get|got|receive|received)\b.{0,45}\bcard\b"
    r".{0,25}\b(?:arriv\w*|deliver\w*|receiv\w*)\b",
    re.IGNORECASE,
)
_WAITING_FOR_CARD = re.compile(r"\b(?:still\s+)?wait(?:ing|ed)\s+for\b.{0,30}\bcard\b", re.I)
_DELAYED_CARD = re.compile(r"\bcard\b.{0,35}\b(?:delayed|late|overdue)\b", re.I)
_DELIVERY_QUESTION_CUE = re.compile(
    r"\b(?:how\s+long|how\s+many\s+days|when|delivery\s+estimate|"
    r"what(?:'s|\s+is)\s+(?:the\s+)?(?:usual\s+)?delivery\s+(?:time|estimate))\b",
    re.IGNORECASE,
)
_DELIVERY_CONTEXT = re.compile(r"\b(?:deliver\w*|arriv\w*|receiv\w*|ship\w*|post\w*|get)\b", re.I)
_CARD_EXPIRY_CONTEXT = re.compile(r"\b(?:expir\w*|validity|valid\s+until|last\s+for)\b", re.I)
_DELIVERY_CONFLICT = re.compile(
    r"\b(?:direct\s+debit|transaction|payment|withdrawal|transfer|charge)\b"
    r".{0,50}\b(?:unrecogni[sz]ed|not\s+(?:mine|recogn[iz]ed)|"
    r"unauthori[sz]ed|did(?:n't|\s+not)\s+make)\b|"
    r"\b(?:unrecogni[sz]ed|unauthori[sz]ed|not\s+(?:mine|recogn[iz]ed))\b"
    r".{0,50}\b(?:direct\s+debit|transaction|payment|withdrawal|transfer|charge)\b",
    re.IGNORECASE,
)


def correct_card_delivery_intent(text: str) -> str | None:
    """Return an existing BANKING77 label when card-delivery wording is clear.

    A direct report that a card has not arrived maps to ``card_arrival``. A
    question about the usual delivery duration maps to
    ``card_delivery_estimate``. Ambiguous text, text without a card reference,
    and complaints that also explicitly describe an unrecognized transaction
    are left to the BERT prediction. This is a narrow wording rule, not a
    replacement classifier.
    """
    if not isinstance(text, str) or not text.strip() or not _CARD.search(text):
        return None
    if _DELIVERY_CONFLICT.search(text):
        return None

    if (
        _NOT_ARRIVED_AFTER_CARD.search(text)
        or _NOT_ARRIVED_BEFORE_CARD.search(text)
        or _DID_NOT_GET_CARD_DELIVERED.search(text)
        or _WAITING_FOR_CARD.search(text)
        or _DELAYED_CARD.search(text)
    ):
        return "card_arrival"
    if (
        _DELIVERY_QUESTION_CUE.search(text)
        and _DELIVERY_CONTEXT.search(text)
        and not _CARD_EXPIRY_CONTEXT.search(text)
    ):
        return "card_delivery_estimate"
    return None
