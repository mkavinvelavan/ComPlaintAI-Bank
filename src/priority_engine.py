"""Deterministic internal complaint priority and queue suggestions.

The intent defaults below are an explicit project configuration based on
CPB-POL-008 §§4.1–5.2 and its named internal queues in §6.1. These defaults
are triage aids, not official response commitments or determinations that an
allegation is true. CPB-POL-003 §10.3 identifies ongoing suspicious activity,
exposed credentials, recoverability, and inability to secure a service as
urgency considerations. The current processor input does not include enough
narrative facts to evaluate those conditions directly.

The policy does not set a numeric confidence threshold. By default, callers
must not assume one: a threshold may be supplied as a project configuration,
in which case lower-confidence results trigger general review and at least
MEDIUM priority. A high-risk prediction without a majority of high-risk labels
among its returned top candidates also falls back to general review. Unknown
labels, missing/invalid confidence, and credential exposure receive safe
review handling.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, TypedDict


@dataclass(frozen=True)
class IntentRule:
    """Configured default treatment for one classifier intent label."""

    priority: str
    reason_code: str
    department_suggestion: str


class PriorityAssessment(TypedDict):
    """Internal-only priority assessment returned by the engine."""

    priority: str
    reason_codes: list[str]
    reason: str
    department_suggestion: str
    manual_review_required: bool


# Policy-supported project queue names: account services, cards,
# fraud-related review, transfers and remittances, fees and interest,
# digital banking, and general complaint resolution (CPB-POL-008 §6.1).
# Every entry corresponds to an exact label in the trained model's mapping.
_INTENT_RULE_DATA: dict[str, tuple[str, str, str]] = {
    # Higher-priority candidates: possible unauthorized activity, compromise,
    # or loss of an access instrument. These remain allegations until reviewed.
    "card_payment_not_recognised": ("HIGH", "POSSIBLE_UNAUTHORIZED_ACTIVITY", "fraud-related review"),
    "cash_withdrawal_not_recognised": ("HIGH", "POSSIBLE_UNAUTHORIZED_ACTIVITY", "fraud-related review"),
    "direct_debit_payment_not_recognised": ("HIGH", "POSSIBLE_UNAUTHORIZED_ACTIVITY", "fraud-related review"),
    "compromised_card": ("HIGH", "POSSIBLE_ACCOUNT_COMPROMISE", "fraud-related review"),
    "lost_or_stolen_card": ("HIGH", "LOST_OR_STOLEN_INSTRUMENT", "cards"),
    "lost_or_stolen_phone": ("HIGH", "POSSIBLE_ACCOUNT_COMPROMISE", "digital banking"),

    # Medium defaults: the label describes a transaction, charge, access, or
    # service issue that may need investigation or follow-up (CPB-POL-008 §4.1).
    "balance_not_updated_after_bank_transfer": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "transfers and remittances"),
    "balance_not_updated_after_cheque_or_cash_deposit": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "account services"),
    "cancel_transfer": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "transfers and remittances"),
    "card_not_working": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "cards"),
    "card_payment_fee_charged": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "fees and interest"),
    "card_payment_wrong_exchange_rate": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "fees and interest"),
    "card_swallowed": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "cards"),
    "cash_withdrawal_charge": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "fees and interest"),
    "contactless_not_working": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "cards"),
    "declined_card_payment": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "cards"),
    "declined_cash_withdrawal": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "cards"),
    "declined_transfer": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "transfers and remittances"),
    "exchange_charge": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "fees and interest"),
    "extra_charge_on_statement": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "fees and interest"),
    "failed_transfer": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "transfers and remittances"),
    "pending_card_payment": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "cards"),
    "pending_cash_withdrawal": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "cards"),
    "pending_top_up": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "account services"),
    "pending_transfer": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "transfers and remittances"),
    "pin_blocked": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "digital banking"),
    "Refund_not_showing_up": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "transfers and remittances"),
    "request_refund": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "transfers and remittances"),
    "reverted_card_payment?": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "cards"),
    "top_up_failed": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "account services"),
    "top_up_reverted": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "account services"),
    "transaction_charged_twice": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "cards"),
    "transfer_fee_charged": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "fees and interest"),
    "transfer_not_received_by_recipient": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "transfers and remittances"),
    "unable_to_verify_identity": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "digital banking"),
    "wrong_amount_of_cash_received": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "cards"),
    "wrong_exchange_rate_for_cash_withdrawal": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "fees and interest"),

    # Remaining supported labels are explicit LOW defaults for routine
    # information or ordinary service requests absent an added risk indicator.
    "activate_my_card": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "age_limit": ("LOW", "ROUTINE_SERVICE_QUERY", "account services"),
    "apple_pay_or_google_pay": ("LOW", "ROUTINE_SERVICE_QUERY", "digital banking"),
    "atm_support": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "automatic_top_up": ("LOW", "ROUTINE_SERVICE_QUERY", "account services"),
    "beneficiary_not_allowed": ("LOW", "ROUTINE_SERVICE_QUERY", "transfers and remittances"),
    "card_about_to_expire": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "card_acceptance": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "card_arrival": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "card_delivery_estimate": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "card_linking": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "country_support": ("LOW", "ROUTINE_SERVICE_QUERY", "account services"),
    "change_pin": ("LOW", "ROUTINE_SERVICE_QUERY", "digital banking"),
    "disposable_card_limits": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "edit_personal_details": ("LOW", "ROUTINE_SERVICE_QUERY", "account services"),
    "exchange_rate": ("LOW", "ROUTINE_SERVICE_QUERY", "fees and interest"),
    "exchange_via_app": ("LOW", "ROUTINE_SERVICE_QUERY", "digital banking"),
    "fiat_currency_support": ("LOW", "ROUTINE_SERVICE_QUERY", "account services"),
    "get_disposable_virtual_card": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "get_physical_card": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "getting_spare_card": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "getting_virtual_card": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "order_physical_card": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "passcode_forgotten": ("LOW", "ROUTINE_SERVICE_QUERY", "digital banking"),
    "receiving_money": ("LOW", "ROUTINE_SERVICE_QUERY", "transfers and remittances"),
    "supported_cards_and_currencies": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "terminate_account": ("LOW", "ROUTINE_SERVICE_QUERY", "account services"),
    "top_up_by_bank_transfer_charge": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "fees and interest"),
    "top_up_by_card_charge": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "fees and interest"),
    "top_up_by_cash_or_cheque": ("LOW", "ROUTINE_SERVICE_QUERY", "account services"),
    "top_up_limits": ("LOW", "ROUTINE_SERVICE_QUERY", "account services"),
    "topping_up_by_card": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "transfer_into_account": ("LOW", "ROUTINE_SERVICE_QUERY", "transfers and remittances"),
    "transfer_timing": ("LOW", "ROUTINE_SERVICE_QUERY", "transfers and remittances"),
    "verify_my_identity": ("LOW", "ROUTINE_SERVICE_QUERY", "digital banking"),
    "verify_source_of_funds": ("LOW", "ROUTINE_SERVICE_QUERY", "account services"),
    "verify_top_up": ("LOW", "ROUTINE_SERVICE_QUERY", "account services"),
    "virtual_card_not_working": ("MEDIUM", "INVESTIGATION_OR_FOLLOW_UP", "cards"),
    "visa_or_mastercard": ("LOW", "ROUTINE_SERVICE_QUERY", "cards"),
    "why_verify_identity": ("LOW", "ROUTINE_SERVICE_QUERY", "digital banking"),
}


INTENT_RULES: dict[str, IntentRule] = {
    intent: IntentRule(priority, reason_code, queue)
    for intent, (priority, reason_code, queue) in _INTENT_RULE_DATA.items()
}

_REASON_TEXT = {
    "POSSIBLE_UNAUTHORIZED_ACTIVITY": (
        "The predicted intent is a possible unrecognized transaction; this is a "
        "triage signal, not confirmation of unauthorized activity."
    ),
    "POSSIBLE_ACCOUNT_COMPROMISE": (
        "The predicted intent may indicate compromised access or an access device; "
        "the report requires authorized review and is not a confirmed finding."
    ),
    "LOST_OR_STOLEN_INSTRUMENT": (
        "The predicted intent may indicate a lost or stolen payment instrument; "
        "the report requires prompt authorized review."
    ),
    "INVESTIGATION_OR_FOLLOW_UP": (
        "The predicted intent describes a transaction, charge, access, or service "
        "issue that may require investigation or follow-up."
    ),
    "ROUTINE_SERVICE_QUERY": (
        "The predicted intent is configured as a routine information or service "
        "request; no additional urgent indicator was supplied to this assessment."
    ),
}

_PRIORITY_RANK = {"LOW": 0, "MEDIUM": 1, "HIGH": 2}


def _high_risk_prediction_is_ambiguous(processed: Mapping[str, Any]) -> bool:
    """Flag a high-risk winner without majority agreement in its top candidates.

    This categorical agreement check uses the candidates already returned by
    BERT and introduces no new confidence threshold. Unknown or missing
    candidates cannot establish agreement, so they fail closed to review.
    """
    candidates = processed.get("top_predictions")
    if not isinstance(candidates, list) or not candidates:
        return True

    high_risk_votes = 0
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            return True
        intent = candidate.get("intent")
        rule = INTENT_RULES.get(intent) if isinstance(intent, str) else None
        if rule is None:
            return True
        if rule.priority == "HIGH":
            high_risk_votes += 1
    return high_risk_votes * 2 <= len(candidates)


class ComplaintPriorityEngine:
    """Assign a rule-based internal priority and project queue suggestion.

    ``low_confidence_threshold`` is intentionally unset by default because
    CPB-POL-008 does not specify a numeric cutoff. Supplying one is an explicit
    project configuration, not an official banking threshold.
    """

    def __init__(self, low_confidence_threshold: float | None = None) -> None:
        if low_confidence_threshold is not None:
            if (
                isinstance(low_confidence_threshold, bool)
                or not isinstance(low_confidence_threshold, (int, float))
                or not math.isfinite(low_confidence_threshold)
                or not 0.0 <= low_confidence_threshold <= 1.0
            ):
                raise ValueError("low_confidence_threshold must be between 0.0 and 1.0 or None.")
        self.low_confidence_threshold = low_confidence_threshold

    @staticmethod
    def _minimum_priority(current: str, minimum: str) -> str:
        return current if _PRIORITY_RANK[current] >= _PRIORITY_RANK[minimum] else minimum

    def assess(self, processed_complaint: dict[str, Any]) -> PriorityAssessment:
        """Return an internal assessment from a ComplaintProcessor result.

        The input is read only and the complaint text is never copied to output,
        logged, or persisted. Unknown or absent intent labels use a medium
        general-review fallback rather than a guessed label-specific rule.
        """
        if not isinstance(processed_complaint, dict):
            raise TypeError("processed_complaint must be a dictionary.")

        intent = processed_complaint.get("predicted_intent")
        known_rule = isinstance(intent, str) and intent in INTENT_RULES
        ambiguous_high_risk = bool(
            known_rule
            and INTENT_RULES[intent].priority == "HIGH"
            and _high_risk_prediction_is_ambiguous(processed_complaint)
        )
        if known_rule:
            rule = INTENT_RULES[intent]
            priority = rule.priority
            queue = rule.department_suggestion
            reason_codes = [rule.reason_code]
            reason_parts = [_REASON_TEXT[rule.reason_code]]
            manual_review = rule.priority in {"HIGH", "MEDIUM"}
        else:
            priority = "MEDIUM"
            queue = "general complaint resolution"
            reason_codes = ["UNKNOWN_OR_MISSING_INTENT"]
            reason_parts = [
                "The intent label is missing or unsupported; use the general "
                "complaint-resolution queue for authorized review."
            ]
            manual_review = True

        if ambiguous_high_risk:
            priority = "MEDIUM"
            queue = "general complaint resolution"
            reason_codes = ["AMBIGUOUS_HIGH_RISK_INTENT_REVIEW"]
            reason_parts = [
                "The top classifier candidates do not agree on a high-risk category; "
                "use general complaint review without treating the predicted label as fact."
            ]
            manual_review = True

        confidence = processed_complaint.get("confidence")
        valid_confidence = (
            isinstance(confidence, (int, float))
            and not isinstance(confidence, bool)
            and math.isfinite(confidence)
            and 0.0 <= confidence <= 1.0
        )
        if not valid_confidence:
            if known_rule and rule.priority == "HIGH":
                priority = "MEDIUM"
            else:
                priority = self._minimum_priority(priority, "MEDIUM")
            if known_rule:
                queue = "general complaint resolution"
            reason_codes.append("MISSING_OR_INVALID_CONFIDENCE_REVIEW")
            reason_parts.append(
                "A usable classifier confidence was not supplied, so the result "
                "requires review and is not treated as a confirmed category."
            )
            manual_review = True
        elif (
            self.low_confidence_threshold is not None
            and confidence < self.low_confidence_threshold
        ):
            if known_rule and rule.priority == "HIGH":
                priority = "MEDIUM"
            else:
                priority = self._minimum_priority(priority, "MEDIUM")
            if known_rule:
                queue = "general complaint resolution"
            reason_codes.append("LOW_CONFIDENCE_REVIEW")
            reason_parts.append(
                "Classifier confidence is below the explicitly configured project "
                "threshold; use review rather than treating the prediction as fact."
            )
            manual_review = True

        disclosure_flag = processed_complaint.get("sensitive_disclosure_detected", False)
        sensitive_categories = processed_complaint.get("sensitive_categories", [])
        if not isinstance(disclosure_flag, bool):
            raise ValueError("sensitive_disclosure_detected must be a Boolean when provided.")
        if not isinstance(sensitive_categories, list) or any(
            not isinstance(category, str) for category in sensitive_categories
        ):
            raise ValueError("sensitive_categories must be a list of category names.")

        if disclosure_flag or sensitive_categories:
            priority = "HIGH"
            if not known_rule or rule.priority != "HIGH":
                queue = "digital banking"
            reason_codes.append("POSSIBLE_CREDENTIAL_EXPOSURE_REVIEW")
            reason_parts.append(
                "The extractor flagged possible credential exposure; handle it as a "
                "security indicator without including or repeating any credential value."
            )
            manual_review = True

        return {
            "priority": priority,
            "reason_codes": reason_codes,
            "reason": " ".join(reason_parts),
            "department_suggestion": queue,
            "manual_review_required": manual_review,
        }
