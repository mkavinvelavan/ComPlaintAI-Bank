"""Unit tests for rule-based internal complaint priority assessment."""

import json
import unittest
from copy import deepcopy
from pathlib import Path

from src.priority_engine import ComplaintPriorityEngine, INTENT_RULES


class ComplaintPriorityEngineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = ComplaintPriorityEngine()

    @staticmethod
    def processed(
        intent: str = "card_payment_not_recognised",
        confidence: float = 0.91,
        **overrides: object,
    ) -> dict[str, object]:
        result: dict[str, object] = {
            "predicted_intent": intent,
            "confidence": confidence,
            "top_predictions": [{"intent": intent, "confidence": confidence}],
            "entities": [],
            "sensitive_disclosure_detected": False,
            "sensitive_categories": [],
        }
        result.update(overrides)
        return result

    def test_suspected_unrecognized_transaction_is_high_not_confirmed(self) -> None:
        assessment = self.engine.assess(self.processed())
        self.assertEqual(assessment["priority"], "HIGH")
        self.assertIn("POSSIBLE_UNAUTHORIZED_ACTIVITY", assessment["reason_codes"])
        self.assertTrue(assessment["manual_review_required"])
        self.assertIn("not confirmation", assessment["reason"])
        self.assertEqual(assessment["department_suggestion"], "fraud-related review")

    def test_routine_account_question_is_low(self) -> None:
        assessment = self.engine.assess(
            self.processed(intent="age_limit", confidence=0.98)
        )
        self.assertEqual(assessment["priority"], "LOW")
        self.assertEqual(assessment["reason_codes"], ["ROUTINE_SERVICE_QUERY"])
        self.assertFalse(assessment["manual_review_required"])
        self.assertEqual(assessment["department_suggestion"], "account services")

    def test_medium_transaction_follow_up_is_internal_review(self) -> None:
        assessment = self.engine.assess(
            self.processed(intent="pending_transfer", confidence=0.88)
        )
        self.assertEqual(assessment["priority"], "MEDIUM")
        self.assertEqual(assessment["reason_codes"], ["INVESTIGATION_OR_FOLLOW_UP"])
        self.assertTrue(assessment["manual_review_required"])
        self.assertEqual(
            assessment["department_suggestion"], "transfers and remittances"
        )

    def test_explicit_configured_low_confidence_triggers_review(self) -> None:
        engine = ComplaintPriorityEngine(low_confidence_threshold=0.70)
        assessment = engine.assess(
            self.processed(intent="activate_my_card", confidence=0.42)
        )
        self.assertEqual(assessment["priority"], "MEDIUM")
        self.assertIn("LOW_CONFIDENCE_REVIEW", assessment["reason_codes"])
        self.assertTrue(assessment["manual_review_required"])
        self.assertEqual(
            assessment["department_suggestion"], "general complaint resolution"
        )
        self.assertIn("explicitly configured project threshold", assessment["reason"])

    def test_credentials_raise_high_review_without_returning_values(self) -> None:
        processed = self.processed(
            intent="age_limit",
            sensitive_disclosure_detected=True,
            sensitive_categories=["OTP", "PIN"],
            entities=[],
        )
        assessment = self.engine.assess(processed)
        serialized = json.dumps(assessment)

        self.assertEqual(assessment["priority"], "HIGH")
        self.assertIn("POSSIBLE_CREDENTIAL_EXPOSURE_REVIEW", assessment["reason_codes"])
        self.assertTrue(assessment["manual_review_required"])
        self.assertEqual(assessment["department_suggestion"], "digital banking")
        self.assertNotIn("739184", serialized)
        self.assertNotIn("4821", serialized)

    def test_unknown_and_missing_intents_use_safe_review_fallback(self) -> None:
        for processed in (
            self.processed(intent="made_up_intent"),
            {"confidence": 0.95},
        ):
            with self.subTest(processed=processed):
                assessment = self.engine.assess(processed)
                self.assertEqual(assessment["priority"], "MEDIUM")
                self.assertIn("UNKNOWN_OR_MISSING_INTENT", assessment["reason_codes"])
                self.assertTrue(assessment["manual_review_required"])
                self.assertEqual(
                    assessment["department_suggestion"], "general complaint resolution"
                )

    def test_missing_or_invalid_confidence_requires_review(self) -> None:
        assessment = self.engine.assess({"predicted_intent": "age_limit"})
        self.assertEqual(assessment["priority"], "MEDIUM")
        self.assertIn(
            "MISSING_OR_INVALID_CONFIDENCE_REVIEW", assessment["reason_codes"]
        )
        self.assertTrue(assessment["manual_review_required"])

    def test_conflicting_top_candidates_route_high_prediction_to_general_review(self) -> None:
        processed = self.processed(
            intent="direct_debit_payment_not_recognised",
            confidence=0.312023,
            top_predictions=[
                {"intent": "direct_debit_payment_not_recognised", "confidence": 0.312023},
                {"intent": "card_arrival", "confidence": 0.093650},
                {"intent": "card_delivery_estimate", "confidence": 0.088288},
            ],
        )

        assessment = self.engine.assess(processed)

        self.assertEqual(assessment["priority"], "MEDIUM")
        self.assertEqual(
            assessment["department_suggestion"], "general complaint resolution"
        )
        self.assertIn(
            "AMBIGUOUS_HIGH_RISK_INTENT_REVIEW", assessment["reason_codes"]
        )
        self.assertTrue(assessment["manual_review_required"])

    def test_configured_low_confidence_high_prediction_uses_general_review(self) -> None:
        engine = ComplaintPriorityEngine(low_confidence_threshold=0.5)
        assessment = engine.assess(
            self.processed(
                intent="direct_debit_payment_not_recognised",
                confidence=0.312023,
                top_predictions=[
                    {"intent": "direct_debit_payment_not_recognised", "confidence": 0.312023}
                ],
            )
        )
        self.assertEqual(assessment["priority"], "MEDIUM")
        self.assertEqual(
            assessment["department_suggestion"], "general complaint resolution"
        )
        self.assertIn("LOW_CONFIDENCE_REVIEW", assessment["reason_codes"])

    def test_reason_code_and_explanation_match_the_fired_rule(self) -> None:
        assessment = self.engine.assess(
            self.processed(intent="lost_or_stolen_card", confidence=0.93)
        )
        self.assertEqual(assessment["reason_codes"], ["LOST_OR_STOLEN_INSTRUMENT"])
        self.assertIn("lost or stolen payment instrument", assessment["reason"])
        self.assertEqual(assessment["priority"], "HIGH")

    def test_validates_input_and_does_not_mutate_it(self) -> None:
        with self.assertRaises(TypeError):
            self.engine.assess("not a processed result")  # type: ignore[arg-type]

        processed = self.processed(
            intent="age_limit", entities=[{"text": "synthetic", "start": 0, "end": 9}]
        )
        before = deepcopy(processed)
        self.engine.assess(processed)
        self.assertEqual(processed, before)

        with self.assertRaises(ValueError):
            ComplaintPriorityEngine(low_confidence_threshold=1.5)

    def test_assessment_is_json_serializable_and_contains_no_complaint_text(self) -> None:
        assessment = self.engine.assess(self.processed())
        serialized = json.dumps(assessment)
        self.assertIsInstance(serialized, str)
        self.assertNotIn("customer complaint", serialized)

    def test_every_saved_model_intent_has_exactly_one_explicit_rule(self) -> None:
        mapping_path = (
            Path(__file__).resolve().parents[1]
            / "models"
            / "bert_banking77_model"
            / "label_mapping.json"
        )
        label_mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        expected_labels = set(label_mapping["id_to_label"].values())

        self.assertEqual(len(expected_labels), 77)
        self.assertEqual(set(INTENT_RULES), expected_labels)
        self.assertEqual(len(INTENT_RULES), 77)
        for intent, rule in INTENT_RULES.items():
            with self.subTest(intent=intent):
                self.assertIn(rule.priority, {"HIGH", "MEDIUM", "LOW"})
                self.assertTrue(rule.reason_code)
                self.assertTrue(rule.department_suggestion)


if __name__ == "__main__":
    unittest.main()
