"""Tests for composing the local complaint processor and priority engine."""

from __future__ import annotations

import json
import unittest
from unittest.mock import Mock, patch

from src.complaint_workflow import ComplaintWorkflow
from src.priority_engine import ComplaintPriorityEngine


def processed_result(
    intent: str | None = "age_limit", confidence: float | None = 0.94, **changes: object
) -> dict[str, object]:
    result: dict[str, object] = {
        "predicted_intent": intent,
        "confidence": confidence,
        "top_predictions": [{"intent": intent, "confidence": confidence}],
        "entities": [],
        "sensitive_disclosure_detected": False,
        "sensitive_categories": [],
    }
    result.update(changes)
    return result


class ComplaintWorkflowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.processor = Mock()
        self.engine = Mock()
        self.processor.process_complaint.return_value = processed_result(
            "age_limit", 0.94,
            entities=[
                {"text": "AED 125.50", "entity_type": "MONEY_AMOUNT", "start": 12, "end": 22},
                {"text": "TXN-AB1234", "entity_type": "TRANSACTION_REFERENCE", "start": 50, "end": 60},
            ],
        )
        self.engine.assess.return_value = {
            "priority": "LOW",
            "reason_codes": ["ROUTINE_SERVICE_QUERY"],
            "reason": "Project-configured routine-query rule.",
            "department_suggestion": "account services",
            "manual_review_required": False,
        }
        self.workflow = ComplaintWorkflow(self.processor, self.engine)

    def test_calls_existing_components_and_combines_structured_results(self) -> None:
        text = "Synthetic request with AED 125.50 and transaction reference TXN-AB1234."
        result = self.workflow.assess_complaint(text)

        self.processor.process_complaint.assert_called_once_with(text)
        self.engine.assess.assert_called_once_with(self.processor.process_complaint.return_value)
        self.assertEqual(result["classification"]["predicted_intent"], "age_limit")
        self.assertEqual(result["entities"], self.processor.process_complaint.return_value["entities"])
        self.assertEqual(result["priority_assessment"], self.engine.assess.return_value)
        self.assertTrue(result["classification"]["is_model_prediction_not_verified_finding"])

    def test_routine_complaint_uses_project_rule(self) -> None:
        engine = ComplaintPriorityEngine()
        processor = Mock()
        processor.process_complaint.return_value = processed_result("age_limit", 0.94)
        result = ComplaintWorkflow(processor, engine).assess_complaint("What is the age limit?")
        self.assertEqual(result["priority_assessment"]["priority"], "LOW")

    def test_suspected_unauthorized_transaction_is_not_treated_as_verified(self) -> None:
        processor = Mock()
        processor.process_complaint.return_value = processed_result(
            "card_payment_not_recognised", 0.81
        )
        result = ComplaintWorkflow(processor).assess_complaint(
            "I do not recognize a synthetic card transaction."
        )
        self.assertEqual(result["priority_assessment"]["priority"], "HIGH")
        self.assertIn("POSSIBLE_UNAUTHORIZED_ACTIVITY", result["priority_assessment"]["reason_codes"])
        self.assertTrue(result["priority_assessment"]["manual_review_required"])
        self.assertTrue(result["classification"]["is_model_prediction_not_verified_finding"])

    def test_rejects_empty_and_invalid_input_before_components(self) -> None:
        for value in ("", " \n"):
            with self.subTest(value=repr(value)), self.assertRaises(ValueError):
                self.workflow.assess_complaint(value)
        with self.assertRaises(TypeError):
            self.workflow.assess_complaint(None)  # type: ignore[arg-type]
        self.processor.process_complaint.assert_not_called()

    def test_missing_or_uncertain_classification_uses_engine_fallback(self) -> None:
        processor = Mock()
        processor.process_complaint.return_value = processed_result(None, None)
        result = ComplaintWorkflow(processor).assess_complaint("A synthetic unclear request.")
        self.assertIsNone(result["classification"]["predicted_intent"])
        self.assertEqual(result["priority_assessment"]["priority"], "MEDIUM")
        self.assertIn("UNKNOWN_OR_MISSING_INTENT", result["priority_assessment"]["reason_codes"])
        self.assertIn("MISSING_OR_INVALID_CONFIDENCE_REVIEW", result["priority_assessment"]["reason_codes"])

    def test_unknown_intent_preserves_safe_engine_fallback(self) -> None:
        processor = Mock()
        processor.process_complaint.return_value = processed_result("unrecognized_test_label", 0.8)
        result = ComplaintWorkflow(processor).assess_complaint("Synthetic unknown category.")
        self.assertEqual(result["priority_assessment"]["priority"], "MEDIUM")
        self.assertIn("UNKNOWN_OR_MISSING_INTENT", result["priority_assessment"]["reason_codes"])

    def test_sensitive_disclosure_indicators_do_not_return_credential_values(self) -> None:
        processor = Mock()
        processor.process_complaint.return_value = processed_result(
            "age_limit", 0.8,
            sensitive_disclosure_detected=True,
            sensitive_categories=["OTP", "PIN"],
        )
        text = "Synthetic disclosure: OTP 739184 and PIN 4821."
        result = ComplaintWorkflow(processor).assess_complaint(text)
        serialized = json.dumps(result)
        self.assertTrue(result["security_disclosure"]["detected"])
        self.assertEqual(result["security_disclosure"]["categories"], ["OTP", "PIN"])
        self.assertTrue(
            all(value not in serialized for value in ("739184", "4821")),
            "credential values must not appear in workflow output",
        )
        self.assertTrue(text not in serialized, "source complaint text must not be returned")

    def test_result_is_json_serializable_and_excludes_source_text(self) -> None:
        text = "Synthetic complaint REF-PRIVATE-TEXT"
        result = self.workflow.assess_complaint(text)
        output = json.dumps(result)
        self.assertIsInstance(output, str)
        self.assertNotIn(text, output)
        self.assertEqual(
            set(result),
            {"classification", "entities", "security_disclosure", "priority_assessment"},
        )

    def test_optional_threshold_propagates_via_engine_constructor(self) -> None:
        with patch("src.complaint_workflow.ComplaintPriorityEngine") as engine_class:
            configured_engine = engine_class.return_value
            configured_engine.assess.return_value = {"priority": "MEDIUM"}
            workflow = ComplaintWorkflow(self.processor, self.engine)
            workflow.assess_complaint("Synthetic complaint.", confidence_threshold=0.7)

        engine_class.assert_called_once_with(low_confidence_threshold=0.7)
        configured_engine.assess.assert_called_once_with(self.processor.process_complaint.return_value)
        self.engine.assess.assert_not_called()

    def test_threshold_is_unset_by_default(self) -> None:
        engine = ComplaintPriorityEngine()
        self.assertIsNone(engine.low_confidence_threshold)


if __name__ == "__main__":
    unittest.main()
