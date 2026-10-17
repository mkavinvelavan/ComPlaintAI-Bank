"""Integration tests for the local complaint processing pipeline."""

import json
import unittest
from unittest.mock import Mock

from src.bert_intent_classifier import BertIntentClassifier
from src.complaint_processor import ComplaintProcessor
from src.entity_extractor import SpacyEntityExtractor


class ComplaintProcessorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # Reuse the local trained model and initialized components across tests.
        cls.processor = ComplaintProcessor()

    def test_initializes_and_uses_both_components(self) -> None:
        self.assertIsInstance(self.processor.classifier, BertIntentClassifier)
        self.assertIsInstance(self.processor.entity_extractor, SpacyEntityExtractor)

        result = self.processor.process_complaint(
            "My account ending in 1234 has a transaction reference TXN-AB1234."
        )
        self.assertTrue(result["predicted_intent"])
        self.assertTrue(result["top_predictions"])
        self.assertTrue(result["entities"])

    def test_combines_prediction_and_supported_entities(self) -> None:
        text = (
            "I was charged AED 125.50 on 17/10/2026 for transaction reference "
            "TXN-AB1234. My card ending in 4821 was used."
        )
        result = self.processor.process_complaint(text)

        self.assertEqual(
            set(result),
            {
                "predicted_intent",
                "confidence",
                "top_predictions",
                "entities",
                "sensitive_disclosure_detected",
                "sensitive_categories",
                "model_prediction",
                "intent_correction",
            },
        )
        self.assertTrue(result["predicted_intent"])
        self.assertIsInstance(result["confidence"], float)
        self.assertEqual(len(result["top_predictions"]), 3)

        found = {entity["entity_type"]: entity["text"] for entity in result["entities"]}
        self.assertEqual(found["TRANSACTION_REFERENCE"], "TXN-AB1234")
        self.assertEqual(found["MONEY_AMOUNT"], "AED 125.50")
        self.assertEqual(found["DATE"], "17/10/2026")
        self.assertEqual(found["CARD_LAST4"], "4821")
        for entity in result["entities"]:
            self.assertEqual(text[entity["start"] : entity["end"]], entity["text"])

    def test_processes_multiple_complaints_with_reused_components(self) -> None:
        classifier = self.processor.classifier
        extractor = self.processor.entity_extractor
        first = self.processor.process_complaint("A card ending in 4821 was charged USD 40.00.")
        second = self.processor.process_complaint("My account ending in 1234 is unavailable.")

        self.assertTrue(first["predicted_intent"])
        self.assertTrue(second["predicted_intent"])
        self.assertIn("ACCOUNT_LAST4", [entity["entity_type"] for entity in second["entities"]])
        self.assertIs(self.processor.classifier, classifier)
        self.assertIs(self.processor.entity_extractor, extractor)

    def test_rejects_empty_and_invalid_input(self) -> None:
        for text in ("", "  \n"):
            with self.subTest(text=repr(text)):
                with self.assertRaises(ValueError):
                    self.processor.process_complaint(text)
        with self.assertRaises(TypeError):
            self.processor.process_complaint(None)  # type: ignore[arg-type]

    def test_credential_values_are_not_exposed(self) -> None:
        text = "I accidentally shared my OTP 739184 and PIN: 4821."
        result = self.processor.process_complaint(text)
        serialized = json.dumps(result)

        self.assertTrue(result["sensitive_disclosure_detected"])
        self.assertIn("OTP", result["sensitive_categories"])
        self.assertIn("PIN", result["sensitive_categories"])
        self.assertTrue(
            all(value not in serialized for value in ("739184", "4821")),
            "credential values must not appear in processing output",
        )
        self.assertTrue(text not in serialized, "source complaint text must not be returned")

    def test_clear_missing_card_delivery_is_corrected_and_raw_model_scores_retained(self) -> None:
        raw_prediction = {
            "predicted_intent": "direct_debit_payment_not_recognised",
            "confidence": 0.312023,
            "top_predictions": [
                {"intent": "direct_debit_payment_not_recognised", "confidence": 0.312023},
                {"intent": "card_arrival", "confidence": 0.093650},
                {"intent": "card_delivery_estimate", "confidence": 0.088288},
            ],
        }
        classifier = Mock()
        classifier.predict.return_value = raw_prediction
        extractor = Mock()
        extractor.extract.return_value = {
            "entities": [],
            "security_flags": {"credential_disclosed": False, "credential_categories": []},
        }
        processor = ComplaintProcessor(classifier, extractor)

        result = processor.process_complaint("My debit card hasn't arrived after 20 days")

        self.assertEqual(result["predicted_intent"], "card_arrival")
        self.assertIsNone(result["confidence"])
        self.assertEqual(result["model_prediction"], raw_prediction)
        self.assertEqual(result["top_predictions"], raw_prediction["top_predictions"])
        self.assertEqual(
            result["intent_correction"], {"applied": True, "rule": "card_arrival"}
        )

    def test_local_model_card_arrival_example_uses_existing_intent(self) -> None:
        result = self.processor.process_complaint(
            "My debit card hasn't arrived after 20 days"
        )
        self.assertEqual(result["predicted_intent"], "card_arrival")
        self.assertEqual(
            result["model_prediction"]["predicted_intent"],
            "direct_debit_payment_not_recognised",
        )

    def test_local_model_unrecognized_direct_debit_keeps_existing_intent(self) -> None:
        result = self.processor.process_complaint(
            "I do not recognize a direct debit transaction."
        )
        self.assertEqual(
            result["predicted_intent"], "direct_debit_payment_not_recognised"
        )
        self.assertFalse(result["intent_correction"]["applied"])

    def test_delivery_duration_question_uses_estimate_intent(self) -> None:
        result = self.processor.process_complaint(
            "How long does card delivery usually take?"
        )
        self.assertEqual(result["predicted_intent"], "card_delivery_estimate")

    def test_card_expiry_question_is_not_rewritten_as_delivery_estimate(self) -> None:
        result = self.processor.process_complaint("How long will my debit card last?")
        self.assertFalse(result["intent_correction"]["applied"])
        self.assertEqual(
            result["predicted_intent"], result["model_prediction"]["predicted_intent"]
        )

    def test_unrecognized_direct_debit_is_not_overridden_as_card_delivery(self) -> None:
        raw_prediction = {
            "predicted_intent": "direct_debit_payment_not_recognised",
            "confidence": 0.91,
            "top_predictions": [
                {"intent": "direct_debit_payment_not_recognised", "confidence": 0.91}
            ],
        }
        classifier = Mock()
        classifier.predict.return_value = raw_prediction
        extractor = Mock()
        extractor.extract.return_value = {
            "entities": [],
            "security_flags": {"credential_disclosed": False, "credential_categories": []},
        }
        result = ComplaintProcessor(classifier, extractor).process_complaint(
            "I do not recognize an unrecognized direct debit transaction."
        )
        self.assertEqual(result["predicted_intent"], "direct_debit_payment_not_recognised")
        self.assertFalse(result["intent_correction"]["applied"])

    def test_result_is_json_serializable_and_contains_no_source_text(self) -> None:
        text = "Please check transaction reference TXN-EX1234."
        result = self.processor.process_complaint(text)
        serialized = json.dumps(result)

        self.assertIsInstance(serialized, str)
        self.assertNotIn(text, serialized)


if __name__ == "__main__":
    unittest.main()
