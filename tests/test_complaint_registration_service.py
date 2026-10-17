"""Offline tests for the analysis-to-persistence registration boundary."""

from __future__ import annotations

import unittest
from unittest.mock import Mock

from src.complaint_registration_service import (
    ComplaintRegistrationError,
    ComplaintRegistrationService,
)


class ComplaintRegistrationServiceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.customer_id = "trusted-server-customer-27"
        self.complaint_text = "Synthetic complaint about a pending transfer."
        self.assessment = {
            "classification": {
                "predicted_intent": "balance_not_updated_after_bank_transfer",
                "confidence": 0.87,
                "top_predictions": [],
                "is_model_prediction_not_verified_finding": True,
            },
            "entities": [
                {
                    "text": "TXN-SYNTHETIC-42",
                    "entity_type": "TRANSACTION_REFERENCE",
                    "start": 0,
                    "end": 16,
                }
            ],
            "security_disclosure": {"detected": False, "categories": []},
            "priority_assessment": {
                "priority": "MEDIUM",
                "department_suggestion": "transfers and remittances",
                "reason_code": "SYNTHETIC_RULE",
            },
        }
        self.workflow = Mock()
        self.workflow.assess_complaint.return_value = self.assessment
        self.repository = Mock()
        self.repository.register_complaint.return_value = "CPB-20261017-ABCDEF0123"
        self.service = ComplaintRegistrationService(self.workflow, self.repository)

    def test_success_analyzes_then_persists_actual_workflow_result(self) -> None:
        result = self.service.register_complaint(self.customer_id, self.complaint_text)

        self.workflow.assess_complaint.assert_called_once_with(self.complaint_text)
        self.repository.register_complaint.assert_called_once_with(
            self.customer_id,
            self.complaint_text,
            self.assessment,
            contact_email=None,
            contact_phone=None,
        )
        self.assertEqual(
            result,
            {"registered": True, "case_reference": "CPB-20261017-ABCDEF0123"},
        )
        self.assertNotIn("priority", result)
        self.assertNotIn("department_suggestion", result)
        self.assertNotIn("predicted_intent", result)

    def test_result_contains_reference_only_after_repository_returns_success(self) -> None:
        expected_reference = "CPB-20261017-ABCDEF0123"
        self.repository.register_complaint.return_value = expected_reference

        result = self.service.register_complaint(self.customer_id, self.complaint_text)

        self.assertEqual(result["case_reference"], expected_reference)
        self.assertTrue(result["registered"])

    def test_analysis_failure_prevents_persistence_and_hides_exception_details(self) -> None:
        synthetic_credential = "SyntheticOtp739184"
        self.workflow.assess_complaint.side_effect = RuntimeError(
            f"analysis failed for {synthetic_credential}: {self.complaint_text}"
        )

        with self.assertRaises(ComplaintRegistrationError) as caught:
            self.service.register_complaint(self.customer_id, self.complaint_text)

        self.repository.register_complaint.assert_not_called()
        message = str(caught.exception)
        self.assertTrue("registration was not attempted" in message)
        self.assertTrue(synthetic_credential not in message)
        self.assertTrue(self.complaint_text not in message)

    def test_persistence_failure_returns_no_success_or_case_reference(self) -> None:
        synthetic_reference = "CPB-20261017-ABCDEF0123"
        synthetic_db_detail = "synthetic private database detail"
        self.repository.register_complaint.side_effect = RuntimeError(
            f"{synthetic_db_detail}; candidate {synthetic_reference}"
        )

        with self.assertRaises(ComplaintRegistrationError) as caught:
            self.service.register_complaint(self.customer_id, self.complaint_text)

        message = str(caught.exception)
        self.assertTrue("could not be confirmed" in message)
        self.assertTrue(synthetic_reference not in message)
        self.assertTrue(synthetic_db_detail not in message)
        self.assertTrue("registered" not in message)

    def test_invalid_inputs_are_rejected_before_analysis_or_persistence(self) -> None:
        invalid_pairs = (
            ("", self.complaint_text, ValueError),
            (" \t", self.complaint_text, ValueError),
            (self.customer_id, "\n ", ValueError),
            (self.customer_id, "x" * 50_001, ValueError),
            (123, self.complaint_text, TypeError),
            (self.customer_id, None, TypeError),
        )
        for customer_id, complaint_text, error_type in invalid_pairs:
            with self.subTest(error_type=error_type.__name__):
                with self.assertRaises(error_type):
                    self.service.register_complaint(customer_id, complaint_text)  # type: ignore[arg-type]

        self.workflow.assess_complaint.assert_not_called()
        self.repository.register_complaint.assert_not_called()

    def test_customer_identifier_is_passed_through_not_authenticated_here(self) -> None:
        result = self.service.register_complaint(self.customer_id, self.complaint_text)

        self.repository.register_complaint.assert_called_once()
        passed_customer_id = self.repository.register_complaint.call_args.args[0]
        self.assertEqual(passed_customer_id, self.customer_id)
        self.assertTrue(result["registered"])
        # The service only accepts a caller-provided identifier; authentication
        # and authorization remain the responsibility of the future server.

    def test_guest_registration_requires_and_maps_normalized_contact_details(self) -> None:
        result = self.service.register_complaint(
            None,
            self.complaint_text,
            contact_email=" Guest.Example@Example.test ",
            contact_phone="+971 (50) 123-4567",
        )

        self.assertTrue(result["registered"])
        self.repository.register_complaint.assert_called_once_with(
            None,
            self.complaint_text,
            self.assessment,
            contact_email="guest.example@example.test",
            contact_phone="+971501234567",
        )

    def test_guest_missing_or_invalid_contact_is_rejected_before_analysis(self) -> None:
        for email, phone in (
            (None, None),
            ("invalid-email", None),
            (None, "invalid-phone"),
        ):
            with self.subTest(email_present=email is not None, phone_present=phone is not None):
                with self.assertRaises(ValueError):
                    self.service.register_complaint(
                        None,
                        self.complaint_text,
                        contact_email=email,
                        contact_phone=phone,
                    )
        self.workflow.assess_complaint.assert_not_called()
        self.repository.register_complaint.assert_not_called()

    def test_service_defers_repository_construction_until_after_analysis(self) -> None:
        repository_factory = Mock(return_value=self.repository)
        service = ComplaintRegistrationService(
            workflow=self.workflow,
            repository_factory=repository_factory,
        )

        self.assertIsNone(service.repository)
        repository_factory.assert_not_called()
        service.register_complaint(self.customer_id, self.complaint_text)

        repository_factory.assert_called_once_with()
        self.workflow.assess_complaint.assert_called_once_with(self.complaint_text)

    def test_non_mapping_analysis_result_does_not_persist(self) -> None:
        self.workflow.assess_complaint.return_value = ["invalid", "result"]

        with self.assertRaises(ComplaintRegistrationError) as caught:
            self.service.register_complaint(self.customer_id, self.complaint_text)

        self.assertTrue("registration was not attempted" in str(caught.exception))
        self.repository.register_complaint.assert_not_called()

    def test_missing_or_invalid_repository_reference_is_not_success(self) -> None:
        for invalid_reference in (None, "", 123):
            with self.subTest(reference_type=type(invalid_reference).__name__):
                self.repository.register_complaint.return_value = invalid_reference
                with self.assertRaises(ComplaintRegistrationError) as caught:
                    self.service.register_complaint(self.customer_id, self.complaint_text)
                self.assertTrue("could not be confirmed" in str(caught.exception))
                self.assertTrue("registered" not in str(caught.exception))


if __name__ == "__main__":
    unittest.main()
