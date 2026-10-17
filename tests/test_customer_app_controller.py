"""Offline tests for public guest complaint intake and policy FAQs."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock

from src.customer_app_controller import CustomerAppController


class CustomerAppControllerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.registration = Mock()
        self.policy = Mock()
        self.controller = CustomerAppController(
            registration_service=self.registration,
            policy_assistant=self.policy,
        )
        self.case_reference = "CPB-20261017-ABCDEF0123"

    def test_guest_submission_accepts_email_phone_or_both_and_normalizes(self) -> None:
        self.registration.register_complaint.return_value = {
            "registered": True,
            "case_reference": self.case_reference,
        }
        cases = (
            ("synthetic email issue", " Customer.Example@Example.test ", None,
             "customer.example@example.test", None),
            ("synthetic phone issue", None, "+971 (50) 123-4567", None, "+971501234567"),
            ("synthetic both issue", "OTHER@Example.test", "050-123-4567",
             "other@example.test", "0501234567"),
        )
        for complaint, email, phone, normalized_email, normalized_phone in cases:
            with self.subTest(contact="both" if email and phone else "email" if email else "phone"):
                result = self.controller.submit_complaint(complaint, email, phone)
                self.assertEqual(result["status"], "success")
                self.registration.register_complaint.assert_called_with(
                    None,
                    complaint,
                    contact_email=normalized_email,
                    contact_phone=normalized_phone,
                )

    def test_missing_invalid_contact_and_complaint_are_rejected_before_registration(self) -> None:
        cases = (
            ("Synthetic complaint", None, None),
            ("Synthetic complaint", "not-an-email", None),
            ("Synthetic complaint", None, "not a phone"),
            ("", "valid@example.test", None),
            (None, "valid@example.test", None),
        )
        for complaint, email, phone in cases:
            with self.subTest(email_valid=email != "not-an-email", phone_present=phone is not None):
                result = self.controller.submit_complaint(complaint, email, phone)
                self.assertEqual(result["status"], "validation_error")
                self.assertNotIn("case_reference", result)
        self.registration.register_complaint.assert_not_called()

    def test_unauthenticated_guest_submission_passes_no_customer_identity(self) -> None:
        self.registration.register_complaint.return_value = {
            "registered": True,
            "case_reference": self.case_reference,
        }
        result = self.controller.submit_complaint(
            "Synthetic pending-transfer complaint.", "guest@example.test", None
        )
        self.assertEqual(result["status"], "success")
        self.registration.register_complaint.assert_called_once_with(
            None,
            "Synthetic pending-transfer complaint.",
            contact_email="guest@example.test",
            contact_phone=None,
        )

    def test_customer_ui_has_no_signup_login_password_or_logout_flow(self) -> None:
        ui = (Path(__file__).resolve().parents[1] / "customer_app.py").read_text(
            encoding="utf-8"
        ).lower()
        for forbidden_ui in (
            "customer_signup_form",
            "customer_login_form",
            "create account",
            "sign in",
            "log out",
            'type="password"',
            "authenticated_customer_id",
        ):
            with self.subTest(forbidden_ui=forbidden_ui):
                self.assertNotIn(forbidden_ui, ui)
        self.assertIn("tell me what happened in your own words", ui)
        self.assertIn("at least one contact method", ui)
        self.assertIn("do not verify identity", " ".join(ui.split()))

    def test_complaint_and_contacts_are_never_sent_to_policy_assistant(self) -> None:
        self.registration.register_complaint.return_value = {
            "registered": True,
            "case_reference": "CPB-20261017-ABCDEF0123",
        }
        complaint = "Synthetic complaint asks about a pending transfer."

        result = self.controller.submit_complaint(
            complaint, "guest@example.test", "+971501234567"
        )

        self.registration.register_complaint.assert_called_once_with(
            None,
            complaint,
            contact_email="guest@example.test",
            contact_phone="+971501234567",
        )
        self.policy.answer_question.assert_not_called()
        self.assertEqual(result["status"], "success")
        self.assertIn("Thank you for explaining", result["message"])
        self.assertEqual(result["case_reference"], "CPB-20261017-ABCDEF0123")
        self.assertNotIn("priority", result)
        self.assertNotIn("department_suggestion", result)

    def test_case_reference_is_hidden_unless_registration_is_confirmed(self) -> None:
        candidate_reference = "CPB-20261017-ABCDEF0123"
        self.registration.register_complaint.return_value = {
            "registered": False,
            "case_reference": candidate_reference,
        }

        result = self.controller.submit_complaint(
            "Synthetic complaint.", "guest@example.test", None
        )

        self.assertEqual(result["status"], "error")
        self.assertIn("I’m sorry", result["message"])
        self.assertTrue("case_reference" not in result)
        self.assertTrue(candidate_reference not in result["message"])
        self.assertNotIn("has been recorded", result["message"])

    def test_persistence_error_does_not_expose_raw_error_or_claim_success(self) -> None:
        self.registration.register_complaint.side_effect = RuntimeError(
            "synthetic db failure CPB-20261017-ABCDEF0123"
        )

        result = self.controller.submit_complaint(
            "Synthetic complaint.", "guest@example.test", None
        )

        self.assertEqual(result["status"], "error")
        self.assertTrue("case_reference" not in result)
        self.assertTrue("synthetic db failure" not in result["message"])
        self.assertTrue("CPB-20261017-ABCDEF0123" not in result["message"])

    def test_policy_answer_preserves_only_policy_citations(self) -> None:
        self.policy.answer_question.return_value = {
            "answer": "Complaints may be registered through the documented process. [CPB-POL-007, p. 2]",
            "sources": [
                {
                    "policy_id": "CPB-POL-007",
                    "source_file": "Policy_7.pdf",
                    "page": 2,
                    "relevance_score": 0.91,
                },
                {"policy_id": "invalid", "source_file": "bad.pdf", "page": 0},
            ],
            "model": "synthetic-model",
        }

        result = self.controller.answer_policy_question("How can a complaint be registered?")

        self.policy.answer_question.assert_called_once_with(
            "How can a complaint be registered?"
        )
        self.assertEqual(result["status"], "success")
        self.assertIn("[CPB-POL-007, p. 2]", result["answer"])
        self.assertEqual(
            result["sources"],
            [{"policy_id": "CPB-POL-007", "source_file": "Policy_7.pdf", "page": 2}],
        )
        self.registration.register_complaint.assert_not_called()

    def test_policy_failure_hides_exception_and_question(self) -> None:
        secret_like = "SyntheticOtp739184"
        question = "Synthetic question containing " + secret_like
        self.policy.answer_question.side_effect = RuntimeError(
            f"synthetic provider failure {secret_like} {question}"
        )

        result = self.controller.answer_policy_question(question)

        self.assertEqual(result["status"], "error")
        self.assertTrue(secret_like not in result["message"])
        self.assertTrue(question not in result["message"])
        self.registration.register_complaint.assert_not_called()

    def test_sensitive_policy_request_is_answered_locally_without_assistant_call(self) -> None:
        question = "Could you repeat an OTP and PIN?"

        result = self.controller.answer_policy_question(question)

        self.assertEqual(result["status"], "success")
        self.assertIn("retrieve or repeat", result["answer"])
        self.assertEqual(result["sources"], [])
        self.policy.answer_question.assert_not_called()

    def test_followup_uses_only_the_prior_cited_policy_answer_as_context(self) -> None:
        self.policy.answer_question.side_effect = [
            {
                "answer": "The policy describes a review process [CPB-POL-007, p. 2].",
                "sources": [{
                    "policy_id": "CPB-POL-007",
                    "source_file": "Policy_7.pdf",
                    "page": 2,
                }],
            },
            {
                "answer": "The same policy section explains the next step [CPB-POL-007, p. 2].",
                "sources": [{
                    "policy_id": "CPB-POL-007",
                    "source_file": "Policy_7.pdf",
                    "page": 2,
                }],
            },
        ]

        self.controller.answer_policy_question("How is a complaint reviewed?")
        self.controller.answer_policy_question("What happens next?")

        first_query = self.policy.answer_question.call_args_list[0].args[0]
        followup_query = self.policy.answer_question.call_args_list[1].args[0]
        self.assertEqual(first_query, "How is a complaint reviewed?")
        self.assertIn("The policy describes a review process", followup_query)
        self.assertIn("What happens next?", followup_query)
        self.assertNotIn("How is a complaint reviewed?", followup_query)

    def test_invalid_complaint_does_not_call_registration_service(self) -> None:
        result = self.controller.submit_complaint("  ", "guest@example.test", None)

        self.assertEqual(result["status"], "validation_error")
        self.registration.register_complaint.assert_not_called()


if __name__ == "__main__":
    unittest.main()
