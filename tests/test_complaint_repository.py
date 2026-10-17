"""Offline tests for transactional complaint persistence."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from src.complaint_repository import (
    ComplaintPersistenceError,
    ComplaintRepository,
)


class FakeCursor:
    def __init__(
        self,
        events: list[str],
        execute_error: Exception | None = None,
        close_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.execute_error = execute_error
        self.close_error = close_error
        self.statement = ""
        self.parameters: tuple[object, ...] = ()

    def execute(self, statement: str, parameters: tuple[object, ...]) -> None:
        self.events.append("execute")
        self.statement = statement
        self.parameters = parameters
        if self.execute_error is not None:
            raise self.execute_error

    def fetchone(self) -> tuple[str]:
        return ("internal-id",)

    def close(self) -> None:
        self.events.append("cursor.close")
        if self.close_error is not None:
            raise self.close_error


class FakeConnection:
    def __init__(
        self,
        events: list[str],
        *,
        execute_error: Exception | None = None,
        commit_error: Exception | None = None,
        rollback_error: Exception | None = None,
        cursor_close_error: Exception | None = None,
        connection_close_error: Exception | None = None,
    ) -> None:
        self.events = events
        self.execute_error = execute_error
        self.commit_error = commit_error
        self.rollback_error = rollback_error
        self.connection_close_error = connection_close_error
        self.cursor_instance = FakeCursor(events, execute_error, cursor_close_error)

    def cursor(self) -> FakeCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.events.append("commit")
        if self.commit_error is not None:
            raise self.commit_error

    def rollback(self) -> None:
        self.events.append("rollback")
        if self.rollback_error is not None:
            raise self.rollback_error

    def close(self) -> None:
        self.events.append("connection.close")
        if self.connection_close_error is not None:
            raise self.connection_close_error


class SyntheticUniqueViolation(Exception):
    sqlstate = "23505"
    diag = SimpleNamespace(constraint_name="complaints_case_reference_key")


class ComplaintRepositoryTests(unittest.TestCase):
    def make_repository(
        self,
        connections: list[FakeConnection],
        references: list[str] | None = None,
    ) -> tuple[ComplaintRepository, Mock]:
        factory = Mock(side_effect=connections)
        refs = iter(references or ["CPB-20261017-ABCDEF0123"])
        repository = ComplaintRepository(
            "postgresql://offline.invalid/test",
            connection_factory=factory,
            case_reference_factory=lambda: next(refs),
        )
        return repository, factory

    def test_migration_defines_constraints_timestamps_and_history_indexes(self) -> None:
        migration = (
            Path(__file__).resolve().parents[1]
            / "migrations"
            / "001_create_complaints.sql"
        ).read_text(encoding="utf-8")
        normalized = " ".join(migration.lower().split())

        for expected in (
            "id uuid primary key",
            "case_reference text not null",
            "unique (case_reference)",
            "customer_id text not null",
            "complaint_text text not null",
            "priority in ('low', 'medium', 'high')",
            "credential_disclosure_detected boolean not null default false",
            "created_at timestamptz not null default current_timestamp",
            "updated_at timestamptz not null default current_timestamp",
            "on complaints (customer_id, created_at desc)",
            "on complaints (status, created_at desc)",
            "before update on complaints",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, normalized)
        self.assertNotIn("entities json", normalized)
        self.assertNotIn("credential_value", normalized)
        guest_migration = (
            Path(__file__).resolve().parents[1]
            / "migrations"
            / "003_guest_complaint_contacts.sql"
        ).read_text(encoding="utf-8").lower()
        guest_normalized = " ".join(guest_migration.split())
        self.assertIn("alter column customer_id drop not null", guest_normalized)
        self.assertIn("add column contact_email varchar(254)", guest_normalized)
        self.assertIn("add column contact_phone varchar(16)", guest_normalized)
        self.assertIn("contact_email is not null or contact_phone is not null", guest_normalized)

    def test_success_returns_reference_after_commit_and_uses_parameters(self) -> None:
        events: list[str] = []
        connection = FakeConnection(events)
        repository, factory = self.make_repository([connection])
        assessment = {
            "classification": {"predicted_intent": "failed_transfer", "confidence": 0.91},
            "entities": [{"text": "TXN-AB1234", "entity_type": "TRANSACTION_REFERENCE"}],
            "security_disclosure": {"detected": False, "categories": []},
            "priority_assessment": {
                "priority": "MEDIUM",
                "department_suggestion": "transfers and remittances",
            },
        }

        case_reference = repository.register_complaint(
            "synthetic-customer-17",
            "A synthetic transfer complaint.",
            assessment,
        )

        self.assertEqual(case_reference, "CPB-20261017-ABCDEF0123")
        self.assertLess(events.index("execute"), events.index("commit"))
        self.assertIn("commit", events)
        self.assertEqual(events.count("cursor.close"), 1)
        self.assertEqual(events.count("connection.close"), 1)
        self.assertEqual(factory.call_count, 1)
        cursor = connection.cursor_instance
        self.assertIn("INSERT INTO complaints", cursor.statement)
        self.assertNotIn("entities", cursor.statement)
        self.assertEqual(cursor.parameters[2:9], (
            "synthetic-customer-17",
            None,
            None,
            "A synthetic transfer complaint.",
            "failed_transfer",
            "MEDIUM",
            "transfers and remittances",
        ))
        self.assertFalse(cursor.parameters[-1])

    def test_validation_errors_do_not_open_a_connection(self) -> None:
        connection = FakeConnection([])
        repository, factory = self.make_repository([connection])
        cases = (
            ("", "A complaint."),
            ("synthetic-customer-17", " \n "),
            ("synthetic-customer-17", "A" * 50_001),
        )
        for customer_id, complaint_text in cases:
            with self.subTest(customer_id=customer_id[:4], complaint_length=len(complaint_text)):
                with self.assertRaises((TypeError, ValueError)):
                    repository.register_complaint(customer_id, complaint_text)
        with self.assertRaises(TypeError):
            repository.register_complaint("synthetic-customer-17", "A complaint.", [])  # type: ignore[arg-type]
        factory.assert_not_called()

    def test_insert_failure_rolls_back_and_does_not_disclose_reference(self) -> None:
        events: list[str] = []
        connection = FakeConnection(events, execute_error=RuntimeError("private database detail"))
        repository, _ = self.make_repository([connection])

        with self.assertRaises(ComplaintPersistenceError) as caught:
            repository.register_complaint("synthetic-customer-17", "A synthetic complaint.")

        self.assertIn("rollback", events)
        self.assertIn("cursor.close", events)
        self.assertIn("connection.close", events)
        self.assertNotIn("CPB-20261017-ABCDEF0123", str(caught.exception))
        self.assertNotIn("private database detail", str(caught.exception))

    def test_commit_failure_rolls_back_and_returns_no_reference(self) -> None:
        events: list[str] = []
        connection = FakeConnection(events, commit_error=RuntimeError("private commit detail"))
        repository, _ = self.make_repository([connection])

        with self.assertRaises(ComplaintPersistenceError) as caught:
            repository.register_complaint("synthetic-customer-17", "A synthetic complaint.")

        self.assertIn("commit", events)
        self.assertIn("rollback", events)
        self.assertNotIn("CPB-20261017-ABCDEF0123", str(caught.exception))
        self.assertNotIn("private commit detail", str(caught.exception))
        self.assertIn("no case reference was returned", str(caught.exception))
        self.assertIn("cursor.close", events)
        self.assertIn("connection.close", events)

    def test_rollback_failure_does_not_disclose_reference_or_raw_error(self) -> None:
        events: list[str] = []
        connection = FakeConnection(
            events,
            execute_error=RuntimeError("synthetic private insert detail"),
            rollback_error=RuntimeError("synthetic private rollback detail"),
        )
        repository, _ = self.make_repository([connection])

        with self.assertRaises(ComplaintPersistenceError) as caught:
            repository.register_complaint("synthetic-customer-17", "Synthetic complaint text.")

        message = str(caught.exception)
        self.assertNotIn("CPB-20261017-ABCDEF0123", message)
        self.assertNotIn("synthetic private insert detail", message)
        self.assertNotIn("synthetic private rollback detail", message)
        self.assertIn("connection.close", events)

    def test_cleanup_failures_do_not_turn_failed_persistence_into_success(self) -> None:
        events: list[str] = []
        connection = FakeConnection(
            events,
            execute_error=RuntimeError("synthetic database detail"),
            rollback_error=RuntimeError("synthetic rollback detail"),
            cursor_close_error=RuntimeError("synthetic cursor cleanup detail"),
            connection_close_error=RuntimeError("synthetic connection cleanup detail"),
        )
        repository, _ = self.make_repository([connection])

        with self.assertRaises(ComplaintPersistenceError) as caught:
            repository.register_complaint("synthetic-customer-17", "Synthetic complaint text.")

        self.assertNotIn("CPB-20261017-ABCDEF0123", str(caught.exception))
        self.assertNotIn("synthetic", str(caught.exception))
        self.assertIn("cursor.close", events)
        self.assertIn("rollback", events)
        self.assertIn("connection.close", events)

    def test_cleanup_failure_after_confirmed_commit_keeps_confirmed_reference(self) -> None:
        events: list[str] = []
        connection = FakeConnection(
            events,
            connection_close_error=RuntimeError("synthetic close detail"),
        )
        repository, _ = self.make_repository([connection])

        result = repository.register_complaint(
            "synthetic-customer-17", "Synthetic complaint text."
        )

        self.assertEqual(result, "CPB-20261017-ABCDEF0123")
        self.assertIn("commit", events)
        self.assertIn("connection.close", events)

    def test_case_reference_unique_collision_retries_with_new_reference(self) -> None:
        first_events: list[str] = []
        second_events: list[str] = []
        first = FakeConnection(first_events, execute_error=SyntheticUniqueViolation())
        second = FakeConnection(second_events)
        repository, factory = self.make_repository(
            [first, second],
            ["CPB-20261017-AAAAAAAAAA", "CPB-20261017-BBBBBBBBBB"],
        )

        result = repository.register_complaint("synthetic-customer-17", "A synthetic complaint.")

        self.assertEqual(result, "CPB-20261017-BBBBBBBBBB")
        self.assertIn("rollback", first_events)
        self.assertIn("commit", second_events)
        self.assertEqual(factory.call_count, 2)

    def test_repeated_reference_collisions_fail_without_disclosure(self) -> None:
        connections: list[FakeConnection] = []
        for _ in range(3):
            connections.append(
                FakeConnection([], execute_error=SyntheticUniqueViolation())
            )
        references = [
            "CPB-20261017-AAAAAAAAAA",
            "CPB-20261017-BBBBBBBBBB",
            "CPB-20261017-CCCCCCCCCC",
        ]
        repository, factory = self.make_repository(connections, references)

        with self.assertRaises(ComplaintPersistenceError) as caught:
            repository.register_complaint("synthetic-customer-17", "A synthetic complaint.")

        self.assertEqual(factory.call_count, 3)
        self.assertTrue(all("rollback" in connection.events for connection in connections))
        for reference in references:
            self.assertNotIn(reference, str(caught.exception))

    def test_entity_values_and_confidence_are_not_persisted(self) -> None:
        events: list[str] = []
        connection = FakeConnection(events)
        repository, _ = self.make_repository([connection])
        private_fixture = "synthetic-credential-value"
        assessment = {
            "classification": {"predicted_intent": "activate_my_card", "confidence": 0.99},
            "entities": [{"text": private_fixture, "entity_type": "OTP"}],
            "security_disclosure": {"detected": True, "categories": ["OTP"]},
            "priority_assessment": {
                "priority": "HIGH",
                "department_suggestion": "fraud-related review",
            },
        }

        repository.register_complaint("synthetic-customer-17", "A synthetic complaint.", assessment)

        serialized_parameters = repr(connection.cursor_instance.parameters)
        self.assertTrue(
            private_fixture not in serialized_parameters,
            "credential entity values must not be persisted separately",
        )
        self.assertNotIn("0.99", serialized_parameters)
        self.assertTrue(connection.cursor_instance.parameters[-1])

    def test_labeled_credentials_are_redacted_before_database_write(self) -> None:
        secrets_in_fixture = (
            "SyntheticPass123!",
            "4821",
            "731",
            "739184",
            "BlueCode42",
            "Verification77",
            "word phrase",
        )
        complaint = (
            "Password: SyntheticPass123! and I cannot sign in. "
            "PIN: 4821, CVV is 731; OTP 739184. "
            "Passcode = BlueCode42. Verification code: Verification77, "
            "one-time password is 'word phrase'. The transfer is still pending."
        )
        events: list[str] = []
        connection = FakeConnection(events)
        repository, _ = self.make_repository([connection])
        assessment = {"security_disclosure": {"detected": False}}

        repository.register_complaint(
            "synthetic-customer-17", complaint, assessment
        )

        persisted_text = connection.cursor_instance.parameters[5]
        self.assertIsInstance(persisted_text, str)
        self.assertNotIn("[REDACTED]", complaint)
        self.assertGreaterEqual(persisted_text.count("[REDACTED]"), 7)
        self.assertTrue(all(secret not in persisted_text for secret in secrets_in_fixture))
        self.assertIn("The transfer is still pending.", persisted_text)
        self.assertTrue(connection.cursor_instance.parameters[-1])

    def test_guest_contact_details_are_normalized_and_persisted_with_redacted_text(self) -> None:
        connection = FakeConnection([])
        repository, _ = self.make_repository([connection])
        complaint = "My OTP is 739184 and the synthetic transfer is pending."

        repository.register_complaint(
            None,
            complaint,
            contact_email=" Guest.Example@Example.test ",
            contact_phone="+971 (50) 123-4567",
        )

        params = connection.cursor_instance.parameters
        self.assertIsNone(params[2])
        self.assertEqual(params[3], "guest.example@example.test")
        self.assertEqual(params[4], "+971501234567")
        self.assertNotIn("739184", params[5])
        self.assertIn("[REDACTED]", params[5])

    def test_guest_requires_a_valid_contact_method_before_database_connection(self) -> None:
        connection = FakeConnection([])
        repository, factory = self.make_repository([connection])
        invalid_cases = (
            (None, None),
            ("invalid-email", None),
            (None, "not-a-phone"),
        )
        for email, phone in invalid_cases:
            with self.subTest(email_present=email is not None, phone_present=phone is not None):
                with self.assertRaises((TypeError, ValueError)):
                    repository.register_complaint(
                        None,
                        "A synthetic guest complaint.",
                        contact_email=email,
                        contact_phone=phone,
                    )
        factory.assert_not_called()

    def test_ambiguous_labeled_value_fails_closed_before_connection(self) -> None:
        connection = FakeConnection([])
        repository, factory = self.make_repository([connection])

        with self.assertRaises(ValueError) as caught:
            repository.register_complaint(
                "synthetic-customer-17",
                "Password is blue green and I cannot sign in.",
            )

        self.assertIn("no registration was attempted", str(caught.exception))
        self.assertNotIn("blue", str(caught.exception))
        factory.assert_not_called()

    def test_invalid_assessment_fields_fail_before_database_write(self) -> None:
        connection = FakeConnection([])
        repository, factory = self.make_repository([connection])
        invalid_assessments = (
            {"classification": {"predicted_intent": " "}},
            {"priority_assessment": {"priority": "URGENT"}},
            {"security_disclosure": {"detected": "yes"}},
        )
        for assessment in invalid_assessments:
            with self.subTest(fields=tuple(assessment)):
                with self.assertRaises(ValueError):
                    repository.register_complaint(
                        "synthetic-customer-17", "A synthetic complaint.", assessment
                    )
        factory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
