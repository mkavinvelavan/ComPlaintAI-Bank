"""Offline tests for customer account creation and password authentication."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import Mock

from src.customer_auth import (
    CustomerAccountCreationError,
    CustomerAuthService,
    CustomerAuthenticationError,
    ScryptPasswordHasher,
)
from src.customer_repository import (
    CustomerCredentialRecord,
    DuplicateCustomerEmailError,
)


class CustomerAuthTests(unittest.TestCase):
    def setUp(self) -> None:
        self.email = "Customer.Example@Example.test"
        self.normalized_email = "customer.example@example.test"
        self.password = "Synthetic-Password-For-Test-8274"
        self.customer_id = "3f3c0f1a-5c34-4ea8-a34a-76f394e1be5b"
        self.repository = Mock()
        self.repository.create_customer.return_value = self.customer_id
        self.hasher = ScryptPasswordHasher()
        self.service = CustomerAuthService(
            repository=self.repository,
            password_hasher=self.hasher,
        )

    def test_signup_normalizes_email_and_stores_only_a_salted_hash(self) -> None:
        result = self.service.sign_up(self.email, self.password)

        self.repository.create_customer.assert_called_once()
        saved_email, saved_hash = self.repository.create_customer.call_args.args
        self.assertEqual(saved_email, self.normalized_email)
        self.assertTrue(saved_hash.startswith("scrypt$"), "a versioned password hash is required")
        self.assertTrue(saved_hash != self.password, "plaintext password must not be stored")
        self.assertTrue(self.hasher.verify_password(self.password, saved_hash))
        self.assertEqual(result, {"created": True})
        self.assertNotIn("customer_id", result)
        self.assertNotIn("password_hash", result)

    def test_duplicate_normalized_email_is_rejected_with_safe_error(self) -> None:
        self.repository.create_customer.side_effect = DuplicateCustomerEmailError(
            "synthetic duplicate account detail"
        )

        with self.assertRaises(CustomerAccountCreationError) as caught:
            self.service.sign_up(" CUSTOMER.EXAMPLE@example.test ", self.password)

        saved_email, saved_hash = self.repository.create_customer.call_args.args
        self.assertEqual(saved_email, self.normalized_email)
        self.assertTrue(saved_hash.startswith("scrypt$"))
        self.assertTrue(str(caught.exception) == "Account could not be created.")
        self.assertTrue(self.normalized_email not in str(caught.exception))
        self.assertTrue("synthetic duplicate" not in str(caught.exception))

    def test_correct_password_returns_only_authenticated_customer_id(self) -> None:
        password_hash = self.hasher.hash_password(self.password)
        self.repository.find_by_normalized_email.return_value = CustomerCredentialRecord(
            self.customer_id, password_hash, "ACTIVE"
        )

        identity = self.service.log_in(" CUSTOMER.EXAMPLE@example.test ", self.password)

        self.repository.find_by_normalized_email.assert_called_once_with(
            self.normalized_email
        )
        self.assertEqual(identity, {"customer_id": self.customer_id})
        self.assertNotIn("password_hash", identity)
        self.assertNotIn("email", identity)

    def test_incorrect_password_returns_generic_login_error(self) -> None:
        password_hash = self.hasher.hash_password(self.password)
        self.repository.find_by_normalized_email.return_value = CustomerCredentialRecord(
            self.customer_id, password_hash, "ACTIVE"
        )

        with self.assertRaises(CustomerAuthenticationError) as caught:
            self.service.log_in(self.email, "Synthetic-Wrong-Password-44")

        self.assertEqual(str(caught.exception), "Invalid email or password.")
        self.assertTrue(self.customer_id not in str(caught.exception))

    def test_inactive_account_returns_same_generic_login_error(self) -> None:
        password_hash = self.hasher.hash_password(self.password)
        self.repository.find_by_normalized_email.return_value = CustomerCredentialRecord(
            self.customer_id, password_hash, "INACTIVE"
        )

        with self.assertRaises(CustomerAuthenticationError) as caught:
            self.service.log_in(self.email, self.password)

        self.assertEqual(str(caught.exception), "Invalid email or password.")
        self.assertTrue(self.customer_id not in str(caught.exception))

    def test_unknown_email_returns_same_generic_login_error(self) -> None:
        self.repository.find_by_normalized_email.return_value = None

        with self.assertRaises(CustomerAuthenticationError) as caught:
            self.service.log_in(self.email, self.password)

        self.assertEqual(str(caught.exception), "Invalid email or password.")
        self.assertTrue(self.normalized_email not in str(caught.exception))

    def test_signup_input_validation_prevents_repository_calls(self) -> None:
        invalid_inputs = (
            ("not-an-email", self.password, ValueError),
            ("a..b@example.test", self.password, ValueError),
            (self.email, "short", ValueError),
            (self.email, "x" * 1025, ValueError),
            (None, self.password, TypeError),
            (self.email, None, TypeError),
        )
        for email, password, error_type in invalid_inputs:
            with self.subTest(error_type=error_type.__name__):
                with self.assertRaises(error_type):
                    self.service.sign_up(email, password)  # type: ignore[arg-type]
        self.repository.create_customer.assert_not_called()

    def test_invalid_login_input_uses_generic_error(self) -> None:
        for email, password in (("invalid", self.password), (self.email, "")):
            with self.subTest(email_type=type(email).__name__):
                with self.assertRaises(CustomerAuthenticationError) as caught:
                    self.service.log_in(email, password)
                self.assertEqual(str(caught.exception), "Invalid email or password.")
        self.repository.find_by_normalized_email.assert_not_called()

    def test_repository_errors_do_not_expose_hash_or_exception_details(self) -> None:
        synthetic_hash_marker = "synthetic-private-hash-marker"
        synthetic_db_detail = "synthetic private lookup failure"
        self.repository.find_by_normalized_email.side_effect = RuntimeError(
            f"{synthetic_db_detail} {synthetic_hash_marker} {self.email}"
        )

        with self.assertRaises(CustomerAuthenticationError) as caught:
            self.service.log_in(self.email, self.password)

        message = str(caught.exception)
        self.assertEqual(message, "Invalid email or password.")
        self.assertTrue(synthetic_hash_marker not in message)
        self.assertTrue(synthetic_db_detail not in message)
        self.assertTrue(self.email not in message)

    def test_repository_is_constructed_only_when_database_access_is_requested(self) -> None:
        factory = Mock(side_effect=ValueError("DATABASE_URL is not configured"))
        service = CustomerAuthService(repository_factory=factory)

        factory.assert_not_called()
        with self.assertRaises(CustomerAccountCreationError) as caught:
            service.sign_up(self.email, self.password)
        factory.assert_called_once_with()
        self.assertEqual(str(caught.exception), "Account could not be created.")

    def test_customer_schema_is_additive_and_compatible_with_complaint_customer_id(self) -> None:
        migration = (
            Path(__file__).resolve().parents[1]
            / "migrations"
            / "002_create_customers.sql"
        ).read_text(encoding="utf-8").lower()
        normalized = " ".join(migration.split())
        for expected in (
            "create table if not exists customers",
            "customer_id uuid primary key",
            "normalized_email varchar(254) not null",
            "unique (normalized_email)",
            "password_hash text not null",
            "account_status varchar(16) not null default 'active'",
            "created_at timestamptz not null default current_timestamp",
            "updated_at timestamptz not null default current_timestamp",
        ):
            with self.subTest(expected=expected):
                self.assertIn(expected, normalized)
        self.assertIn("complaints.customer_id remains text", normalized)
        self.assertNotIn("admin", normalized)


if __name__ == "__main__":
    unittest.main()
