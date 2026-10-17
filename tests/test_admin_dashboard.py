"""Offline tests for administrator authentication and complaint operations."""

from __future__ import annotations

import unittest
import hmac
from pathlib import Path
from unittest.mock import Mock, patch

from admin_app import _contact_value_for_display
from src.admin_auth import AdminAuthenticator, clear_admin_session
from src.admin_repository import (
    ALLOWED_PRIORITIES,
    ALLOWED_STATUSES,
    AdminComplaintRepository,
    AdminRepositoryError,
)
from src.admin_service import (
    AdminAuthorizationError,
    AdminComplaintService,
    AdminOperationError,
)


class FakeCursor:
    def __init__(self, *, rows=(), one=None, fail_execute=None):
        self.rows = list(rows)
        self.one = one
        self.fail_execute = fail_execute
        self.executions = []
        self.closed = False

    def execute(self, query, parameters=None):
        self.executions.append((query, parameters))
        if self.fail_execute is not None:
            raise self.fail_execute

    def fetchone(self):
        return self.one

    def fetchall(self):
        return self.rows

    def close(self):
        self.closed = True


class FakeConnection:
    def __init__(self, cursor):
        self._cursor = cursor
        self.committed = False
        self.rolled_back = False
        self.closed = False

    def cursor(self):
        return self._cursor

    def commit(self):
        self.committed = True

    def rollback(self):
        self.rolled_back = True

    def close(self):
        self.closed = True


class AdminDashboardTests(unittest.TestCase):
    def test_correct_credentials_are_accepted_and_incorrect_credentials_rejected(self):
        auth = AdminAuthenticator(username="demo-admin", password="synthetic-password")

        with patch("src.admin_auth.hmac.compare_digest", wraps=hmac.compare_digest) as compare:
            self.assertTrue(auth.authenticate("demo-admin", "synthetic-password"))
        self.assertEqual(compare.call_count, 2)
        self.assertFalse(auth.authenticate("demo-admin", "wrong-password"))
        self.assertFalse(auth.authenticate("wrong-user", "synthetic-password"))

    def test_missing_credentials_are_rejected_without_disclosure(self):
        auth = AdminAuthenticator(username="", password="")

        self.assertFalse(auth.is_configured)
        self.assertFalse(auth.authenticate("", ""))
        self.assertNotIn("synthetic", repr(auth))

    def test_auth_failure_copy_and_source_do_not_expose_configured_credentials(self):
        auth_source = (Path(__file__).resolve().parents[1] / "src" / "admin_auth.py").read_text(
            encoding="utf-8"
        )
        app_source = (Path(__file__).resolve().parents[1] / "admin_app.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("12345", auth_source + app_source)
        self.assertNotIn("kavin", auth_source + app_source)
        self.assertIn("The username or password is incorrect.", app_source)

    def test_authenticated_service_allows_overview_access(self):
        repository = Mock()
        repository.get_overview.return_value = {
            "total": 0,
            "by_status": {},
            "by_priority": {},
            "recent": [],
        }
        service = AdminComplaintService(repository, is_authenticated=lambda: True)

        result = service.get_overview()

        self.assertEqual(result["total"], 0)
        repository.get_overview.assert_called_once_with()

    def test_all_db_operations_require_server_side_authentication(self):
        repository = Mock()
        is_authenticated = Mock(return_value=False)
        service = AdminComplaintService(repository, is_authenticated=is_authenticated)
        operations = (
            service.get_overview,
            service.list_complaints,
            lambda: service.get_complaint("CPB-20261017-ABCDEF0123"),
            lambda: service.update_status("CPB-20261017-ABCDEF0123", "CLOSED"),
        )

        for operation in operations:
            with self.subTest(operation=operation):
                with self.assertRaises(AdminAuthorizationError):
                    operation()

        self.assertEqual(is_authenticated.call_count, 4)
        repository.get_overview.assert_not_called()
        repository.list_complaints.assert_not_called()
        repository.get_complaint.assert_not_called()
        repository.update_status.assert_not_called()

    def test_search_and_filters_are_passed_as_sql_parameters(self):
        cursor = FakeCursor(rows=[])
        connection = FakeConnection(cursor)
        repository = AdminComplaintRepository(
            "postgresql://synthetic",
            connection_factory=lambda _url: connection,
        )

        result = repository.list_complaints(
            case_reference="CPB-TEST",
            status="RECEIVED",
            priority="HIGH",
            limit=25,
        )

        self.assertEqual(result, [])
        query, parameters = cursor.executions[0]
        self.assertIn("case_reference ILIKE %s", query)
        self.assertIn("status = %s", query)
        self.assertIn("priority = %s", query)
        self.assertEqual(parameters, ("%CPB-TEST%", "RECEIVED", "HIGH", 25))
        self.assertTrue(cursor.closed)
        self.assertTrue(connection.closed)

    def test_detail_returns_contact_email_and_phone_from_authorized_query(self):
        case_reference = "CPB-20261017-ABCDEF0123"
        cursor = FakeCursor(
            one=(
                case_reference,
                None,
                "guest@example.test",
                "+971501234567",
                "Synthetic complaint text.",
                "cash_withdrawal",
                "HIGH",
                "Review Team",
                False,
                "RECEIVED",
                "created",
                "updated",
            )
        )
        connection = FakeConnection(cursor)
        repository = AdminComplaintRepository(
            "postgresql://synthetic",
            connection_factory=lambda _url: connection,
        )
        service = AdminComplaintService(repository, is_authenticated=lambda: True)

        detail = service.get_complaint(case_reference)

        self.assertEqual(detail["contact_email"], "guest@example.test")
        self.assertEqual(detail["contact_phone"], "+971501234567")
        query, parameters = cursor.executions[0]
        self.assertIn("contact_email", query)
        self.assertIn("contact_phone", query)
        self.assertEqual(parameters, (case_reference,))

    def test_overview_row_does_not_include_contact_details(self):
        row = AdminComplaintRepository._summary_row(
            (
                "CPB-20261017-ABCDEF0123",
                "RECEIVED",
                "cash_withdrawal",
                "HIGH",
                "Review Team",
                False,
                "created",
            )
        )

        self.assertNotIn("contact_email", row)
        self.assertNotIn("contact_phone", row)

    def test_null_contact_values_are_displayed_as_not_provided(self):
        self.assertEqual(_contact_value_for_display(None), "Not provided")
        self.assertEqual(_contact_value_for_display(""), "Not provided")
        self.assertEqual(_contact_value_for_display("   "), "Not provided")
        self.assertEqual(
            _contact_value_for_display("guest@example.test"),
            "guest@example.test",
        )

    def test_unauthenticated_detail_request_never_reads_contacts(self):
        repository = Mock()
        service = AdminComplaintService(repository, is_authenticated=lambda: False)

        with self.assertRaises(AdminAuthorizationError):
            service.get_complaint("CPB-20261017-ABCDEF0123")

        repository.get_complaint.assert_not_called()

    def test_schema_status_and_priority_values_are_enforced(self):
        self.assertEqual(ALLOWED_STATUSES, ("RECEIVED", "IN_REVIEW", "RESOLVED", "CLOSED"))
        self.assertEqual(ALLOWED_PRIORITIES, ("LOW", "MEDIUM", "HIGH"))
        connection_factory = Mock()
        repository = AdminComplaintRepository(
            "postgresql://synthetic",
            connection_factory=connection_factory,
        )

        for value in ("NEW", "DELETED", "PENDING"):
            with self.subTest(status=value):
                with self.assertRaises(ValueError):
                    repository.update_status("CPB-20261017-ABCDEF0123", value)
        connection_factory.assert_not_called()

    def test_status_update_is_parameterized_and_committed(self):
        cursor = FakeCursor(one=("CPB-20261017-ABCDEF0123", "IN_REVIEW", "timestamp"))
        connection = FakeConnection(cursor)
        repository = AdminComplaintRepository(
            "postgresql://synthetic",
            connection_factory=lambda _url: connection,
        )
        service = AdminComplaintService(repository, is_authenticated=lambda: True)

        result = service.update_status("CPB-20261017-ABCDEF0123", "IN_REVIEW")

        self.assertEqual(result["status"], "IN_REVIEW")
        self.assertEqual(
            cursor.executions[0][1],
            ("IN_REVIEW", "CPB-20261017-ABCDEF0123"),
        )
        self.assertTrue(connection.committed)
        self.assertTrue(cursor.closed)
        self.assertTrue(connection.closed)

    def test_failed_update_and_database_errors_are_safe(self):
        secret_like_error = "synthetic raw SQL failure with credential detail"
        cursor = FakeCursor(fail_execute=RuntimeError(secret_like_error))
        connection = FakeConnection(cursor)
        repository = AdminComplaintRepository(
            "postgresql://synthetic",
            connection_factory=lambda _url: connection,
        )
        service = AdminComplaintService(repository, is_authenticated=lambda: True)

        with self.assertRaises(AdminOperationError) as raised:
            service.update_status("CPB-20261017-ABCDEF0123", "CLOSED")

        self.assertNotIn(secret_like_error, str(raised.exception))
        self.assertNotIn("postgresql://synthetic", str(raised.exception))
        self.assertTrue(connection.rolled_back)
        self.assertTrue(cursor.closed)
        self.assertTrue(connection.closed)

    def test_missing_complaint_status_update_is_not_reported_as_success(self):
        repository = Mock()
        repository.update_status.return_value = None
        service = AdminComplaintService(repository, is_authenticated=lambda: True)

        result = service.update_status("CPB-20261017-ABCDEF0123", "CLOSED")

        self.assertIsNone(result)
        repository.update_status.assert_called_once_with(
            "CPB-20261017-ABCDEF0123", "CLOSED"
        )

    def test_repository_masks_connection_failure(self):
        repository = AdminComplaintRepository(
            "postgresql://synthetic",
            connection_factory=lambda _url: (_ for _ in ()).throw(
                RuntimeError("synthetic raw database failure")
            ),
        )

        with self.assertRaises(AdminRepositoryError) as raised:
            repository.get_overview()
        self.assertNotIn("synthetic raw database failure", str(raised.exception))

    def test_logout_clears_admin_session_only(self):
        state = {
            "_admin_authenticated": True,
            "admin_selected_case": "CPB-20261017-ABCDEF0123",
            "complaint_chat_messages": [{"role": "assistant", "content": "Guest chat"}],
        }

        clear_admin_session(state)

        self.assertNotIn("_admin_authenticated", state)
        self.assertNotIn("admin_selected_case", state)
        self.assertIn("complaint_chat_messages", state)

    def test_customer_app_remains_separate_and_guest_based(self):
        customer_source = (Path(__file__).resolve().parents[1] / "customer_app.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("AdminAuthenticator", customer_source)
        self.assertNotIn("admin_app", customer_source)
        self.assertIn("controller.submit_complaint", customer_source)


if __name__ == "__main__":
    unittest.main()
