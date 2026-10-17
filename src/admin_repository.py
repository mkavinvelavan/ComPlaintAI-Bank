"""Read and update complaint records for authenticated administrator operations."""

from __future__ import annotations

from typing import Any

from src.runtime_config import get_setting

# These values match the CHECK constraints in migrations/001_create_complaints.sql.
ALLOWED_STATUSES = ("RECEIVED", "IN_REVIEW", "RESOLVED", "CLOSED")
ALLOWED_PRIORITIES = ("LOW", "MEDIUM", "HIGH")

_OVERVIEW_TOTAL_SQL = "SELECT COUNT(*) FROM complaints"
_OVERVIEW_STATUS_SQL = (
    "SELECT status, COUNT(*) FROM complaints GROUP BY status ORDER BY status"
)
_OVERVIEW_PRIORITY_SQL = (
    "SELECT priority, COUNT(*) FROM complaints GROUP BY priority ORDER BY priority"
)
_RECENT_COMPLAINTS_SQL = """
SELECT case_reference, status, predicted_intent, priority, department_suggestion,
       credential_disclosure_detected, created_at
FROM complaints
ORDER BY created_at DESC
LIMIT %s
"""
_LIST_COMPLAINTS_SQL = """
SELECT case_reference, status, predicted_intent, priority, department_suggestion,
       credential_disclosure_detected, created_at
FROM complaints
"""
_DETAIL_COMPLAINT_SQL = """
SELECT case_reference, customer_id, contact_email, contact_phone, complaint_text,
       predicted_intent, priority, department_suggestion,
       credential_disclosure_detected, status, created_at, updated_at
FROM complaints
WHERE case_reference = %s
"""
_UPDATE_STATUS_SQL = """
UPDATE complaints
SET status = %s
WHERE case_reference = %s
RETURNING case_reference, status, updated_at
"""


class AdminRepositoryError(RuntimeError):
    """Safe database error that does not contain driver or query details."""


def _default_connection_factory(database_url: str) -> Any:
    try:
        import psycopg
    except ImportError:
        raise AdminRepositoryError(
            "The PostgreSQL driver is unavailable for administrator operations."
        ) from None
    return psycopg.connect(database_url)


class AdminComplaintRepository:
    """Parameterized PostgreSQL queries for the admin dashboard.

    This repository contains no authentication logic. Call it only through
    AdminComplaintService, which authorizes every operation before connecting.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        connection_factory=None,
    ) -> None:
        configured_url = database_url if database_url is not None else get_setting("DATABASE_URL")
        if not isinstance(configured_url, str) or not configured_url.strip():
            raise AdminRepositoryError(
                "The complaint database is not configured for administrator access."
            )
        self._database_url = configured_url.strip()
        self._connection_factory = connection_factory or _default_connection_factory

    @staticmethod
    def _summary_row(row: Any) -> dict[str, Any]:
        return {
            "case_reference": row[0],
            "status": row[1],
            "predicted_intent": row[2],
            "priority": row[3],
            "department_suggestion": row[4],
            "credential_disclosure_detected": bool(row[5]),
            "created_at": row[6],
        }

    @staticmethod
    def _detail_row(row: Any) -> dict[str, Any]:
        return {
            "case_reference": row[0],
            "customer_id": row[1],
            "contact_email": row[2],
            "contact_phone": row[3],
            "complaint_text": row[4],
            "predicted_intent": row[5],
            "priority": row[6],
            "department_suggestion": row[7],
            "credential_disclosure_detected": bool(row[8]),
            "status": row[9],
            "created_at": row[10],
            "updated_at": row[11],
        }

    def _run(self, operation):
        connection = None
        cursor = None
        try:
            connection = self._connection_factory(self._database_url)
            cursor = connection.cursor()
            return operation(cursor, connection)
        except AdminRepositoryError:
            raise
        except Exception:
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    pass
            raise AdminRepositoryError(
                "The complaint database operation could not be completed."
            ) from None
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def get_overview(self, *, recent_limit: int = 10) -> dict[str, Any]:
        if isinstance(recent_limit, bool) or not isinstance(recent_limit, int) or not 1 <= recent_limit <= 100:
            raise ValueError("recent_limit must be between 1 and 100.")

        def operation(cursor, _connection):
            cursor.execute(_OVERVIEW_TOTAL_SQL)
            total = cursor.fetchone()[0]
            cursor.execute(_OVERVIEW_STATUS_SQL)
            statuses = {row[0]: row[1] for row in cursor.fetchall()}
            cursor.execute(_OVERVIEW_PRIORITY_SQL)
            priorities = {
                ("UNASSIGNED" if row[0] is None else row[0]): row[1]
                for row in cursor.fetchall()
            }
            cursor.execute(_RECENT_COMPLAINTS_SQL, (recent_limit,))
            recent = [self._summary_row(row) for row in cursor.fetchall()]
            return {
                "total": total,
                "by_status": statuses,
                "by_priority": priorities,
                "recent": recent,
            }

        return self._run(operation)

    def list_complaints(
        self,
        *,
        case_reference: str | None = None,
        status: str | None = None,
        priority: str | None = None,
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        if case_reference is not None:
            if not isinstance(case_reference, str) or len(case_reference) > 64:
                raise ValueError("case_reference search must be at most 64 characters.")
            case_reference = case_reference.strip()
            if any(ord(char) < 32 for char in case_reference):
                raise ValueError("case_reference search contains invalid characters.")
        if status is not None and status not in ALLOWED_STATUSES:
            raise ValueError("status is not permitted by the complaints schema.")
        if priority is not None and priority not in ALLOWED_PRIORITIES:
            raise ValueError("priority is not permitted by the complaints schema.")
        if isinstance(limit, bool) or not isinstance(limit, int) or not 1 <= limit <= 500:
            raise ValueError("limit must be between 1 and 500.")

        conditions: list[str] = []
        parameters: list[Any] = []
        if case_reference:
            conditions.append("case_reference ILIKE %s")
            parameters.append(f"%{case_reference}%")
        if status:
            conditions.append("status = %s")
            parameters.append(status)
        if priority:
            conditions.append("priority = %s")
            parameters.append(priority)
        query = _LIST_COMPLAINTS_SQL
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY created_at DESC LIMIT %s"
        parameters.append(limit)

        def operation(cursor, _connection):
            cursor.execute(query, tuple(parameters))
            return [self._summary_row(row) for row in cursor.fetchall()]

        return self._run(operation)

    def get_complaint(self, case_reference: str) -> dict[str, Any] | None:
        if not isinstance(case_reference, str) or not case_reference.strip() or len(case_reference) > 64:
            raise ValueError("case_reference is invalid.")

        def operation(cursor, _connection):
            cursor.execute(_DETAIL_COMPLAINT_SQL, (case_reference.strip(),))
            row = cursor.fetchone()
            return self._detail_row(row) if row else None

        return self._run(operation)

    def update_status(self, case_reference: str, status: str) -> dict[str, Any] | None:
        if status not in ALLOWED_STATUSES:
            raise ValueError("status is not permitted by the complaints schema.")
        if not isinstance(case_reference, str) or not case_reference.strip() or len(case_reference) > 64:
            raise ValueError("case_reference is invalid.")

        def operation(cursor, connection):
            cursor.execute(_UPDATE_STATUS_SQL, (status, case_reference.strip()))
            row = cursor.fetchone()
            if row is None:
                connection.rollback()
                return None
            connection.commit()
            return {
                "case_reference": row[0],
                "status": row[1],
                "updated_at": row[2],
            }

        return self._run(operation)
