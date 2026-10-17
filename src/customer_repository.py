"""Parameterized PostgreSQL repository for customer authentication records."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable
from uuid import uuid4

from src.runtime_config import get_setting


class CustomerRepositoryError(RuntimeError):
    """Safe repository error that excludes driver details and credentials."""


class DuplicateCustomerEmailError(CustomerRepositoryError):
    """Raised internally when the normalized email is already registered."""


@dataclass(frozen=True)
class CustomerCredentialRecord:
    """Internal credential record; password hashes are excluded from repr."""

    customer_id: str
    password_hash: str = field(repr=False)
    account_status: str


def _default_connection_factory(database_url: str) -> Any:
    try:
        import psycopg
    except ImportError:
        raise CustomerRepositoryError(
            "The psycopg PostgreSQL driver is required for customer accounts."
        ) from None
    return psycopg.connect(database_url)


class CustomerRepository:
    """Create and fetch customer credential records without authenticating users.

    Construction validates configuration but does not connect. A PostgreSQL
    connection is opened only by an explicit repository operation.
    """

    def __init__(
        self,
        database_url: str | None = None,
        *,
        connection_factory: Callable[[str], Any] | None = None,
    ) -> None:
        configured_url = database_url if database_url is not None else get_setting("DATABASE_URL")
        if not isinstance(configured_url, str) or not configured_url.strip():
            raise ValueError("DATABASE_URL must be configured before customer account access.")
        self._database_url = configured_url.strip()
        self._connection_factory = connection_factory or _default_connection_factory

    @staticmethod
    def _validate_normalized_email(normalized_email: str) -> None:
        if not isinstance(normalized_email, str):
            raise TypeError("normalized_email must be a string.")
        if (
            not normalized_email
            or len(normalized_email) > 254
            or normalized_email != normalized_email.strip().casefold()
            or any(ord(char) < 32 or char.isspace() for char in normalized_email)
            or normalized_email.count("@") != 1
        ):
            raise ValueError("normalized_email is invalid or not normalized.")

    @staticmethod
    def _is_duplicate_email(error: Exception) -> bool:
        diagnostic = getattr(error, "diag", None)
        return (
            getattr(error, "sqlstate", None) == "23505"
            and getattr(diagnostic, "constraint_name", None)
            == "customers_normalized_email_key"
        )

    def create_customer(self, normalized_email: str, password_hash: str) -> str:
        """Insert an account and return its UUID string after commit succeeds."""
        self._validate_normalized_email(normalized_email)
        if not isinstance(password_hash, str) or not password_hash.startswith("scrypt$"):
            raise ValueError("password_hash must use the configured password-hash format.")

        customer_id = uuid4()
        connection: Any | None = None
        cursor: Any | None = None
        try:
            connection = self._connection_factory(self._database_url)
            cursor = connection.cursor()
            cursor.execute(
                """INSERT INTO customers (
                       customer_id, normalized_email, password_hash, account_status
                   ) VALUES (%s, %s, %s, 'ACTIVE')
                   RETURNING customer_id""",
                (customer_id, normalized_email, password_hash),
            )
            inserted = cursor.fetchone()
            if not inserted:
                raise RuntimeError("insert returned no record")
            cursor.close()
            cursor = None
            connection.commit()
            return str(inserted[0])
        except Exception as exc:
            duplicate = self._is_duplicate_email(exc)
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if connection is not None:
                try:
                    connection.rollback()
                except Exception:
                    pass
            if duplicate:
                raise DuplicateCustomerEmailError(
                    "An account with this email could not be created."
                ) from None
            raise CustomerRepositoryError(
                "Customer account creation could not be confirmed."
            ) from None
        finally:
            if connection is not None:
                try:
                    connection.close()
                except Exception:
                    pass

    def find_by_normalized_email(
        self, normalized_email: str
    ) -> CustomerCredentialRecord | None:
        """Fetch internal verification data for one normalized email."""
        self._validate_normalized_email(normalized_email)
        connection: Any | None = None
        cursor: Any | None = None
        try:
            connection = self._connection_factory(self._database_url)
            cursor = connection.cursor()
            cursor.execute(
                """SELECT customer_id, password_hash, account_status
                   FROM customers WHERE normalized_email = %s""",
                (normalized_email,),
            )
            row = cursor.fetchone()
            if row is None:
                return None
            return CustomerCredentialRecord(
                customer_id=str(row[0]),
                password_hash=row[1],
                account_status=row[2],
            )
        except Exception:
            raise CustomerRepositoryError(
                "Customer account lookup could not be completed."
            ) from None
        finally:
            if cursor is not None:
                try:
                    cursor.close()
                except Exception:
                    pass
            if connection is not None:
                try:
                    # SELECT opens a transaction with psycopg's default mode.
                    connection.rollback()
                except Exception:
                    pass
                try:
                    connection.close()
                except Exception:
                    pass
