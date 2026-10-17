"""Customer sign-up and login service with salted scrypt password hashes."""

from __future__ import annotations

import base64
import hashlib
import hmac
import re
import secrets
from collections.abc import Callable
from typing import TypedDict

from src.customer_repository import (
    CustomerRepository,
    CustomerRepositoryError,
    DuplicateCustomerEmailError,
)

_LOCAL_PART_RE = re.compile(r"^[A-Za-z0-9!#$%&'*+/=?^_`{|}~.-]+$")
_DOMAIN_LABEL_RE = re.compile(r"^[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?$")
SIGNUP_PASSWORD_MIN_CHARACTERS = 12
SIGNUP_PASSWORD_MAX_UTF8_BYTES = 1024


class AuthenticatedCustomer(TypedDict):
    """Minimal identity returned only after active-account password verification."""

    customer_id: str


class AccountCreated(TypedDict):
    """Non-authenticated sign-up acknowledgment; contains no customer identity."""

    created: bool


class CustomerAuthenticationError(RuntimeError):
    """Generic login failure that does not reveal account existence or status."""


class CustomerAccountCreationError(RuntimeError):
    """Safe sign-up failure that does not reveal whether email is registered."""


class ScryptPasswordHasher:
    """Versioned hashes using stdlib scrypt, a random salt, and constant-time compare.

    Parameters use the OWASP-listed scrypt option N=2^14, r=8, p=5. Reassess
    latency and memory cost on the eventual deployment platform before launch.
    """

    _N = 2**14
    _R = 8
    _P = 5
    _KEY_LENGTH = 32

    @staticmethod
    def _b64encode(value: bytes) -> str:
        return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")

    @staticmethod
    def _b64decode(value: str) -> bytes:
        padding = "=" * (-len(value) % 4)
        return base64.urlsafe_b64decode(value + padding)

    def hash_password(self, password: str) -> str:
        """Return an encoded salted hash; never return or retain the plaintext."""
        salt_bytes = secrets.token_bytes(16)
        derived = hashlib.scrypt(
            password.encode("utf-8"),
            salt=salt_bytes,
            n=self._N,
            r=self._R,
            p=self._P,
            dklen=self._KEY_LENGTH,
            maxmem=64 * 1024 * 1024,
        )
        return (
            f"scrypt${self._N}${self._R}${self._P}$"
            f"{self._b64encode(salt_bytes)}${self._b64encode(derived)}"
        )

    def verify_password(self, password: str, encoded_hash: str) -> bool:
        """Verify a password in constant time; malformed hashes fail closed."""
        try:
            algorithm, n_text, r_text, p_text, salt_text, digest_text = encoded_hash.split("$")
            if algorithm != "scrypt":
                return False
            n, r, p = int(n_text), int(r_text), int(p_text)
            if (
                n < 2**14
                or n > 2**18
                or n & (n - 1)
                or not 1 <= r <= 16
                or not 1 <= p <= 5
            ):
                return False
            salt = self._b64decode(salt_text)
            expected = self._b64decode(digest_text)
            if not 16 <= len(salt) <= 64 or len(expected) != self._KEY_LENGTH:
                return False
            actual = hashlib.scrypt(
                password.encode("utf-8"),
                salt=salt,
                n=n,
                r=r,
                p=p,
                dklen=len(expected),
                maxmem=64 * 1024 * 1024,
            )
            return hmac.compare_digest(actual, expected)
        except (AttributeError, TypeError, ValueError, UnicodeError, OverflowError):
            return False


def normalize_email(email: str) -> str:
    """Trim and case-normalize a basic ASCII email address consistently.

    The application treats the local part as case-insensitive. International
    domain names are converted to their IDNA ASCII form before normalization.
    """
    if not isinstance(email, str):
        raise TypeError("email must be a string.")
    candidate = email.strip()
    if not candidate or len(candidate) > 254 or candidate.count("@") != 1:
        raise ValueError("email address is invalid.")
    local, domain = candidate.rsplit("@", 1)
    if (
        not local
        or len(local) > 64
        or not _LOCAL_PART_RE.fullmatch(local)
        or local.startswith(".")
        or local.endswith(".")
        or ".." in local
        or not domain
    ):
        raise ValueError("email address is invalid.")
    try:
        ascii_domain = domain.encode("idna").decode("ascii").casefold()
    except UnicodeError:
        raise ValueError("email address is invalid.") from None
    labels = ascii_domain.split(".")
    if (
        len(labels) < 2
        or any(not _DOMAIN_LABEL_RE.fullmatch(label) for label in labels)
        or len(ascii_domain) > 253
    ):
        raise ValueError("email address is invalid.")
    normalized = f"{local.casefold()}@{ascii_domain}"
    if len(normalized) > 254:
        raise ValueError("email address is invalid.")
    return normalized


def _validate_signup_password(password: str) -> None:
    if not isinstance(password, str):
        raise TypeError("password must be a string.")
    try:
        byte_length = len(password.encode("utf-8"))
    except UnicodeError:
        raise ValueError("password contains unsupported characters.") from None
    if (
        len(password) < SIGNUP_PASSWORD_MIN_CHARACTERS
        or byte_length > SIGNUP_PASSWORD_MAX_UTF8_BYTES
    ):
        raise ValueError(
            "password must be at least "
            f"{SIGNUP_PASSWORD_MIN_CHARACTERS} characters and no more than "
            f"{SIGNUP_PASSWORD_MAX_UTF8_BYTES} UTF-8 bytes."
        )


class CustomerAuthService:
    """Create customer accounts and return identity only after verified login.

    Customer sign-up never grants administrator privileges. Login returns only
    a customer ID; a caller must still enforce session, transport, and request
    authorization controls before using it for complaint registration.
    """

    def __init__(
        self,
        repository: CustomerRepository | None = None,
        *,
        password_hasher: ScryptPasswordHasher | None = None,
        repository_factory: Callable[[], CustomerRepository] | None = None,
    ) -> None:
        if repository is not None and repository_factory is not None:
            raise ValueError("Provide either repository or repository_factory, not both.")
        self.repository = repository
        self._repository_factory = repository_factory or CustomerRepository
        self._password_hasher = password_hasher or ScryptPasswordHasher()
        # Use a random, process-local hash to perform comparable KDF work when
        # an email is absent. This is neither a usable account nor a secret.
        self._dummy_password_hash = self._password_hasher.hash_password(
            secrets.token_urlsafe(32)
        )

    def _get_repository(self) -> CustomerRepository:
        if self.repository is None:
            self.repository = self._repository_factory()
        return self.repository

    def sign_up(self, email: str, password: str) -> AccountCreated:
        """Create an account without returning an authenticated identity."""
        normalized_email = normalize_email(email)
        _validate_signup_password(password)
        try:
            password_hash = self._password_hasher.hash_password(password)
            self._get_repository().create_customer(normalized_email, password_hash)
        except (DuplicateCustomerEmailError, CustomerRepositoryError):
            raise CustomerAccountCreationError(
                "Account could not be created."
            ) from None
        except Exception:
            raise CustomerAccountCreationError(
                "Account could not be created."
            ) from None
        return {"created": True}

    def log_in(self, email: str, password: str) -> AuthenticatedCustomer:
        """Verify credentials and return a minimal identity for an active account.

        All credential lookup, verification, and inactive-account failures use
        the same message so the response does not disclose account existence.
        """
        try:
            normalized_email = normalize_email(email)
            if not isinstance(password, str) or not password:
                raise ValueError("invalid login credentials")
            if len(password.encode("utf-8")) > SIGNUP_PASSWORD_MAX_UTF8_BYTES:
                raise ValueError("invalid login credentials")
            record = self._get_repository().find_by_normalized_email(normalized_email)
            if record is None:
                self._password_hasher.verify_password(password, self._dummy_password_hash)
                raise ValueError("invalid login credentials")
            password_matches = self._password_hasher.verify_password(
                password, record.password_hash
            )
            if not password_matches or record.account_status != "ACTIVE":
                raise ValueError("invalid login credentials")
            return {"customer_id": record.customer_id}
        except Exception:
            raise CustomerAuthenticationError("Invalid email or password.") from None
