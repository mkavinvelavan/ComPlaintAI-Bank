"""Resolve local and Streamlit Community Cloud runtime configuration safely."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env", override=False)


def _read_streamlit_secret(name: str) -> Any:
    """Read one top-level Streamlit secret without exposing secret-store errors."""
    try:
        import streamlit as st

        return st.secrets.get(name)
    except Exception:
        # Outside Streamlit, secrets may be unavailable or unconfigured.
        return None


def get_setting(name: str, default: str | None = None) -> str | None:
    """Return an environment value first, then a Streamlit secret, then default.

    The ignored project-root ``.env`` is loaded without overriding values set by
    the process environment. Blank values are treated as unconfigured.
    """
    if not isinstance(name, str) or not name or not name.replace("_", "").isalnum():
        raise ValueError("Configuration setting name is invalid.")

    environment_value = os.getenv(name)
    if isinstance(environment_value, str) and environment_value.strip():
        return environment_value.strip()

    secret_value = _read_streamlit_secret(name)
    if isinstance(secret_value, str) and secret_value.strip():
        return secret_value.strip()
    return default
