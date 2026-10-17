"""Small Streamlit-session chat-history helpers with per-event deduplication."""

from __future__ import annotations

from collections.abc import Callable, MutableMapping
from typing import Any

_MAX_REMEMBERED_EVENTS = 128


def ensure_conversation(
    state: MutableMapping[str, Any], key: str, welcome: str
) -> list[dict[str, Any]]:
    """Initialize a conversation once and preserve it across app reruns."""
    history = state.get(key)
    if isinstance(history, list):
        return history
    history = [{"role": "assistant", "content": welcome, "sources": []}]
    state[key] = history
    return history


def append_message(
    state: MutableMapping[str, Any],
    key: str,
    role: str,
    content: str,
    *,
    sources: list[dict[str, Any]] | None = None,
) -> None:
    """Append one user/assistant turn without storing service objects."""
    if role not in {"user", "assistant"}:
        raise ValueError("role must be user or assistant.")
    history = state.get(key)
    if not isinstance(history, list):
        history = []
        state[key] = history
    history.append(
        {"role": role, "content": content, "sources": list(sources or [])}
    )


def run_event_once(
    state: MutableMapping[str, Any],
    namespace: str,
    event_id: str,
    action: Callable[[], Any],
) -> tuple[bool, Any]:
    """Run one submitted UI event at most once, including across reruns.

    Streamlit's chat input returns text on its submit rerun only; this additional
    bounded event ledger guards against duplicate dispatch if a rerun replays an
    already handled event. Event IDs are opaque and contain no user content.
    """
    if not isinstance(event_id, str) or not event_id:
        raise ValueError("event_id must be a non-empty string.")
    key = f"_{namespace}_processed_chat_events"
    events = state.get(key)
    if not isinstance(events, list):
        events = []
    if event_id in events:
        state[key] = events
        return False, None
    events.append(event_id)
    state[key] = events[-_MAX_REMEMBERED_EVENTS:]
    return True, action()
