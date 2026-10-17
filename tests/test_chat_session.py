"""Offline tests for per-session conversation history and event deduplication."""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from customer_app import (
    _consume_chat_event_id,
    _queue_chat_event,
    _start_new_policy_conversation,
)
from src.customer_app_controller import CustomerAppController
from src.chat_session import append_message, ensure_conversation, run_event_once


class ChatSessionTests(unittest.TestCase):
    def test_history_survives_rerun_style_reinitialization(self) -> None:
        state: dict[str, object] = {}
        history = ensure_conversation(state, "policy_messages", "Welcome.")
        append_message(state, "policy_messages", "user", "A synthetic question?")
        append_message(state, "policy_messages", "assistant", "A grounded response.")

        history_after_rerun = ensure_conversation(
            state, "policy_messages", "A different welcome must not replace history."
        )

        self.assertIs(history_after_rerun, history)
        self.assertEqual(len(history_after_rerun), 3)
        self.assertEqual(history_after_rerun[1]["content"], "A synthetic question?")

    def test_complaint_and_policy_histories_are_separate(self) -> None:
        state: dict[str, object] = {}
        complaint = ensure_conversation(state, "complaint_messages", "Complaint welcome.")
        faq = ensure_conversation(state, "policy_messages", "Policy welcome.")

        append_message(state, "complaint_messages", "user", "Synthetic complaint.")

        self.assertEqual(len(complaint), 2)
        self.assertEqual(len(faq), 1)
        self.assertNotIn("Synthetic complaint.", repr(faq))

    def test_replayed_event_does_not_call_assistant_twice(self) -> None:
        state: dict[str, object] = {}
        answer = Mock(return_value="Grounded synthetic answer.")

        first = run_event_once(state, "policy", "event-42", answer)
        replay = run_event_once(state, "policy", "event-42", answer)

        self.assertEqual(first, (True, "Grounded synthetic answer."))
        self.assertEqual(replay, (False, None))
        answer.assert_called_once_with()

    def test_streamlit_submit_callback_event_id_is_consumed_and_deduplicated(self) -> None:
        import customer_app

        state: dict[str, object] = {}
        previous_streamlit = customer_app.st
        customer_app.st = SimpleNamespace(session_state=state)
        action = Mock(return_value="one response")
        try:
            _queue_chat_event("policy_chat")
            event_id = _consume_chat_event_id(state, "policy_chat")
            first = run_event_once(state, "policy_chat", event_id, action)
            replay = run_event_once(state, "policy_chat", event_id, action)
        finally:
            customer_app.st = previous_streamlit

        self.assertEqual(first, (True, "one response"))
        self.assertEqual(replay, (False, None))
        self.assertNotIn("_policy_chat_pending_event_id", state)
        action.assert_called_once_with()

    def test_distinct_submissions_are_each_processed_once(self) -> None:
        state: dict[str, object] = {}
        answer = Mock(side_effect=("First answer.", "Second answer."))

        first = run_event_once(state, "policy", "event-1", answer)
        second = run_event_once(state, "policy", "event-2", answer)

        self.assertEqual(first[1], "First answer.")
        self.assertEqual(second[1], "Second answer.")
        self.assertEqual(answer.call_count, 2)

    def test_customer_app_uses_native_chat_and_incremental_rendering(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "customer_app.py").read_text(
            encoding="utf-8"
        )
        for required in (
            "streamlit_module.chat_message(",
            "streamlit_module.chat_input(",
            "streamlit_module.session_state",
            "streamlit_module.spinner(",
            "streamlit_module.write_stream(",
            "_CONTROLLER_CACHE_VERSION = 2",
            "def build_controller(cache_version:",
            "build_controller(_CONTROLLER_CACHE_VERSION)",
            "st.tabs(",
        ):
            with self.subTest(required=required):
                self.assertIn(required, source)

    def test_new_policy_conversation_clears_only_faq_history_and_context(self) -> None:
        state: dict[str, object] = {
            "policy_chat_messages": [
                {"role": "assistant", "content": "Old welcome.", "sources": []},
                {"role": "user", "content": "Old question.", "sources": []},
            ],
            "complaint_chat_messages": [
                {"role": "assistant", "content": "Complaint welcome.", "sources": []},
                {"role": "user", "content": "Synthetic complaint.", "sources": []},
            ],
            "_policy_chat_processed_chat_events": ["old-faq-event"],
            "_complaint_chat_processed_chat_events": ["keep-complaint-event"],
        }
        controller = CustomerAppController(policy_assistant=Mock())
        controller._previous_policy_answer = "Previously cited answer."

        _start_new_policy_conversation(SimpleNamespace(session_state=state), controller)

        self.assertEqual(len(state["policy_chat_messages"]), 1)
        self.assertEqual(state["policy_chat_messages"][0]["role"], "assistant")
        self.assertEqual(
            state["complaint_chat_messages"][1]["content"], "Synthetic complaint."
        )
        self.assertNotIn("_policy_chat_processed_chat_events", state)
        self.assertEqual(
            state["_complaint_chat_processed_chat_events"], ["keep-complaint-event"]
        )
        self.assertIsNone(controller._previous_policy_answer)

    def test_controller_methods_called_by_customer_app_exist(self) -> None:
        source = (Path(__file__).resolve().parents[1] / "customer_app.py").read_text(
            encoding="utf-8"
        )
        required_methods = (
            "submit_complaint",
            "answer_policy_question",
            "clear_policy_context",
        )
        for method in required_methods:
            with self.subTest(method=method):
                self.assertTrue(callable(getattr(CustomerAppController, method, None)))
                self.assertIn(f"controller.{method}(", source)


if __name__ == "__main__":
    unittest.main()
