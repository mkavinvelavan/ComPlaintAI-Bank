"""Public Streamlit complaint chatbot and separate policy FAQ conversation."""

from __future__ import annotations

import re
from pathlib import Path
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

try:
    import streamlit as st
except ImportError:  # Keep controller tests usable if the optional UI package is absent.
    st = None  # type: ignore[assignment]

from src.chat_session import append_message, ensure_conversation, run_event_once
from src.customer_app_controller import CustomerAppController
from src.sensitive_query_guard import is_sensitive_authentication_request

_COMPLAINT_HISTORY = "complaint_chat_messages"
_POLICY_HISTORY = "policy_chat_messages"
_PENDING_COMPLAINT = "pending_guest_complaint_text"
# Changing this key refreshes a previously cached controller after its API changes.
_CONTROLLER_CACHE_VERSION = 2

_COMPLAINT_WELCOME = (
    "Hello, I’m the ComPlaintAI-Bank support assistant. I’m sorry if you’ve had a "
    "frustrating banking experience. Tell me what happened in your own words, and "
    "I’ll guide you through submitting it."
)
_POLICY_WELCOME = (
    "Hi! I can help with general questions about the available policies. What would "
    "you like to know?"
)


if st is not None:
    @st.cache_resource(scope="session", show_spinner=False)
    def build_controller(cache_version: int = _CONTROLLER_CACHE_VERSION) -> CustomerAppController:
        return CustomerAppController()
else:
    def build_controller(cache_version: int = _CONTROLLER_CACHE_VERSION) -> CustomerAppController:
        return CustomerAppController()


def _stream_text(text: str):
    """Progressively render completed text without claiming model token streaming."""
    yield from re.findall(r"\S+\s*", text)


def _queue_chat_event(namespace: str) -> None:
    """Assign one opaque event ID in Streamlit's submit callback."""
    if st is not None:
        st.session_state[f"_{namespace}_pending_event_id"] = uuid4().hex


def _consume_chat_event_id(session_state, namespace: str) -> str:
    """Reuse the submit callback's ID, with a fallback for non-widget actions."""
    key = f"_{namespace}_pending_event_id"
    event_id = session_state.pop(key, None)
    return event_id if isinstance(event_id, str) and event_id else uuid4().hex


def _render_history(streamlit_module, history: list[dict]) -> None:
    for message in history:
        with streamlit_module.chat_message(message["role"]):
            streamlit_module.markdown(message["content"])
            for source in message.get("sources", []):
                streamlit_module.caption(
                    f"{source['policy_id']} | {source['source_file']} | page {source['page']}"
                )


def _reset_complaint_conversation(streamlit_module) -> None:
    streamlit_module.session_state[_COMPLAINT_HISTORY] = [
        {"role": "assistant", "content": _COMPLAINT_WELCOME, "sources": []}
    ]
    streamlit_module.session_state.pop(_PENDING_COMPLAINT, None)
    streamlit_module.session_state.pop("complaint_contact_email", None)
    streamlit_module.session_state.pop("complaint_contact_phone", None)
    streamlit_module.session_state.pop("_complaint_chat_processed_chat_events", None)
    streamlit_module.session_state.pop("_complaint_registration_processed_chat_events", None)
    streamlit_module.session_state.pop("_complaint_chat_pending_event_id", None)
    streamlit_module.session_state.pop("_complaint_registration_pending_event_id", None)


def _start_or_continue_complaint(streamlit_module, text: str) -> None:
    event_id = _consume_chat_event_id(streamlit_module.session_state, "complaint_chat")

    def accept_message() -> str:
        append_message(streamlit_module.session_state, _COMPLAINT_HISTORY, "user", text)
        previous = streamlit_module.session_state.get(_PENDING_COMPLAINT)
        if isinstance(previous, str) and previous:
            streamlit_module.session_state[_PENDING_COMPLAINT] = previous + "\n" + text
            return (
                "Thanks, I’ve added that detail. To submit this complaint, please provide "
                "an email address, a phone number, or both, then choose **Submit complaint**."
            )
        streamlit_module.session_state[_PENDING_COMPLAINT] = text
        return (
            "I’m sorry you’re dealing with this. I can help submit your description for review. "
            "To register it, please share an email address, a phone number, or both so the "
            "project can contact you. Contact details do not verify identity. Choose **Submit "
            "complaint** when you’re ready; sending a chat message alone does not register it."
        )

    accepted, assistant_message = run_event_once(
        streamlit_module.session_state,
        "complaint_chat",
        event_id,
        accept_message,
    )
    if accepted:
        append_message(
            streamlit_module.session_state,
            _COMPLAINT_HISTORY,
            "assistant",
            assistant_message,
        )


def _submit_pending_complaint(streamlit_module, controller: CustomerAppController) -> None:
    complaint_text = streamlit_module.session_state.get(_PENDING_COMPLAINT)
    if not isinstance(complaint_text, str) or not complaint_text.strip():
        return

    email = streamlit_module.session_state.get("complaint_contact_email", "")
    phone = streamlit_module.session_state.get("complaint_contact_phone", "")
    event_id = _consume_chat_event_id(streamlit_module.session_state, "complaint_registration")
    with streamlit_module.chat_message("assistant"):
        with streamlit_module.spinner("Reviewing your details and processing the submission…"):
            accepted, result = run_event_once(
                streamlit_module.session_state,
                "complaint_registration",
                event_id,
                lambda: controller.submit_complaint(complaint_text, email, phone),
            )
        if not accepted or not isinstance(result, dict):
            return
        status = result.get("status")
        if status == "success":
            response = (
                "Thank you for explaining what happened. Your complaint has been recorded. "
                f"Your case reference is **{result['case_reference']}**."
            )
            streamlit_module.write_stream(_stream_text(response))
            streamlit_module.session_state.pop(_PENDING_COMPLAINT, None)
        elif status == "validation_error":
            response = result["message"]
            streamlit_module.write_stream(_stream_text(response))
        else:
            response = result["message"]
            streamlit_module.write_stream(_stream_text(response))
            # A failed commit can have an uncertain outcome. Avoid encouraging
            # automatic retries that could create a duplicate registration.
            streamlit_module.session_state.pop(_PENDING_COMPLAINT, None)

    append_message(
        streamlit_module.session_state,
        _COMPLAINT_HISTORY,
        "assistant",
        response,
    )


def _render_complaint_chat(streamlit_module, controller: CustomerAppController) -> None:
    streamlit_module.subheader("File a complaint")
    streamlit_module.write(
        "Describe the issue in your own words. We’ll ask for a follow-up contact method "
        "before you choose to submit it."
    )
    streamlit_module.caption(
        "Please do not share passwords, PINs, CVVs, one-time codes, or other sign-in secrets."
    )
    if streamlit_module.button("Start a new complaint conversation", key="new_complaint_chat"):
        _reset_complaint_conversation(streamlit_module)
        streamlit_module.rerun()
    streamlit_module.caption(
        "This clears the chat from this browser session only. Any complaint already confirmed "
        "as registered remains in the database."
    )

    history = ensure_conversation(streamlit_module.session_state, _COMPLAINT_HISTORY, _COMPLAINT_WELCOME)
    _render_history(streamlit_module, history)

    if streamlit_module.session_state.get(_PENDING_COMPLAINT):
        streamlit_module.info(
            "To register this complaint, provide at least one contact method. You may enter "
            "an email address, a phone number, or both. These details are for follow-up only."
        )
        with streamlit_module.form("guest_complaint_contact_form", clear_on_submit=True):
            streamlit_module.text_input(
                "Email address (optional)",
                max_chars=254,
                placeholder="name@example.test",
                key="complaint_contact_email",
            )
            streamlit_module.text_input(
                "Phone number (optional)",
                max_chars=32,
                placeholder="+971 50 123 4567",
                key="complaint_contact_phone",
            )
            submitted = streamlit_module.form_submit_button(
                "Submit complaint",
                use_container_width=True,
                on_click=_queue_chat_event,
                args=("complaint_registration",),
            )
        if submitted:
            _submit_pending_complaint(streamlit_module, controller)

    submitted_text = streamlit_module.chat_input(
        "Describe your banking issue…",
        key="complaint_chat_input",
        max_chars=50_000,
        on_submit=_queue_chat_event,
        args=("complaint_chat",),
    )
    if submitted_text:
        _start_or_continue_complaint(streamlit_module, str(submitted_text))
        streamlit_module.rerun()


def _start_new_policy_conversation(streamlit_module, controller: CustomerAppController) -> None:
    """Reset FAQ history/context only, retaining complaint state and services."""
    streamlit_module.session_state[_POLICY_HISTORY] = [
        {"role": "assistant", "content": _POLICY_WELCOME, "sources": []}
    ]
    streamlit_module.session_state.pop("_policy_chat_processed_chat_events", None)
    streamlit_module.session_state.pop("_policy_chat_pending_event_id", None)
    controller.clear_policy_context()


def _render_policy_chat(streamlit_module, controller: CustomerAppController) -> None:
    streamlit_module.subheader("Policy FAQ")
    streamlit_module.caption(
        "Ask about the generic banking policies. This chat is separate from complaint intake. "
        "When Groq generation is enabled, your policy question and retrieved policy excerpts "
        "are sent to the configured external service. Please do not enter complaint or contact details."
    )
    if streamlit_module.button("Start a new policy conversation", key="new_policy_chat"):
        _start_new_policy_conversation(streamlit_module, controller)
        streamlit_module.rerun()

    history = ensure_conversation(streamlit_module.session_state, _POLICY_HISTORY, _POLICY_WELCOME)
    if len(history) == 1:
        suggestion_columns = streamlit_module.columns(2)
        suggestions = (
            "How do I register a complaint?",
            "What should I do about an unauthorised transaction?",
        )
        selected_suggestion = None
        for column, suggestion, index in zip(suggestion_columns, suggestions, range(2)):
            with column:
                if streamlit_module.button(suggestion, key=f"policy_suggestion_{index}"):
                    selected_suggestion = suggestion
    else:
        selected_suggestion = None

    _render_history(streamlit_module, history)
    response_area = streamlit_module.container()
    question_input = streamlit_module.chat_input(
        "Ask a policy question…",
        key="policy_chat_input",
        max_chars=2_000,
        on_submit=_queue_chat_event,
        args=("policy_chat",),
    )
    question = selected_suggestion or (str(question_input) if question_input else None)
    if question:
        event_id = (
            uuid4().hex
            if selected_suggestion
            else _consume_chat_event_id(streamlit_module.session_state, "policy_chat")
        )
        display_question = (
            "[Sensitive request omitted for safety.]"
            if is_sensitive_authentication_request(question)
            else question
        )

        def answer_once() -> dict:
            append_message(streamlit_module.session_state, _POLICY_HISTORY, "user", display_question)
            with streamlit_module.chat_message("user"):
                streamlit_module.markdown(display_question)
            with streamlit_module.chat_message("assistant"):
                with streamlit_module.spinner("Checking the policy information…"):
                    result = controller.answer_policy_question(question)
                response = result.get("answer", result.get("message", "I couldn’t prepare an answer."))
                streamlit_module.write_stream(_stream_text(response))
                for source in result.get("sources", []):
                    streamlit_module.caption(
                        f"{source['policy_id']} | {source['source_file']} | page {source['page']}"
                    )
            append_message(
                streamlit_module.session_state,
                _POLICY_HISTORY,
                "assistant",
                response,
                sources=result.get("sources", []),
            )
            return result

        with response_area:
            run_event_once(
                streamlit_module.session_state,
                "policy_chat",
                event_id,
                answer_once,
            )


def main() -> None:
    if st is None:
        raise SystemExit(
            "Streamlit is not installed. Install the declared dependency, then run: "
            ".venv\\Scripts\\python.exe -m streamlit run customer_app.py"
        )

    st.set_page_config(page_title="ComPlaintAI-Bank", page_icon="💬", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 900px; padding-top: 1.6rem;}
        [data-testid="stChatMessage"] {border-radius: 12px; padding: .5rem .75rem;}
        [data-testid="stChatInput"] {max-width: 900px; margin: auto;}
        h1, h2, h3 {color: #15345b;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.title("ComPlaintAI-Bank")
    st.caption("A calm, conversational place to submit a complaint or ask about policy.")
    st.caption(
        "University project demonstration. Complaints are recorded only after database "
        "commit confirmation; an AI prediction does not verify an allegation."
    )
    complaint_tab, policy_tab = st.tabs(["File a Complaint", "Policy FAQ"])
    controller = build_controller(_CONTROLLER_CACHE_VERSION)
    with complaint_tab:
        _render_complaint_chat(st, controller)
    with policy_tab:
        _render_policy_chat(st, controller)
if __name__ == "__main__":
    main()
