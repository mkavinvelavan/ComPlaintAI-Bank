"""Separate Streamlit administrator dashboard for complaint operations."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env", override=False)

try:
    import streamlit as st
except ImportError:  # Allows offline service tests without the optional UI.
    st = None

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

_AUTH_KEY = "_admin_authenticated"


def _logout() -> None:
    if st is not None:
        clear_admin_session(st.session_state)


def _build_service() -> AdminComplaintService:
    """Build a connection-free repository wrapper; each operation authorizes."""
    if st is None:
        raise RuntimeError("Streamlit is required to run the administrator dashboard.")
    repository = AdminComplaintRepository()
    return AdminComplaintService(
        repository,
        is_authenticated=lambda: st.session_state.get(_AUTH_KEY) is True,
    )


def _contact_value_for_display(value: object) -> str:
    """Format an authorised contact field without logging or echoing elsewhere."""
    return value if isinstance(value, str) and value.strip() else "Not provided"


def _login(authenticator: AdminAuthenticator) -> None:
    st.title("ComPlaintAI-Bank Admin")
    st.write("Sign in to review and manage complaint records.")
    if not authenticator.is_configured:
        st.error(
            "Administrator access is unavailable. Configure the server-side username and a "
            "strong password of at least 16 characters."
        )
        return
    with st.form("admin_login_form"):
        username = st.text_input("Username", key="admin_login_username")
        password = st.text_input("Password", type="password", key="admin_login_password")
        submitted = st.form_submit_button("Sign in", type="primary")
    if submitted:
        if authenticator.authenticate(username, password):
            st.session_state[_AUTH_KEY] = True
            st.rerun()
        else:
            st.error("The username or password is incorrect.")
    st.caption("Administrator credentials are required. There is no public admin registration.")


def _show_overview(service: AdminComplaintService) -> None:
    st.header("Overview")
    overview = service.get_overview()
    st.metric("Registered complaints", overview["total"])

    status_counts = overview["by_status"]
    priority_counts = overview["by_priority"]
    status_columns = st.columns(max(1, len(ALLOWED_STATUSES)))
    for column, status in zip(status_columns, ALLOWED_STATUSES):
        column.metric(status.replace("_", " ").title(), status_counts.get(status, 0))

    priority_columns = st.columns(len(ALLOWED_PRIORITIES) + 1)
    for column, priority in zip(priority_columns, (*ALLOWED_PRIORITIES, "UNASSIGNED")):
        column.metric(
            "No priority" if priority == "UNASSIGNED" else f"{priority.title()} priority",
            priority_counts.get(priority, 0),
        )

    st.subheader("Recently submitted")
    if overview["recent"]:
        st.dataframe(overview["recent"], use_container_width=True, hide_index=True)
    else:
        st.info("No complaints are available.")


def _show_complaints(service: AdminComplaintService) -> None:
    st.header("Complaint management")
    with st.container(border=True):
        search_column, status_column, priority_column = st.columns((2, 1, 1))
        search = search_column.text_input(
            "Search case reference",
            key="admin_case_reference_search",
            max_chars=64,
        )
        status_options = ("All statuses", *ALLOWED_STATUSES)
        selected_status = status_column.selectbox("Status", status_options)
        priority_options = ("All priorities", *ALLOWED_PRIORITIES)
        selected_priority = priority_column.selectbox("Priority", priority_options)

    rows = service.list_complaints(
        case_reference=search or None,
        status=None if selected_status == "All statuses" else selected_status,
        priority=None if selected_priority == "All priorities" else selected_priority,
    )
    st.dataframe(rows, use_container_width=True, hide_index=True)
    if not rows:
        st.info("No complaints match these filters.")
        return

    cases = [row["case_reference"] for row in rows]
    selected_case = st.selectbox("Open complaint details", cases, key="admin_selected_case")
    detail = service.get_complaint(selected_case)
    if detail is None:
        st.info("That complaint is no longer available in the register.")
        return

    st.subheader(f"Complaint {detail['case_reference']}")
    metadata_left, metadata_right = st.columns(2)
    metadata_left.write(f"**Status:** {detail['status']}")
    metadata_left.write(f"**Predicted intent:** {detail['predicted_intent'] or 'Not available'}")
    metadata_left.write(f"**Priority:** {detail['priority'] or 'Not assigned'}")
    metadata_right.write(
        f"**Department suggestion:** {detail['department_suggestion'] or 'Not assigned'}"
    )
    metadata_right.write(f"**Created:** {detail['created_at']}")
    metadata_right.write(
        "**Credential disclosure detected:** "
        + ("Yes" if detail["credential_disclosure_detected"] else "No")
    )
    with st.expander("Complaint and contact details"):
        st.write(detail["complaint_text"])
        st.write(
            f"**Contact email:** {_contact_value_for_display(detail.get('contact_email'))}"
        )
        st.write(
            f"**Contact phone:** {_contact_value_for_display(detail.get('contact_phone'))}"
        )

    with st.form("admin_status_update_form"):
        new_status = st.selectbox(
            "Update status",
            ALLOWED_STATUSES,
            index=ALLOWED_STATUSES.index(detail["status"]),
        )
        update = st.form_submit_button("Save status", type="primary")
    if update:
        updated = service.update_status(detail["case_reference"], new_status)
        if updated is None:
            st.error("The status was not changed because the complaint could not be found.")
        else:
            st.success(f"Status updated to {updated['status']}.")
            st.rerun()


def main() -> None:
    if st is None:
        raise SystemExit(
            "Streamlit is not installed. Run: .venv\\Scripts\\python.exe -m streamlit run admin_app.py"
        )
    st.set_page_config(page_title="ComPlaintAI-Bank Admin", page_icon="🗂️", layout="wide")
    st.markdown(
        """
        <style>
        .block-container {max-width: 1180px; padding-top: 1.6rem;}
        h1, h2, h3 {color: #15345b;}
        </style>
        """,
        unsafe_allow_html=True,
    )
    authenticator = AdminAuthenticator()
    if st.session_state.get(_AUTH_KEY) is not True:
        _login(authenticator)
        return

    st.sidebar.title("ComPlaintAI-Bank Admin")
    st.sidebar.success("Administrator session active")
    st.sidebar.button("Log out", on_click=_logout, use_container_width=True)
    section = st.sidebar.radio("Dashboard", ("Overview", "Complaints"))

    try:
        service = _build_service()
    except AdminRepositoryError:
        st.error("The complaint database is not configured or the PostgreSQL driver is unavailable.")
        return

    try:
        if section == "Overview":
            _show_overview(service)
        else:
            _show_complaints(service)
    except AdminAuthorizationError:
        clear_admin_session(st.session_state)
        st.error("Your administrator session is no longer authorized. Please sign in again.")
    except (AdminOperationError, AdminRepositoryError):
        st.error("The requested complaint information could not be loaded or updated.")
    except ValueError as exc:
        # Only fixed validation messages from the service/repository are displayed.
        st.error(str(exc))


if __name__ == "__main__":
    main()
