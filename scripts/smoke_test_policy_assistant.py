"""Make one live, read-only policy assistant request against the local index."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policy_assistant import PolicyAssistant  # noqa: E402
from src.policy_rag import (  # noqa: E402
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    PolicyRetriever,
)

POLICY_QUESTION = "What is the process for registering a customer complaint?"


def main() -> int:
    """Run one generic question without invoking any index mutation operation."""
    if not os.getenv("GROQ_API_KEY", "").strip():
        print("Error: GROQ_API_KEY is not set. Configure it in the environment and retry.", file=sys.stderr)
        return 1

    database_file = DEFAULT_PERSIST_DIRECTORY / "chroma.sqlite3"
    if not database_file.is_file():
        print(
            "Error: the existing ChromaDB index file was not found at the configured data path. "
            "No index was created; run the separate ingestion step first.",
            file=sys.stderr,
        )
        return 1

    try:
        import chromadb
    except ImportError:
        print("Error: ChromaDB is unavailable in this Python environment.", file=sys.stderr)
        return 1

    try:
        client = chromadb.PersistentClient(path=str(DEFAULT_PERSIST_DIRECTORY))
        # get_collection is intentionally used instead of get_or_create_collection.
        collection = client.get_collection(name=DEFAULT_COLLECTION_NAME)
    except Exception as exc:
        print(
            "Error: the existing policy collection is missing or could not be opened "
            f"({type(exc).__name__}). No collection was created.",
            file=sys.stderr,
        )
        return 1

    retriever = PolicyRetriever(
        persist_directory=DEFAULT_PERSIST_DIRECTORY,
        client=client,
        local_files_only=True,
    )
    # Seed the retriever with the collection that was verified above so its
    # lazy property cannot fall back to get_or_create_collection.
    retriever._collection = collection
    assistant = PolicyAssistant(retriever=retriever)

    try:
        result = assistant.answer_question(POLICY_QUESTION, top_k=3)
    except RuntimeError as exc:
        # PolicyAssistant emits safe configuration/API errors without request
        # text, retrieved content, or credential values.
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: policy assistant request failed ({type(exc).__name__}).", file=sys.stderr)
        return 1

    print("Question:", POLICY_QUESTION)
    print("Answer:")
    print(result.get("answer", ""))
    print("Sources:")
    sources = result.get("sources", [])
    if not sources:
        print("  No supporting policy sources were returned.")
    for source in sources:
        print(
            "  - "
            f"{source.get('source_file', 'Unknown file')} | "
            f"page {source.get('page', 'unknown')} | "
            f"{source.get('policy_id', 'Unknown policy ID')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
