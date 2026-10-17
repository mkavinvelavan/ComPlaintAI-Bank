"""Lazy, idempotent policy-index setup for the public application."""

from __future__ import annotations

import json
from pathlib import Path
from threading import Lock

from src.policy_rag import (
    DEFAULT_COLLECTION_NAME,
    DEFAULT_PERSIST_DIRECTORY,
    PolicyRetriever,
)
from src.runtime_config import PROJECT_ROOT, get_setting

_EXPECTED_POLICY_IDS = {f"CPB-POL-{number:03d}" for number in range(1, 9)}
_EXPECTED_FILES = {
    path.name for path in (PROJECT_ROOT / "knowledge_base" / "policies").glob("*.pdf")
}
_INITIALIZATION_LOCK = Lock()
_RETRIEVER: PolicyRetriever | None = None


def _valid_ingestion_manifest(retriever: PolicyRetriever) -> bool:
    manifest_path = (
        retriever.persist_directory
        / f"{retriever.collection_name}_ingestion.json"
    )
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(manifest, dict):
        return False
    chunk_count = manifest.get("chunks_created")
    indexed_files = manifest.get("indexed_files")
    valid = (
        manifest.get("pdfs_read") == 8
        and isinstance(indexed_files, list)
        and all(isinstance(filename, str) for filename in indexed_files)
        and set(indexed_files) == _EXPECTED_FILES
        and isinstance(chunk_count, int)
        and not isinstance(chunk_count, bool)
        and chunk_count > 0
        and not manifest.get("issues")
    )
    if not valid:
        return False
    try:
        return retriever.collection.count() >= chunk_count
    except Exception:
        return False


def get_policy_retriever() -> PolicyRetriever:
    """Return one process-wide retriever and initialize only an incomplete index.

    Ingestion is non-destructive and uses deterministic chunk IDs with upsert.
    A populated, validated local index is left untouched. A new Cloud filesystem
    is initialized from the committed policy PDFs; repeated starts do not add
    duplicate chunks. The lock serializes initialization across app sessions.
    """
    global _RETRIEVER
    with _INITIALIZATION_LOCK:
        if _RETRIEVER is not None:
            return _RETRIEVER

        configured_directory = get_setting("CHROMA_PERSIST_DIRECTORY")
        persist_directory = (
            Path(configured_directory).expanduser()
            if configured_directory
            else DEFAULT_PERSIST_DIRECTORY
        )
        retriever = PolicyRetriever(
            persist_directory=persist_directory,
            collection_name=DEFAULT_COLLECTION_NAME,
            local_files_only=False,
        )
        if not _valid_ingestion_manifest(retriever):
            report = retriever.ingest_directory(
                PROJECT_ROOT / "knowledge_base" / "policies",
                rebuild=False,
            )
            if (
                report.issues
                or report.pdfs_read != 8
                or set(report.indexed_files) != _EXPECTED_FILES
                or set(report.indexed_policy_ids) != _EXPECTED_POLICY_IDS
                or report.chunks_created <= 0
            ):
                retriever.close()
                raise RuntimeError(
                    "The policy index could not be initialized from all eight approved PDFs."
                )
        _RETRIEVER = retriever
        return retriever
