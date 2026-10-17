"""Local PDF policy ingestion and ChromaDB retrieval (no answer generation)."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol, Sequence

from pypdf import PdfReader


DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_COLLECTION_NAME = "complaintai_bank_policies"
DEFAULT_PERSIST_DIRECTORY = Path(__file__).resolve().parents[1] / "data" / "chroma_db"
_POLICY_ID_PATTERN = re.compile(r"\bCPB-POL-\d{3}\b", re.IGNORECASE)


class TextEmbedder(Protocol):
    """Minimal embedding interface, injectable for offline tests."""

    def encode(self, texts: Sequence[str]) -> Any: ...


@dataclass(frozen=True)
class PolicyChunk:
    """One non-empty text segment and its source metadata."""

    chunk_id: str
    text: str
    metadata: dict[str, str | int]


@dataclass
class IngestionReport:
    """Counts and document-level issues from a policy ingestion run."""

    pdfs_read: int = 0
    pages_processed: int = 0
    chunks_created: int = 0
    indexed_files: list[str] = field(default_factory=list)
    indexed_policy_ids: list[str] = field(default_factory=list)
    issues: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def recursive_character_chunks(
    text: str, chunk_size: int = 800, chunk_overlap: int = 120
) -> list[str]:
    """Split text near paragraph/sentence/word boundaries with character overlap.

    Empty input and whitespace-only segments are omitted. Boundaries prefer
    blank lines, lines, sentence spaces, then word spaces; long unbroken text is
    split at the target character limit.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string.")
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer.")
    if (
        isinstance(chunk_overlap, bool)
        or not isinstance(chunk_overlap, int)
        or chunk_overlap < 0
        or chunk_overlap >= chunk_size
    ):
        raise ValueError("chunk_overlap must be an integer from 0 up to chunk_size - 1.")

    source = text.strip()
    if not source:
        return []
    separators = ("\n\n", "\n", ". ", " ")
    result: list[str] = []
    start = 0
    while start < len(source):
        end = min(start + chunk_size, len(source))
        if end < len(source):
            lower_bound = start + max(1, chunk_size // 2)
            candidates = [(source.rfind(sep, lower_bound, end), sep) for sep in separators]
            candidates = [(position, sep) for position, sep in candidates if position >= lower_bound]
            if candidates:
                position, separator = candidates[0]
                end = position + len(separator)
        piece = source[start:end].strip()
        if piece:
            result.append(piece)
        if end >= len(source):
            break
        next_start = max(start + 1, end - chunk_overlap)
        # Start at a clean boundary when possible while keeping overlap close to target.
        while next_start < end and source[next_start].isspace():
            next_start += 1
        start = next_start
    return result


def build_policy_chunks(
    filename: str,
    page_number: int,
    text: str,
    *,
    policy_id: str | None = None,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
) -> list[PolicyChunk]:
    """Chunk one extracted PDF page and preserve filename, ID, page, and section."""
    policy_match = _POLICY_ID_PATTERN.search(text)
    resolved_policy_id = (
        policy_id.upper()
        if isinstance(policy_id, str) and _POLICY_ID_PATTERN.fullmatch(policy_id)
        else policy_match.group(0).upper() if policy_match else "UNKNOWN"
    )
    pieces = recursive_character_chunks(text, chunk_size, chunk_overlap)
    chunks: list[PolicyChunk] = []
    for index, piece in enumerate(pieces):
        section_match = re.search(r"(?m)^\s*((?:\d+(?:\.\d+)*\.?\s+)[^\n]{1,140})", piece)
        section = section_match.group(1).strip() if section_match else ""
        digest = hashlib.sha256(
            f"{filename}\0{page_number}\0{index}\0{piece}".encode("utf-8")
        ).hexdigest()
        chunks.append(
            PolicyChunk(
                chunk_id=f"chunk-{digest[:32]}",
                text=piece,
                metadata={
                    "source": filename,
                    "filename": filename,
                    "policy_id": resolved_policy_id,
                    "page_number": page_number,
                    "chunk_index": index,
                    "section": section,
                    "embedding_model": DEFAULT_EMBEDDING_MODEL,
                },
            )
        )
    return chunks


class _SentenceTransformerEmbedder:
    """Lazy local SentenceTransformers wrapper used for documents and queries."""

    def __init__(self, model_name: str, *, local_files_only: bool) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError(
                "SentenceTransformers is required; install the declared project dependencies."
            ) from exc
        self.model_name = model_name
        try:
            self.model = SentenceTransformer(model_name, local_files_only=local_files_only)
        except Exception:
            raise RuntimeError(
                f"Could not load embedding model {model_name!r} locally or from its model source."
            ) from None

    def encode(self, texts: Sequence[str]) -> Any:
        return self.model.encode(list(texts), convert_to_numpy=True, normalize_embeddings=True)


class PolicyRetriever:
    """Persist policy chunks in ChromaDB and retrieve evidence by similarity."""

    def __init__(
        self,
        persist_directory: str | Path = DEFAULT_PERSIST_DIRECTORY,
        collection_name: str = DEFAULT_COLLECTION_NAME,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
        *,
        embedder: TextEmbedder | None = None,
        client: Any | None = None,
        local_files_only: bool = True,
    ) -> None:
        if not isinstance(collection_name, str) or not collection_name.strip():
            raise ValueError("collection_name must be a non-empty string.")
        self.persist_directory = Path(persist_directory)
        self.collection_name = collection_name
        self.embedding_model_name = embedding_model_name
        self.local_files_only = local_files_only
        self._embedder = embedder
        self._client = client
        self._collection: Any | None = None

    @property
    def collection(self) -> Any:
        """Open/create the persistent cosine-similarity collection lazily."""
        if self._collection is None:
            try:
                import chromadb
            except ImportError as exc:
                raise RuntimeError("ChromaDB is required; install the declared project dependencies.") from exc
            self.persist_directory.mkdir(parents=True, exist_ok=True)
            self._client = self._client or chromadb.PersistentClient(path=str(self.persist_directory))
            self._collection = self._client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model_name},
            )
        return self._collection

    @property
    def embedder(self) -> TextEmbedder:
        if self._embedder is None:
            self._embedder = _SentenceTransformerEmbedder(
                self.embedding_model_name, local_files_only=self.local_files_only
            )
        return self._embedder

    def close(self) -> None:
        """Release this retriever's Chroma client resources when supported."""
        close = getattr(self._client, "close", None)
        if callable(close):
            close()
        self._collection = None
        self._client = None

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        vectors = self.embedder.encode(texts)
        return vectors.tolist() if hasattr(vectors, "tolist") else [list(vector) for vector in vectors]

    def ingest_directory(
        self,
        policy_directory: str | Path,
        *,
        rebuild: bool = False,
        chunk_size: int = 800,
        chunk_overlap: int = 120,
    ) -> IngestionReport:
        """Read PDFs page-by-page, chunk and upsert them; optionally rebuild index.

        Repeated normal ingestion uses deterministic IDs plus ``upsert`` and
        therefore does not duplicate unchanged chunks. Set ``rebuild=True`` to
        explicitly clear the collection before indexing changed source files.
        """
        directory = Path(policy_directory)
        report = IngestionReport()
        if not directory.is_dir():
            report.issues.append(f"Policy directory does not exist: {directory}")  # type: ignore[union-attr]
            return report
        pdf_files = sorted(directory.glob("*.pdf"))
        if not pdf_files:
            report.issues.append(f"No PDF files found in policy directory: {directory}")
            return report
        if rebuild:
            self._reset_collection()
        all_chunks: list[PolicyChunk] = []
        for path in pdf_files:
            try:
                reader = PdfReader(str(path))
                if reader.is_encrypted:
                    try:
                        reader.decrypt("")
                    except Exception as exc:
                        report.issues.append(f"{path.name}: encrypted PDF could not be opened ({exc}).")  # type: ignore[union-attr]
                        continue
                page_texts = [page.extract_text() or "" for page in reader.pages]
                document_policy_match = _POLICY_ID_PATTERN.search("\n".join(page_texts))
                document_policy_id = (
                    document_policy_match.group(0).upper() if document_policy_match else None
                )
                file_chunks: list[PolicyChunk] = []
                has_text = False
                for page_number, extracted in enumerate(page_texts, start=1):
                    report.pages_processed += 1
                    if extracted.strip():
                        has_text = True
                        file_chunks.extend(
                            build_policy_chunks(
                                path.name,
                                page_number,
                                extracted,
                                policy_id=document_policy_id,
                                chunk_size=chunk_size,
                                chunk_overlap=chunk_overlap,
                            )
                        )
                if not has_text:
                    report.issues.append(f"{path.name}: PDF contains no extractable text.")  # type: ignore[union-attr]
                    continue
                report.pdfs_read += 1
                report.indexed_files.append(path.name)  # type: ignore[union-attr]
                report.indexed_policy_ids = sorted(
                    set(report.indexed_policy_ids).union(
                        str(chunk.metadata["policy_id"])
                        for chunk in file_chunks
                        if chunk.metadata["policy_id"] != "UNKNOWN"
                    )
                )
                all_chunks.extend(file_chunks)
            except Exception as exc:
                report.issues.append(f"{path.name}: unreadable PDF ({type(exc).__name__}: {exc}).")  # type: ignore[union-attr]

        if all_chunks:
            collection = self.collection
            batch_size = 128
            for offset in range(0, len(all_chunks), batch_size):
                batch = all_chunks[offset : offset + batch_size]
                collection.upsert(
                    ids=[chunk.chunk_id for chunk in batch],
                    documents=[chunk.text for chunk in batch],
                    metadatas=[chunk.metadata for chunk in batch],
                    embeddings=self._embed([chunk.text for chunk in batch]),
                )
            report.chunks_created = len(all_chunks)
            for chunk in all_chunks:
                chunk.metadata["embedding_model"] = self.embedding_model_name
            self._record_ingestion_metadata(report, directory)
        return report

    def _reset_collection(self) -> None:
        """Clear this named collection for an explicit rebuild request."""
        current = self.collection
        self._client.delete_collection(name=self.collection_name)
        self._collection = self._client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine", "embedding_model": self.embedding_model_name},
        )

    def _record_ingestion_metadata(self, report: IngestionReport, directory: Path) -> None:
        metadata_path = self.persist_directory / f"{self.collection_name}_ingestion.json"
        metadata_path.write_text(
            json.dumps(
                {
                    "embedding_model": self.embedding_model_name,
                    "source_directory": str(directory.resolve()),
                    "pdfs_read": report.pdfs_read,
                    "pages_processed": report.pages_processed,
                    "chunks_created": report.chunks_created,
                    "indexed_files": report.indexed_files,
                    "issues": report.issues,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def retrieve(self, question: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return top-k evidence chunks with source metadata and cosine similarity.

        Similarity is ``1 - cosine_distance`` where Chroma returns a distance.
        This is a retrieval score, not a calibrated confidence or policy finding.
        """
        if not isinstance(question, str):
            raise TypeError("question must be a string.")
        if not question.strip():
            raise ValueError("question must not be empty or whitespace only.")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or top_k <= 0:
            raise ValueError("top_k must be a positive integer.")
        collection = self.collection
        if collection.count() == 0:
            return []
        result = collection.query(
            query_embeddings=self._embed([question]),
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]
        rows: list[dict[str, Any]] = []
        for index, document in enumerate(documents):
            metadata = metadatas[index] or {}
            distance = distances[index] if index < len(distances) else None
            rows.append(
                {
                    "text": document,
                    "source": metadata.get("source", metadata.get("filename", "")),
                    "filename": metadata.get("filename", metadata.get("source", "")),
                    "page_number": metadata.get("page_number"),
                    "policy_id": metadata.get("policy_id"),
                    "section": metadata.get("section", ""),
                    "relevance_score": 1.0 - float(distance) if distance is not None else None,
                    "chunk_id": ids[index] if index < len(ids) else None,
                }
            )
        return rows
