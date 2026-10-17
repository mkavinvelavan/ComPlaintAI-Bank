"""Offline tests for policy chunking, indexing, and retrieval."""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path
from typing import Sequence
from uuid import uuid4

import chromadb
from pypdf import PdfWriter

from src.policy_rag import PolicyRetriever, build_policy_chunks, recursive_character_chunks


class WorkspaceTemporaryDirectory:
    """Writable, uniquely named test storage under the repository's tests folder."""

    retrievers: list[PolicyRetriever]

    def __enter__(self) -> "WorkspaceTemporaryDirectory":
        self.path = Path(__file__).resolve().parent / f".policy_rag_test_{uuid4().hex}"
        self.path.mkdir()
        self.retrievers = []
        return self

    def __fspath__(self) -> str:
        return str(self.path)

    def __exit__(self, *_: object) -> None:
        for retriever in self.retrievers:
            retriever.close()
        shutil.rmtree(self.path)

    def retriever(self, *args: object, **kwargs: object) -> PolicyRetriever:
        persistent = bool(kwargs.pop("persistent", False))
        if not persistent:
            kwargs.setdefault("client", chromadb.EphemeralClient())
        instance = PolicyRetriever(*args, **kwargs)  # type: ignore[arg-type]
        self.retrievers.append(instance)
        return instance

class TinyDeterministicEmbedder:
    """Small token-hash vectors keep tests independent of model downloads."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            vector = [0.0] * 64
            for token in text.lower().split():
                vector[sum(token.encode("utf-8")) % len(vector)] += 1.0
            norm = sum(value * value for value in vector) ** 0.5 or 1.0
            vectors.append([value / norm for value in vector])
        return vectors


def make_pdf(path: Path, text: str) -> None:
    """Create a tiny synthetic PDF with extractable content."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    # pypdf does not draw text; use a minimal generated PDF content stream.
    from pypdf.generic import DecodedStreamObject, NameObject

    stream = DecodedStreamObject()
    safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(f"BT /F1 12 Tf 20 250 Td ({safe}) Tj ET".encode("latin-1"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    from pypdf.generic import DictionaryObject

    font = DictionaryObject(
        {
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
        }
    )
    fonts = DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    resources = DictionaryObject({NameObject("/Font"): writer._add_object(fonts)})
    page[NameObject("/Resources")] = writer._add_object(resources)
    with path.open("wb") as output:
        writer.write(output)


class PolicyRagTests(unittest.TestCase):
    def test_chunking_keeps_overlap_and_avoids_empty_chunks(self) -> None:
        text = ("Paragraph one ends here.\n\n" + "word " * 220).strip()
        chunks = recursive_character_chunks(text, chunk_size=120, chunk_overlap=20)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunks))
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertTrue(any(set(chunks[i]) & set(chunks[i + 1]) for i in range(len(chunks) - 1)))
        self.assertEqual(recursive_character_chunks("  \n "), [])

    def test_metadata_policy_id_page_and_unique_ids(self) -> None:
        text = "Policy ID: CPB-POL-004\n1. Transfers\n" + "transfer detail " * 100
        chunks = build_policy_chunks("04_policy.pdf", 7, text, chunk_size=100, chunk_overlap=10)
        self.assertGreater(len(chunks), 1)
        self.assertEqual({chunk.metadata["source"] for chunk in chunks}, {"04_policy.pdf"})
        self.assertEqual({chunk.metadata["page_number"] for chunk in chunks}, {7})
        self.assertEqual({chunk.metadata["policy_id"] for chunk in chunks}, {"CPB-POL-004"})
        self.assertEqual(len({chunk.chunk_id for chunk in chunks}), len(chunks))
        self.assertEqual([chunk.chunk_id for chunk in chunks], [
            chunk.chunk_id for chunk in build_policy_chunks("04_policy.pdf", 7, text, chunk_size=100, chunk_overlap=10)
        ])

    def test_document_policy_id_can_be_carried_to_pages_without_repeated_id(self) -> None:
        chunk = build_policy_chunks("policy.pdf", 2, "5.1 Account controls apply.", policy_id="CPB-POL-005")[0]
        self.assertEqual(chunk.metadata["policy_id"], "CPB-POL-005")

    def test_synthetic_pdf_ingestion_retrieval_and_metadata(self) -> None:
        with WorkspaceTemporaryDirectory() as temporary:
            root = Path(temporary)
            make_pdf(root / "CPB-POL-001.pdf", "Policy ID: CPB-POL-001. Card activation and debit card usage.")
            retriever = temporary.retriever(
                root, collection_name="test_relevant", embedder=TinyDeterministicEmbedder()
            )
            report = retriever.ingest_directory(root, chunk_size=800, chunk_overlap=120)
            self.assertEqual((report.pdfs_read, report.pages_processed, report.chunks_created), (1, 1, 1))
            rows = retriever.retrieve("debit card activation", top_k=1)
            self.assertEqual(rows[0]["filename"], "CPB-POL-001.pdf")
            self.assertEqual(rows[0]["policy_id"], "CPB-POL-001")
            self.assertEqual(rows[0]["page_number"], 1)
            self.assertIsInstance(rows[0]["relevance_score"], float)

    def test_persistent_collection_can_be_reopened(self) -> None:
        with WorkspaceTemporaryDirectory() as temporary:
            root = Path(temporary)
            make_pdf(root / "sample.pdf", "Policy ID: CPB-POL-002. Card activation support.")
            first = temporary.retriever(root, collection_name="test_persistence", embedder=TinyDeterministicEmbedder(), persistent=True)
            first.ingest_directory(root)
            reopened = temporary.retriever(root, collection_name="test_persistence", embedder=TinyDeterministicEmbedder(), persistent=True)
            self.assertEqual(reopened.collection.count(), 1)
            self.assertTrue(reopened.retrieve("card activation"))

    def test_invalid_queries_and_top_k_are_rejected(self) -> None:
        retriever = PolicyRetriever(embedder=TinyDeterministicEmbedder())
        for question in ("", " \t"):
            with self.subTest(question=repr(question)), self.assertRaises(ValueError):
                retriever.retrieve(question)
        with self.assertRaises(TypeError):
            retriever.retrieve(None)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            retriever.retrieve("question", top_k=0)

    def test_empty_collection_returns_no_matches(self) -> None:
        with WorkspaceTemporaryDirectory() as temporary:
            retriever = temporary.retriever(Path(temporary), collection_name="test_empty", embedder=TinyDeterministicEmbedder())
            self.assertEqual(retriever.retrieve("policy question"), [])

    def test_missing_empty_and_unreadable_pdfs_are_reported(self) -> None:
        with WorkspaceTemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = temporary.retriever(root, collection_name="test_missing", embedder=TinyDeterministicEmbedder())
            self.assertTrue(missing.ingest_directory(root / "missing-policies").issues)
            (root / "empty.pdf").write_bytes(b"")
            make_pdf(root / "blank.pdf", "")
            retriever = temporary.retriever(root, collection_name="test_invalid_pdfs", embedder=TinyDeterministicEmbedder())
            report = retriever.ingest_directory(root)
            self.assertEqual(report.pdfs_read, 0)
            self.assertEqual(len(report.issues), 2)

    def test_repeat_ingestion_upserts_without_duplicate_chunks(self) -> None:
        with WorkspaceTemporaryDirectory() as temporary:
            root = Path(temporary)
            make_pdf(root / "sample.pdf", "Policy ID: CPB-POL-003. Dispute reporting and card security.")
            retriever = temporary.retriever(root, collection_name="test_idempotent", embedder=TinyDeterministicEmbedder())
            first = retriever.ingest_directory(root)
            second = retriever.ingest_directory(root)
            self.assertEqual(first.chunks_created, second.chunks_created)
            self.assertEqual(retriever.collection.count(), first.chunks_created)


if __name__ == "__main__":
    unittest.main()
