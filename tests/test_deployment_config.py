"""Offline tests for deployment configuration and asset initialization."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.admin_auth import AdminAuthenticator
from src.model_artifacts import resolve_bert_model_dir
from src.runtime_config import get_setting


class RuntimeConfigTests(unittest.TestCase):
    def test_process_environment_takes_precedence_over_streamlit_secret(self) -> None:
        with (
            patch("src.runtime_config.load_dotenv"),
            patch.dict("os.environ", {"DATABASE_URL": "environment-value"}, clear=True),
            patch("src.runtime_config._read_streamlit_secret", return_value="secret-value"),
        ):
            self.assertEqual(get_setting("DATABASE_URL"), "environment-value")

    def test_streamlit_secret_is_used_when_environment_is_unset(self) -> None:
        with (
            patch("src.runtime_config.load_dotenv"),
            patch.dict("os.environ", {}, clear=True),
            patch("src.runtime_config._read_streamlit_secret", return_value="secret-value"),
        ):
            self.assertEqual(get_setting("GROQ_API_KEY"), "secret-value")

    def test_blank_values_are_unconfigured_and_no_secret_is_returned(self) -> None:
        with (
            patch("src.runtime_config.load_dotenv"),
            patch.dict("os.environ", {"DATABASE_URL": " "}, clear=True),
            patch("src.runtime_config._read_streamlit_secret", return_value=None),
        ):
            self.assertIsNone(get_setting("DATABASE_URL"))

    def test_admin_authentication_fails_closed_for_missing_or_short_password(self) -> None:
        self.assertFalse(AdminAuthenticator(username="admin", password="short-demo").is_configured)
        auth = AdminAuthenticator(username="admin", password="a-strong-synthetic-password")
        self.assertTrue(auth.is_configured)
        self.assertTrue(auth.authenticate("admin", "a-strong-synthetic-password"))
        self.assertFalse(auth.authenticate("admin", "wrong-password"))


class ModelArtifactTests(unittest.TestCase):
    def test_complete_local_model_is_used_without_download(self) -> None:
        model_path = Path("synthetic-local-model")
        downloader = Mock(side_effect=AssertionError("must not download"))
        with patch("src.model_artifacts._has_required_model_files", return_value=True):
            self.assertEqual(
                resolve_bert_model_dir(model_path, downloader=downloader), model_path.resolve()
            )
        downloader.assert_not_called()

    def test_configured_model_download_is_pinned_and_file_limited(self) -> None:
        snapshot = Path("synthetic-model-snapshot")
        downloader = Mock(return_value=str(snapshot))
        with patch(
            "src.model_artifacts._has_required_model_files",
            side_effect=lambda directory: Path(directory).name == snapshot.name,
        ):
            resolved = resolve_bert_model_dir(
                Path("missing-model"),
                repo_id="project/bert-model",
                revision="a" * 40,
                token="synthetic-token",
                downloader=downloader,
            )
        self.assertEqual(resolved, snapshot.resolve())
        args = downloader.call_args.kwargs
        self.assertEqual(args["repo_id"], "project/bert-model")
        self.assertEqual(args["revision"], "a" * 40)
        self.assertEqual(args["token"], "synthetic-token")
        self.assertIn("model.safetensors", args["allow_patterns"])
        self.assertNotIn("training_args.bin", args["allow_patterns"])

    def test_unpinned_revision_is_rejected_before_download(self) -> None:
        downloader = Mock()
        with self.assertRaisesRegex(ValueError, "40-character"):
            resolve_bert_model_dir(
                "missing-model",
                repo_id="project/bert-model",
                revision="main",
                downloader=downloader,
            )
        downloader.assert_not_called()

    def test_download_failure_does_not_expose_token_or_provider_error(self) -> None:
        secret = "synthetic-private-token"
        downloader = Mock(side_effect=RuntimeError(f"request contained {secret}"))
        with self.assertRaises(RuntimeError) as raised:
            resolve_bert_model_dir(
                "missing-model",
                repo_id="project/private-model",
                revision="a" * 40,
                token=secret,
                downloader=downloader,
            )
        self.assertNotIn(secret, str(raised.exception))
        self.assertNotIn("request contained", str(raised.exception))


class PolicyIndexInitializationTests(unittest.TestCase):
    def test_new_index_uses_non_destructive_ingestion_once(self) -> None:
        import src.deployment_resources as resources

        fake = Mock()
        fake.persist_directory = Path("temporary-index")
        fake.collection_name = "policies"
        fake.ingest_directory.return_value = SimpleNamespace(
            issues=[],
            pdfs_read=8,
            indexed_files=sorted(resources._EXPECTED_FILES),
            indexed_policy_ids=sorted(resources._EXPECTED_POLICY_IDS),
            chunks_created=365,
        )
        with (
            patch.object(resources, "_RETRIEVER", None),
            patch.object(resources, "PolicyRetriever", return_value=fake) as constructor,
            patch("src.deployment_resources._valid_ingestion_manifest", return_value=False),
        ):
            first = resources.get_policy_retriever()
            second = resources.get_policy_retriever()

        self.assertIs(first, fake)
        self.assertIs(second, fake)
        self.assertEqual(constructor.call_args.kwargs["local_files_only"], False)
        fake.ingest_directory.assert_called_once_with(
            resources.PROJECT_ROOT / "knowledge_base" / "policies", rebuild=False
        )

    def test_valid_existing_manifest_skips_ingestion(self) -> None:
        import src.deployment_resources as resources

        fake = Mock()
        fake.persist_directory = Path("synthetic-index")
        fake.collection_name = "policies"
        fake.collection.count.return_value = 365
        manifest = {
            "pdfs_read": 8,
            "indexed_files": sorted(resources._EXPECTED_FILES),
            "chunks_created": 365,
            "issues": [],
        }
        with (
            patch.object(resources, "_RETRIEVER", None),
            patch.object(resources, "PolicyRetriever", return_value=fake),
            patch.object(resources, "get_setting", return_value=None),
            patch("pathlib.Path.read_text", return_value=json.dumps(manifest)),
        ):
            self.assertIs(resources.get_policy_retriever(), fake)
        fake.ingest_directory.assert_not_called()


if __name__ == "__main__":
    unittest.main()
