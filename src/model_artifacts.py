"""Resolve the existing fine-tuned BERT model locally or from a pinned Hub repo."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Callable

from src.runtime_config import get_setting

_MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
    "label_mapping.json",
    "model_metadata.json",
)
_COMMIT_HASH = re.compile(r"[0-9a-fA-F]{40}\Z")


def _has_required_model_files(directory: Path) -> bool:
    return directory.is_dir() and all((directory / name).is_file() for name in _MODEL_FILES)


def resolve_bert_model_dir(
    local_directory: str | Path,
    *,
    repo_id: str | None = None,
    revision: str | None = None,
    token: str | None = None,
    downloader: Callable[..., str] | None = None,
) -> Path:
    """Use complete local weights or download the same artifact at a pinned SHA.

    No model is downloaded unless a repository and immutable 40-character
    revision are configured. Download errors are deliberately sanitized.
    """
    local_path = Path(local_directory).expanduser().resolve()
    if _has_required_model_files(local_path):
        return local_path

    configured_repo = repo_id if repo_id is not None else get_setting("BERT_MODEL_REPO")
    configured_revision = (
        revision if revision is not None else get_setting("BERT_MODEL_REVISION")
    )
    configured_token = token if token is not None else get_setting("HF_TOKEN")
    if not isinstance(configured_repo, str) or not configured_repo.strip():
        raise FileNotFoundError(
            "The fine-tuned BERT model is unavailable locally. Configure BERT_MODEL_REPO "
            "and BERT_MODEL_REVISION for deployment."
        )
    if not isinstance(configured_revision, str) or not _COMMIT_HASH.fullmatch(
        configured_revision.strip()
    ):
        raise ValueError("BERT_MODEL_REVISION must be an immutable 40-character commit SHA.")

    try:
        if downloader is None:
            from huggingface_hub import snapshot_download

            downloader = snapshot_download
        snapshot = Path(
            downloader(
                repo_id=configured_repo.strip(),
                revision=configured_revision.strip(),
                token=configured_token or None,
                allow_patterns=list(_MODEL_FILES),
            )
        ).resolve()
    except Exception:
        raise RuntimeError(
            "The configured fine-tuned BERT model could not be downloaded. "
            "Check model access and deployment connectivity."
        ) from None

    if not _has_required_model_files(snapshot):
        raise FileNotFoundError(
            "The configured model repository does not contain all required classifier files."
        )
    return snapshot
