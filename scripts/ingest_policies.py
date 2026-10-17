"""Build or refresh the local ChromaDB index from policy PDFs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.policy_rag import DEFAULT_PERSIST_DIRECTORY, PolicyRetriever  # noqa: E402

EXPECTED_POLICY_IDS = {f"CPB-POL-{number:03d}" for number in range(1, 9)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--policies-dir",
        type=Path,
        default=PROJECT_ROOT / "knowledge_base" / "policies",
        help="Directory containing authoritative policy PDFs.",
    )
    parser.add_argument(
        "--persist-dir",
        type=Path,
        default=DEFAULT_PERSIST_DIRECTORY,
        help="Persistent ChromaDB directory (default: data/chroma_db).",
    )
    args = parser.parse_args()

    # Permit downloading the public model on first build. All embedding and
    # retrieval operations run locally; query-time defaults to local-only.
    retriever = PolicyRetriever(persist_directory=args.persist_dir, local_files_only=False)
    try:
        # This entry point only refreshes rows through stable-ID upserts. It
        # never clears or recreates the persistent collection.
        report = retriever.ingest_directory(args.policies_dir)
    except Exception as exc:
        print(f"Policy ingestion failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    missing_ids = sorted(EXPECTED_POLICY_IDS.difference(report.indexed_policy_ids))
    for policy_id in missing_ids:
        report.issues.append(
            f"Expected policy identifier {policy_id} was not found in any indexed PDF."
        )
    print(json.dumps(report.as_dict(), indent=2))
    if report.issues:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
