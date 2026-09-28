#!/usr/bin/env python
"""Export source-grounded dataset records from the local knowledge base.

Usage:
    python scripts/export_dataset.py [--out datasets/agroai_grounded.jsonl]
                                     [--limit N] [--min-chars N]

Every exported record is *extractive*: the output is a passage that really
exists in the knowledge base, plus the citation it came from. No question or
answer text is synthesized, so this cannot introduce hallucinated training data
(see datasets/README.md for what is still missing).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running from a checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agroai.config.settings import get_settings  # noqa: E402
from agroai.database.database import Database  # noqa: E402
from agroai.models.fine_tuning import FineTuningDataset  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        default="datasets/agroai_grounded.jsonl",
        help="output JSONL path (default: datasets/agroai_grounded.jsonl)",
    )
    parser.add_argument("--limit", type=int, default=None, help="maximum records to export")
    parser.add_argument(
        "--min-chars", type=int, default=200, help="skip documents shorter than this"
    )
    args = parser.parse_args(argv)

    settings = get_settings()
    settings.ensure_directories()

    if not settings.database_path.is_file():
        print(f"Database not found: {settings.database_path}", file=sys.stderr)
        print("Ingest knowledge first: agroai ingest-dir ./knowledge", file=sys.stderr)
        return 1

    database = Database(settings.database_path)
    database.connect()
    database.initialize()
    try:
        documents = list(database.iter_documents())
        if not documents:
            print("The knowledge base is empty; nothing to export.", file=sys.stderr)
            return 1

        dataset = FineTuningDataset.from_documents(
            documents, min_chars=args.min_chars, max_records=args.limit
        )
        if len(dataset) == 0:
            print(
                f"No document reached {args.min_chars} characters; nothing exported.",
                file=sys.stderr,
            )
            return 1

        written = dataset.to_jsonl(args.out)
        summary = dataset.summary()
        print(f"Wrote {written} record(s) to {args.out}")
        print(f"Kinds: {summary['kinds']}")
        print(f"Languages: {summary['languages']}")
        print(f"Trust levels: {summary['trust_levels']}")
        if summary["invalid"]:
            print(f"Invalid records: {summary['invalid']}", file=sys.stderr)
            return 1
        return 0
    finally:
        database.close()


if __name__ == "__main__":
    raise SystemExit(main())
