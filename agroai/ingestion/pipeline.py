"""The ingestion pipeline: file -> knowledge base, stage by stage.

Stages, in order (roadmap "Knowledge ingestion system"):

1. **load** - pick the right reader for the extension;
2. **extract** - the reader produces raw text (faked for nothing: unimplemented
   formats raise);
3. **clean** - normalize newlines, whitespace and control characters;
4. **detect language** - script/lexicon heuristic (uz / ru / en);
5. **metadata** - build the provenance record, including trust level;
6. **deduplicate** - content hash lookup (incremental ingestion);
7. **chunk** - split into provenance-preserving chunks;
8. **store** - persist document + chunks + source record.

The pipeline never guesses. Unknown-language text keeps the configured
default, duplicates are reported as skipped rather than silently overwritten,
and a failing file is reported as failed with its error message.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from agroai.config.settings import Settings, get_settings
from agroai.database.database import Database
from agroai.ingestion.base import (
    DocumentReadError,
    UnsupportedFormatError,
    get_reader,
)
from agroai.knowledge.metadata import content_hash, utcnow_iso
from agroai.knowledge.documents import (
    Chunk,
    IngestionReport,
    KnowledgeDocument,
    iter_chunks,
)
from agroai.knowledge.metadata import DocumentMetadata
from agroai.knowledge.sources import SourceRecord
from agroai.knowledge.versions import SQLiteVersionStore, VersionStore

logger = logging.getLogger(__name__)

#: Directory names skipped when walking a knowledge tree.
IGNORED_DIRECTORIES = {"", ".git", ".venv", "__pycache__", ".mypy_cache", ".pytest_cache"}

#: Stage names, in execution order (useful for logging and tests).
STAGES = (
    "load",
    "extract",
    "clean",
    "detect_language",
    "metadata",
    "deduplicate",
    "chunk",
    "store",
)

_BOM = "\ufeff"
# Keep tab, newline; drop other C0/C1 control characters.
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]")
_CRLF = re.compile(r"\r\n?")
_TRAILING_SPACE = re.compile(r"[ \t]+$", re.MULTILINE)
_MANY_BLANKS = re.compile(r"\n{3,}")

# Uzbek Latin-specific signals: apostrophe letters and Uzbek-only vocabulary.
# Bare digraphs like "sh"/"ch"/"ng" are deliberately excluded - they occur all
# over English text and would make the detector guess.
_UZ_MARKERS = (
    "oʻ", "o'", "gʻ", "g'", "o‘", "g‘", "tuproq", "o'simlik", "o‘simlik",
    "kasallik", "sug'or", "sug‘or", "sho'rlan", "sho‘rlan", "bo'ladi",
    "bo‘ladi", "hisoblanadi", "uchun", "bilan", "kerak", "hamda", "qishloq",
    "hosil", "urug'", "o'g'it",
)
_EN_MARKERS = (
    " the ", " and ", " of ", " to ", " is ", " are ", " with ", " for ",
    " soil ", " plant ", " water ", " disease ", " crop ",
)
# Cyrillic letters that only occur in Uzbek Cyrillic (not in Russian).
_UZ_CYRILLIC = ("ў", "қ", "ғ", "ҳ")


@dataclass
class LanguageGuess:
    """Result of the language-detection stage."""

    language: str
    confidence: float
    method: str = "heuristic"

    @property
    def is_confident(self) -> bool:
        return self.confidence >= 0.5


def clean_text(text: str) -> str:
    """Normalize raw text: newlines, trailing spaces, blanks, control chars."""
    text = (text or "").replace(_BOM, "")
    text = _CRLF.sub("\n", text)
    text = _CONTROL.sub("", text)
    text = _TRAILING_SPACE.sub("", text)
    text = _MANY_BLANKS.sub("\n\n", text)
    return text.strip()


def detect_language(text: str, default: str = "uz") -> LanguageGuess:
    """Guess the document language from its script and a small lexicon.

    Deliberately conservative: when the evidence is weak the configured default
    is kept, because mislabelling a document's language is worse than leaving it
    at the project default. Supported answers: ``uz``, ``ru``, ``en``.
    """
    sample = (text or "")[:4000].lower()
    if not sample.strip():
        return LanguageGuess(default, 0.0, "empty")

    letters = [char for char in sample if char.isalpha()]
    if not letters:
        return LanguageGuess(default, 0.0, "no-letters")

    cyrillic = sum(1 for char in letters if "а" <= char <= "я" or char in "ёўқғҳ")
    ratio = cyrillic / len(letters)
    if ratio >= 0.3:
        if any(marker in sample for marker in _UZ_CYRILLIC):
            return LanguageGuess("uz", min(1.0, ratio + 0.2), "cyrillic-uz")
        return LanguageGuess("ru", min(1.0, ratio + 0.2), "cyrillic-ru")

    uz_score = sum(sample.count(marker) for marker in _UZ_MARKERS)
    en_score = sum(sample.count(marker) for marker in _EN_MARKERS)

    if uz_score == en_score:
        return LanguageGuess(default, 0.3, "undecided")
    winner, score, loser = (
        ("uz", uz_score, en_score) if uz_score > en_score else ("en", en_score, uz_score)
    )
    total = score + loser
    confidence = score / total if total else 0.0
    # Require a clear margin before overriding the configured default.
    if score - loser < 2:
        return LanguageGuess(default, confidence / 2, "weak")
    return LanguageGuess(winner, confidence, "latin-lexicon")


@dataclass
class PreparedDocument:
    """A document that has passed every offline stage, ready to be stored."""

    document: KnowledgeDocument
    chunks: list[Chunk] = field(default_factory=list)
    metadata: DocumentMetadata | None = None
    language: LanguageGuess | None = None
    stages_run: list[str] = field(default_factory=list)

    @property
    def metadata_problems(self) -> list[str]:
        return self.metadata.problems() if self.metadata else []


class IngestionPipeline:
    """Drives a file through all ingestion stages."""

    def __init__(
        self,
        database: Database | None = None,
        settings: Settings | None = None,
        versions: VersionStore | None = None,
    ) -> None:
        self._db = database
        self._settings = settings or get_settings()
        self._versions = versions or (SQLiteVersionStore(database) if database is not None else None)

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def versions(self) -> VersionStore | None:
        return self._versions

    # ------------------------------------------------------------------ #
    # Prepare (no writes)
    # ------------------------------------------------------------------ #
    def prepare(self, path: Path) -> PreparedDocument:
        """Run load -> extract -> clean -> language -> metadata -> chunk."""
        path = Path(path)
        stages: list[str] = []
        stages.append("load")
        reader = get_reader(path)
        stages.append("extract")
        document = reader.read(path, self._settings)

        stages.append("clean")
        document.content = clean_text(document.content)
        if not document.content:
            raise DocumentReadError(f"{path} contains no usable text after cleaning.")
        # The content hash must describe the *stored* content, not the raw
        # extraction: deduplication, versioning and 'agroai verify' all compare
        # this hash against what the knowledge base actually holds.
        document.content_hash = content_hash(document.content)
        document.updated_at = utcnow_iso()

        stages.append("detect_language")
        guess = detect_language(document.content, self._settings.default_language)
        document.language = guess.language

        stages.append("metadata")
        metadata = self._build_metadata(document, guess)
        document.metadata.update(self._metadata_fields(metadata, guess))

        stages.append("chunk")
        chunks = list(
            iter_chunks(
                document,
                chunk_size=self._settings.chunk_size,
                overlap=self._settings.chunk_overlap,
            )
        )
        return PreparedDocument(
            document=document,
            chunks=chunks,
            metadata=metadata,
            language=guess,
            stages_run=stages,
        )

    def _build_metadata(self, document: KnowledgeDocument, guess: LanguageGuess) -> DocumentMetadata:
        metadata = DocumentMetadata.for_source(
            title=document.title,
            source_name=document.source,
            source_type=str(document.metadata.get("source_type", "")),
            category=document.category,
            language=guess.language,
            author=document.author,
            source_url=document.source_url,
            publication_date=document.publication_date,
            content_hash=document.content_hash,
            country=str(document.metadata.get("country", "")),
            region=str(document.metadata.get("region", "")),
            license=str(document.metadata.get("license", "")),
            version=str(document.metadata.get("version", "1")),
        )
        metadata.created_at = document.created_at
        metadata.updated_at = document.updated_at
        return metadata

    @staticmethod
    def _metadata_fields(metadata: DocumentMetadata, guess: LanguageGuess) -> dict[str, object]:
        fields: dict[str, object] = {
            "trust_level": metadata.trust_level.value,
            "organization": metadata.organization,
            "publication_date": metadata.publication_date,
            "language_confidence": round(guess.confidence, 3),
            "language_method": guess.method,
        }
        if metadata.country:
            fields["country"] = metadata.country
        if metadata.region:
            fields["region"] = metadata.region
        if metadata.license:
            fields["license"] = metadata.license
        if metadata.version:
            fields["version"] = metadata.version
        problems = metadata.problems()
        if problems:
            fields["metadata_problems"] = problems
        return fields

    # ------------------------------------------------------------------ #
    # Run (prepare + deduplicate + store)
    # ------------------------------------------------------------------ #
    def ingest_file(self, path: Path, *, overwrite: bool = False) -> IngestionReport:
        """Ingest one file. Returns an honest report of what happened."""
        report = IngestionReport(documents_seen=1)
        path = Path(path)
        try:
            prepared = self.prepare(path)
        except (
            DocumentReadError,
            UnsupportedFormatError,
            NotImplementedError,
            OSError,
        ) as exc:
            report.documents_failed = 1
            report.errors.append(f"{path}: {exc}")
            logger.warning("Ingestion failed for %s: %s", path, exc)
            return report

        stored = self._store(prepared, report, overwrite=overwrite)
        if stored:
            report.documents_ingested = 1
            report.chunks_created = len(prepared.chunks)
        else:
            report.documents_skipped = 1
        return report

    def ingest_document(
        self, document: KnowledgeDocument, *, overwrite: bool = False
    ) -> IngestionReport:
        """Store an already-built document (approved candidates, API payloads).

        Chunking still happens here, so provenance metadata and the version
        archive behave exactly as they do for files.
        """
        if self._db is None:
            raise RuntimeError("IngestionPipeline needs a database to store documents.")
        report = IngestionReport(documents_seen=1)
        chunks = list(
            iter_chunks(
                document,
                chunk_size=self._settings.chunk_size,
                overlap=self._settings.chunk_overlap,
            )
        )
        prepared = PreparedDocument(
            document=document, chunks=chunks, stages_run=["chunk", "store"]
        )
        if self._store(prepared, report, overwrite=overwrite):
            report.documents_ingested = 1
            report.chunks_created = len(chunks)
        else:
            report.documents_skipped = 1
        return report

    def ingest_directory(
        self, directory: Path, *, recursive: bool = True, overwrite: bool = False
    ) -> IngestionReport:
        """Ingest every file under ``directory`` (best effort, never aborts)."""
        directory = Path(directory)
        report = IngestionReport()
        if not directory.is_dir():
            report.errors.append(f"Not a directory: {directory}")
            report.documents_failed = 1
            return report

        pattern = "**/*" if recursive else "*"
        for path in sorted(directory.glob(pattern)):
            if not path.is_file() or path.name.startswith("."):
                continue
            if any(part in IGNORED_DIRECTORIES for part in path.parts):
                continue
            report.merge(self.ingest_file(path, overwrite=overwrite))
        return report

    # ------------------------------------------------------------------ #
    # Store
    # ------------------------------------------------------------------ #
    def _store(
        self, prepared: PreparedDocument, report: IngestionReport, *, overwrite: bool
    ) -> bool:
        """Persist document + chunks. Returns False when skipped as duplicate."""
        if self._db is None:
            raise RuntimeError("IngestionPipeline needs a database to store documents.")

        document = prepared.document
        self._archive_previous_versions(document, report)

        existing_id = self._db.find_by_hash(document.content_hash)
        if existing_id and self._settings.deduplicate and not overwrite:
            logger.info(
                "Skipping duplicate (hash %s…): %s", document.content_hash[:12], document.title
            )
            return False

        if existing_id and existing_id != document.id:
            # Same content under a different generated id: keep the trail, then
            # remove the duplicate copy so the database has one row per content.
            duplicate = self._db.get_document(existing_id)
            if duplicate is not None and self._versions is not None:
                self._versions.archive(duplicate, reason="duplicate_content")
                report.versions_archived += 1
            self._db.delete_document(existing_id)

        self._db.upsert_document(document)
        self._db.add_chunks(prepared.chunks, title_map={document.id: document.title})
        self._db.upsert_source(
            SourceRecord(
                name=document.source or "local",
                source_type="file",
                url=document.source_url,
                document_count=1,
                last_ingested_at=str(document.metadata.get("ingested_at", "")),
                metadata={
                    "category": document.category,
                    "format": document.metadata.get("format", ""),
                    "trust_level": document.trust_level.value,
                },
            )
        )
        return True

    def _archive_previous_versions(
        self, document: KnowledgeDocument, report: IngestionReport
    ) -> None:
        """Keep the previous version when the same document id changes.

        The roadmap requires that a changed document keeps *both* versions. The
        new version number is derived from the previous one, and the archived
        copy is stored before the new content overwrites it.
        """
        if self._versions is None or self._db is None:
            return
        previous = self._db.get_document(document.id)
        if previous is None or previous.content_hash == document.content_hash:
            return

        self._versions.archive(previous, reason="content_changed")
        report.versions_archived += 1
        document.metadata["version"] = _next_version(previous.metadata.get("version", "1"))
        document.metadata["previous_version_hash"] = previous.content_hash
        document.metadata["version_archived_at"] = utcnow_iso()


def _next_version(current: object) -> str:
    """Return the next version label (numeric when possible, else '<x>+1')."""
    try:
        return str(int(str(current)) + 1)
    except (TypeError, ValueError):
        return f"{current}+1"


__all__ = [
    "IGNORED_DIRECTORIES",
    "STAGES",
    "IngestionPipeline",
    "LanguageGuess",
    "PreparedDocument",
    "clean_text",
    "detect_language",
]
