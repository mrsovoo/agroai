"""Knowledge manager: the bridge between ingestion and storage.

The manager is a thin facade: it owns the database handle and hands the actual
work to :class:`~agroai.ingestion.pipeline.IngestionPipeline` (load, extract,
clean, detect language, metadata, deduplicate, chunk, store).

It stays the stable entry point for the CLI and tests, and it is the natural
home for the high-level questions an ingester asks: *what is in the knowledge
base?* and *which categories are covered?*
"""

from __future__ import annotations

from pathlib import Path

from agroai.config.settings import Settings, get_settings
from agroai.database.database import Database
from agroai.ingestion.pipeline import IngestionPipeline, PreparedDocument
from agroai.knowledge.candidates import (
    CandidatePipeline,
    KnowledgeCandidate,
    approve,
)
from agroai.knowledge.documents import (
    Chunk,
    IngestionReport,
    KnowledgeDocument,
    iter_chunks,
)
from agroai.knowledge.graph import KnowledgeGraph, build_graph
from agroai.knowledge.versions import DocumentVersion, VersionStore


class KnowledgeManager:
    """High-level facade over readers, chunking and the database."""

    def __init__(self, database: Database, settings: Settings | None = None) -> None:
        self._db = database
        self._settings = settings or get_settings()
        self._pipeline = IngestionPipeline(database, self._settings)
        self._candidates: CandidatePipeline | None = None

    @property
    def settings(self) -> Settings:
        return self._settings

    @property
    def database(self) -> Database:
        return self._db

    @property
    def pipeline(self) -> IngestionPipeline:
        return self._pipeline

    @property
    def versions(self) -> VersionStore | None:
        """Version store used by ingestion (None when no database is attached)."""
        return self._pipeline.versions

    @property
    def candidates(self) -> CandidatePipeline:
        """Review pipeline for unverified material (never auto-approved)."""
        return self._candidate_pipeline()

    # ------------------------------------------------------------------ #
    # Ingestion
    # ------------------------------------------------------------------ #
    def ingest_file(self, path: Path, *, overwrite: bool = False) -> IngestionReport:
        """Ingest one file. Returns an honest report of what happened."""
        return self._pipeline.ingest_file(path, overwrite=overwrite)

    def ingest_directory(
        self, directory: Path, *, recursive: bool = True, overwrite: bool = False
    ) -> IngestionReport:
        """Ingest every supported file under ``directory`` (best effort)."""
        return self._pipeline.ingest_directory(
            directory, recursive=recursive, overwrite=overwrite
        )

    # ------------------------------------------------------------------ #
    # Individual steps (kept public: the CLI, tests and tooling use them)
    # ------------------------------------------------------------------ #
    def prepare_document(self, path: Path) -> PreparedDocument:
        """Run every non-writing pipeline stage for one file."""
        return self._pipeline.prepare(Path(path))

    def load_document(self, path: Path) -> KnowledgeDocument:
        """Read a file into a :class:`KnowledgeDocument` (no persistence)."""
        return self.prepare_document(path).document

    def chunk_document(self, document: KnowledgeDocument) -> list[Chunk]:
        """Split a document into provenance-preserving chunks."""
        return list(
            iter_chunks(
                document,
                chunk_size=self._settings.chunk_size,
                overlap=self._settings.chunk_overlap,
            )
        )

    # ------------------------------------------------------------------ #
    # Versioning
    # ------------------------------------------------------------------ #
    def history(self, document_id: str) -> list[DocumentVersion]:
        """Return archived versions of one document (newest first)."""
        if self.versions is None:
            return []
        return self.versions.history(document_id)

    def versions_count(self) -> int:
        return self.versions.count() if self.versions is not None else 0

    # ------------------------------------------------------------------ #
    # Knowledge candidates (learning pipeline)
    # ------------------------------------------------------------------ #
    def _candidate_pipeline(self) -> CandidatePipeline:
        if self._candidates is None:
            self._candidates = CandidatePipeline(database=self._db)
        return self._candidates

    def submit_candidate(self, candidate: KnowledgeCandidate) -> KnowledgeCandidate:
        """Assess and store a candidate as pending (no automatic acceptance)."""
        return self._candidate_pipeline().submit(candidate)

    def list_candidates(self, status: str | None = None, limit: int | None = None) -> list:
        return self._db.list_candidates(status=status, limit=limit)

    def review_candidate(
        self,
        candidate: KnowledgeCandidate,
        *,
        approved: bool,
        reviewer: str = "",
        notes: str = "",
    ) -> tuple[KnowledgeCandidate, IngestionReport]:
        """Record a human decision; approved candidates enter the knowledge base."""
        reviewed = self._candidate_pipeline().review(
            candidate, approved=approved, reviewer=reviewer, notes=notes
        )
        if not approved:
            return reviewed, IngestionReport()

        document = approve(reviewed, reviewer=reviewer, notes=notes)
        report = self._pipeline.ingest_document(document, overwrite=True)
        return reviewed, report

    # ------------------------------------------------------------------ #
    # Graph + queries
    # ------------------------------------------------------------------ #
    def graph(self) -> KnowledgeGraph:
        """Knowledge graph built from declared metadata only."""
        return build_graph(self._db.iter_documents())

    def stats(self) -> dict[str, object]:
        """Summarise the knowledge base for the CLI and doctor."""
        return {
            "documents": self._db.count_documents(),
            "chunks": self._db.count_chunks(),
            "categories": self._db.category_counts(),
            "sources": len(self._db.list_sources()),
            "versions": self.versions_count(),
            "candidates": self._db.count_candidates(),
        }


__all__ = ["KnowledgeManager"]
