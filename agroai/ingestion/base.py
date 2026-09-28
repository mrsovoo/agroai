"""Reader contracts, errors and the format registry.

A :class:`DocumentReader` turns one file into one
:class:`~agroai.knowledge.documents.KnowledgeDocument`. Concrete readers live in
:mod:`agroai.ingestion.text`, :mod:`agroai.ingestion.pdf` and
:mod:`agroai.ingestion.html`; the pipeline that drives them is
:mod:`agroai.ingestion.pipeline`.

Formats whose phase has not arrived are registered through
:class:`UnimplementedReader`, which raises a clear error instead of pretending
to extract text it cannot read.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from agroai.config.settings import Settings, get_settings
from agroai.knowledge.documents import KnowledgeDocument
from agroai.knowledge.metadata import normalize_date, utcnow_iso


class DocumentReadError(RuntimeError):
    """Raised when a file cannot be read as a knowledge document."""


class UnsupportedFormatError(DocumentReadError):
    """Raised when no reader is implemented for a file format."""


class DocumentReader(ABC):
    """Interface for reading one file into a knowledge document."""

    #: Extensions (lowercase, with dot) this reader handles, e.g. ``".txt"``.
    extensions: tuple[str, ...] = ()
    format_name: str = "base"
    #: Human readable note used in error messages, e.g. "planned for Phase 2".
    planned_phase: str = ""

    @abstractmethod
    def read(self, path: Path, settings: Settings) -> KnowledgeDocument:
        """Read ``path`` and return a document with full provenance."""

    def supports(self, path: Path) -> bool:
        return path.suffix.lower() in self.extensions

    def is_implemented(self) -> bool:
        return not isinstance(self, UnimplementedReader)

    # Shared helpers -----------------------------------------------------
    @staticmethod
    def _doc_id(path: Path, digest: str) -> str:
        """Stable id: file stem + short content hash (dedupes equal content)."""
        stem = "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem.lower())
        return f"doc_{stem}_{digest[:12]}"

    @staticmethod
    def _title_from_text(path: Path, text: str) -> str:
        """First markdown-style heading, else the first line, else the stem."""
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#"):
                return line.lstrip("#").strip() or path.stem
            if line:
                return line[:120]
        return path.stem

    @staticmethod
    def _guess_category(path: Path, settings: Settings) -> str:
        """Infer the category from the parent directory under knowledge_dir."""
        try:
            relative = path.resolve().relative_to(settings.knowledge_dir.resolve())
            parts = [p for p in relative.parts[:-1]]
            for part in parts:
                if part in settings.categories:
                    return part
        except ValueError:
            pass
        return "general"

    def _base_document(
        self,
        path: Path,
        settings: Settings,
        *,
        content: str,
        digest: str,
        title: str | None = None,
        extra_metadata: dict | None = None,
        fields: dict[str, str] | None = None,
    ) -> KnowledgeDocument:
        """Build a document, applying declared front-matter provenance.

        A document may declare its own origin (organization, source_type, URL,
        region, license, ...). Declared values win over the file name, because
        they are what makes a trust level meaningful - and nothing is inferred
        from a declared field that is absent.
        """
        fields = dict(fields or {})
        metadata = {
            "format": self.format_name,
            "file_name": path.name,
            "file_size": path.stat().st_size,
            "ingested_at": utcnow_iso(),
        }
        for key in (
            "source_type",
            "organization",
            "country",
            "region",
            "crop",
            "animal",
            "license",
            "version",
        ):
            value = fields.get(key, "").strip()
            if value:
                metadata[key] = value
        if extra_metadata:
            metadata.update(extra_metadata)

        return KnowledgeDocument(
            id=self._doc_id(path, digest),
            title=(title or fields.get("title") or path.stem).strip(),
            content=content,
            source=fields.get("organization") or path.name,
            source_url=fields.get("source_url", "").strip(),
            category=fields.get("category") or self._guess_category(path, settings),
            language=fields.get("language") or settings.default_language,
            publication_date=normalize_date(fields.get("publication_date", "")),
            author=fields.get("author", "").strip(),
            content_hash=digest,
            path=str(path.resolve()),
            metadata=metadata,
        )


class UnimplementedReader(DocumentReader):
    """Interface placeholder for a format whose phase has not arrived.

    Calling :meth:`read` raises a clear, honest error. The class exists so the
    format registry, CLI messaging and tests can already reference every
    planned format.
    """

    def __init__(self, format_name: str, extensions: tuple[str, ...], phase: str) -> None:
        self.format_name = format_name
        self.extensions = extensions
        self.planned_phase = phase

    def read(self, path: Path, settings: Settings) -> KnowledgeDocument:
        raise NotImplementedError(
            f"{self.format_name.upper()} ingestion is not implemented yet "
            f"(planned for {self.planned_phase}). Supported now: TXT, Markdown."
        )


def _default_readers() -> list[DocumentReader]:
    """Build the format registry.

    Imports are local so that readers can import this module for the base
    contract without creating an import cycle.
    """
    from agroai.ingestion.html import create_html_reader
    from agroai.ingestion.pdf import create_pdf_reader
    from agroai.ingestion.text import MarkdownReader, TxtReader

    return [
        TxtReader(),
        MarkdownReader(),
        create_pdf_reader(),
        create_html_reader(),
        UnimplementedReader("json", (".json",), "a later ingestion phase"),
        UnimplementedReader("csv", (".csv",), "a later ingestion phase"),
    ]


def supported_extensions() -> tuple[str, ...]:
    """Every extension the registry knows about (implemented or not)."""
    return tuple(ext for reader in _default_readers() for ext in reader.extensions)


def implemented_extensions() -> tuple[str, ...]:
    """Extensions that can actually be read today."""
    return tuple(
        ext for reader in _default_readers() if reader.is_implemented() for ext in reader.extensions
    )


def get_reader(path: Path) -> DocumentReader:
    """Return the reader that handles ``path``'s extension."""
    for reader in _default_readers():
        if reader.supports(path):
            return reader
    raise UnsupportedFormatError(
        f"No reader for '{path.suffix or path.name}'. Known extensions: "
        f"{', '.join(supported_extensions())}"
    )


def read_document(path: Path, settings: Settings | None = None) -> KnowledgeDocument:
    """Convenience one-shot reader used by the CLI and tests."""
    settings = settings or get_settings()
    return get_reader(Path(path)).read(Path(path), settings)


__all__ = [
    "DocumentReadError",
    "DocumentReader",
    "UnimplementedReader",
    "UnsupportedFormatError",
    "get_reader",
    "implemented_extensions",
    "read_document",
    "supported_extensions",
]
