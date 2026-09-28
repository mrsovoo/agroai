"""PDF ingestion - real extraction when the optional ``pypdf`` extra is installed.

The roadmap's most valuable document type is the PDF: agriculture and veterinary
textbooks, FAO/USDA reports, ministry guidance. Two requirements shape this
module:

1. **No fabricated text.** A PDF is read page by page; every character stored came
   from a page. Pages whose text cannot be extracted are counted and reported, and
   a PDF with *no* extractable text (a scan) raises with an explanation instead of
   storing an empty document.
2. **No mandatory dependency.** AgroAI Core stays standard-library only, so
   ``pypdf`` is an optional extra::

       pip install 'agroai[pdf]'

   Without it, :func:`create_pdf_reader` returns an
   :class:`~agroai.ingestion.base.UnimplementedReader` that raises a clear error -
   which is the honest behaviour, not a silent empty document.

Page counts go into the document metadata (``pages``, ``pages_processed``) so
:mod:`agroai.ingestion.verification` can report "600/600 pages processed" from
real numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agroai.config.settings import Settings
from agroai.ingestion.base import (
    DocumentReadError,
    DocumentReader,
    UnimplementedReader,
    UnsupportedFormatError,
)
from agroai.knowledge.documents import KnowledgeDocument
from agroai.knowledge.metadata import content_hash

PLANNED_PHASE = "the PDF ingestion phase (install the 'pdf' extra: pip install 'agroai[pdf]')"

#: Backends tried in order. ``pypdf`` is the current name; ``PyPDF2`` is the
#: legacy import that some environments still ship.
BACKENDS: tuple[str, ...] = ("pypdf", "PyPDF2")


def _load_backend() -> tuple[Any, str]:
    """Return ``(PdfReaderClass, backend_name)`` or ``(None, "")``."""
    for name in BACKENDS:
        try:
            module = __import__(name, fromlist=["PdfReader"])
        except ImportError:
            continue
        reader_class = getattr(module, "PdfReader", None)
        if reader_class is not None:
            return reader_class, name
    return None, ""


def is_pdf_available() -> bool:
    """True when a real PDF extraction backend is importable."""
    return _load_backend()[0] is not None


def pdf_backend() -> str:
    """Name of the active backend, or an empty string when none is installed."""
    return _load_backend()[1]


class PdfReader(DocumentReader):
    """Page-by-page PDF text extraction via the optional ``pypdf`` extra."""

    format_name = "pdf"
    extensions = (".pdf",)

    def read(self, path: Path, settings: Settings) -> KnowledgeDocument:
        reader_class, backend = _load_backend()
        if reader_class is None:
            raise NotImplementedError(
                "PDF ingestion needs the optional 'pdf' extra: "
                "pip install 'agroai[pdf]'. No text was extracted."
            )
        if not path.is_file():
            raise DocumentReadError(f"File not found: {path}")

        try:
            reader = reader_class(str(path))
            pages = list(reader.pages)
        except Exception as exc:  # noqa: BLE001 - the library raises many types
            raise DocumentReadError(
                f"Cannot open {path} as a PDF ({backend}): {type(exc).__name__}: {exc}"
            ) from exc

        texts: list[str] = []
        failed_pages: list[int] = []
        empty_pages: list[int] = []
        for number, page in enumerate(pages, start=1):
            try:
                extracted = page.extract_text() or ""
            except Exception:  # noqa: BLE001 - one bad page must not lose the book
                failed_pages.append(number)
                texts.append("")
                continue
            if not extracted.strip():
                empty_pages.append(number)
            texts.append(extracted)

        content = "\n\n".join(text.strip() for text in texts if text.strip()).strip()
        if not content:
            raise DocumentReadError(
                f"{path} contains no extractable text across {len(pages)} page(s). "
                "It is most likely a scanned/image PDF, which needs OCR - not "
                "implemented in this phase. Nothing was stored."
            )

        metadata = _pdf_metadata(reader)
        title = str(metadata.get("title") or "").strip() or self._title_from_text(path, content)
        digest = content_hash(content)
        pages_processed = len(pages) - len(failed_pages)

        return self._base_document(
            path,
            settings,
            content=content,
            digest=digest,
            title=title,
            extra_metadata={
                "pages": len(pages),
                "pages_processed": pages_processed,
                "pages_without_text": empty_pages,
                "pages_failed": failed_pages,
                "pdf_backend": backend,
                "pdf_metadata": metadata,
            },
            fields={},
        )


def _pdf_metadata(reader: Any) -> dict[str, Any]:
    """Read the PDF's own metadata dict (empty when absent)."""
    try:
        raw = reader.metadata or {}
    except Exception:  # noqa: BLE001
        return {}
    result: dict[str, Any] = {}
    for key in ("/Title", "/Author", "/Producer", "/Creator", "/Subject", "/CreationDate"):
        try:
            value = raw.get(key) if hasattr(raw, "get") else raw[key]
        except Exception:  # noqa: BLE001
            continue
        if value:
            result[key.lstrip("/").lower()] = str(value)
    return result


def create_pdf_reader() -> DocumentReader:
    """Return the PDF reader registered with the format registry.

    With the ``pdf`` extra installed this is a real, page-by-page reader.
    Without it, it is an :class:`UnimplementedReader` whose ``read`` raises a
    clear ``NotImplementedError`` instead of returning text it never read.
    """
    if is_pdf_available():
        return PdfReader()
    return UnimplementedReader("pdf", (".pdf",), PLANNED_PHASE)


#: Convenience alias for callers that prefer a module-level object.
PDF_READER: DocumentReader = create_pdf_reader()

__all__ = [
    "BACKENDS",
    "PDF_READER",
    "PLANNED_PHASE",
    "DocumentReadError",
    "DocumentReader",
    "PdfReader",
    "UnsupportedFormatError",
    "create_pdf_reader",
    "is_pdf_available",
    "pdf_backend",
]
