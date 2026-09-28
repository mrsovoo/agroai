"""Ingestion pipeline: files -> documents -> chunks -> database.

Implemented formats today: TXT and Markdown. PDF becomes real as soon as the
optional ``pypdf`` extra is installed (``pip install 'agroai[pdf]'``); without it
:func:`~agroai.ingestion.pdf.create_pdf_reader` returns an honest
``UnimplementedReader``. HTML, JSON and CSV are registered the same way - AgroAI
never pretends to have extracted text it cannot read.

Two analysis modules sit alongside the pipeline:

* :mod:`agroai.ingestion.structure` - chapter/section detection (real, regex);
* :mod:`agroai.ingestion.verification` - proof that a document was actually
  chunked, indexed and covered.
"""

from __future__ import annotations

from agroai.ingestion.base import (
    DocumentReadError,
    DocumentReader,
    UnimplementedReader,
    UnsupportedFormatError,
    get_reader,
    implemented_extensions,
    read_document,
    supported_extensions,
)
from agroai.ingestion.html import HTML_READER, create_html_reader
from agroai.ingestion.pdf import (
    PDF_READER,
    PdfReader,
    create_pdf_reader,
    is_pdf_available,
    pdf_backend,
)
from agroai.ingestion.pipeline import (
    STAGES,
    IngestionPipeline,
    LanguageGuess,
    PreparedDocument,
    clean_text,
    detect_language,
)
from agroai.ingestion.structure import (
    DocumentStructure,
    Section,
    detect_sections,
    extract_tables,
    section_index,
    section_text,
)
from agroai.ingestion.text import MarkdownReader, TextLikeReader, TxtReader
from agroai.ingestion.verification import (
    BatchVerification,
    Check,
    DocumentVerification,
    coverage_report,
    covered_chars,
    verify_document,
    verify_ingestion,
)

__all__ = [
    "HTML_READER",
    "PDF_READER",
    "STAGES",
    "BatchVerification",
    "Check",
    "DocumentReadError",
    "DocumentReader",
    "DocumentStructure",
    "DocumentVerification",
    "IngestionPipeline",
    "LanguageGuess",
    "MarkdownReader",
    "PdfReader",
    "PreparedDocument",
    "Section",
    "TextLikeReader",
    "TxtReader",
    "UnimplementedReader",
    "UnsupportedFormatError",
    "clean_text",
    "coverage_report",
    "covered_chars",
    "create_html_reader",
    "create_pdf_reader",
    "detect_language",
    "detect_sections",
    "extract_tables",
    "get_reader",
    "implemented_extensions",
    "is_pdf_available",
    "pdf_backend",
    "read_document",
    "section_index",
    "section_text",
    "supported_extensions",
    "verify_document",
    "verify_ingestion",
]
