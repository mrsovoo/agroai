"""HTML ingestion.

Interface only: real HTML text extraction (boilerplate stripping, then the same
cleaning/chunking stages as text) is planned for a later ingestion phase. This
module registers the format so the pipeline and CLI can name it, and raises a
clear error instead of pretending to have read a page.

No parser is bundled on purpose: AgroAI Core stays standard-library only, and a
hand-rolled regex "HTML parser" would silently corrupt provenance. When the
phase arrives, use a real parser (e.g. the optional ``html`` extra) and replace
:class:`HtmlReader`.
"""

from __future__ import annotations

from agroai.ingestion.base import (
    DocumentReadError,
    DocumentReader,
    UnimplementedReader,
    UnsupportedFormatError,
)

PLANNED_PHASE = "a later ingestion phase"


def create_html_reader() -> DocumentReader:
    """Return the HTML reader registered with the format registry."""
    return UnimplementedReader("html", (".html", ".htm"), PLANNED_PHASE)


#: Convenience alias for callers that prefer a module-level object.
HTML_READER: DocumentReader = create_html_reader()

__all__ = [
    "HTML_READER",
    "PLANNED_PHASE",
    "DocumentReadError",
    "DocumentReader",
    "UnsupportedFormatError",
    "create_html_reader",
]
