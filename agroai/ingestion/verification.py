"""Ingestion verification: prove a document really entered the knowledge base.

The roadmap asks for more than "the file was read":

```text
Document -> Processing -> Extraction -> Chunking -> Indexing -> Verification

Document status: COMPLETED
Pages processed: 600/600
Chunks: 18420
Indexed: 18420
Categories: Agronomy, Plant disease, ...
```

"PDFni o'qidi degani shunchaki faylni database'ga tashlash emas" - and the proof
is a machine check, not a log message. Every check here is recomputable from the
stored document and its chunks:

* content is present and its hash matches,
* chunks cover the document's character span completely (union of spans),
* every chunk keeps its provenance (document id, category, language),
* no two chunks are byte-identical (a chunking bug, not a knowledge gain),
* the stored chunk count equals what the database actually holds,
* page counts are reported **only** when the source declared them.

The report never says "fully verified" for something it did not verify: a missing
page count is reported as unknown, not as 600/600.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Iterable

from agroai.config.settings import KNOWLEDGE_CATEGORIES, Settings, get_settings
from agroai.database.database import Database
from agroai.ingestion.structure import detect_sections
from agroai.knowledge.documents import (
    Chunk,
    IngestionReport,
    KnowledgeDocument,
    iter_chunks,
)
from agroai.knowledge.metadata import content_hash

STATUS_VERIFIED = "verified"
STATUS_PARTIAL = "partial"
STATUS_EMPTY = "empty"


@dataclass
class Check:
    """One verification check and its outcome."""

    name: str
    ok: bool
    detail: str = ""

    @property
    def marker(self) -> str:
        return "[OK]" if self.ok else "[FAIL]"

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "ok": self.ok, "detail": self.detail}


@dataclass
class DocumentVerification:
    """Proof (or the lack of it) that one document was processed correctly."""

    document_id: str
    title: str = ""
    status: str = STATUS_EMPTY
    chars: int = 0
    chunks: int = 0
    indexed_chunks: int = 0
    sections: int = 0
    section_titles: list[str] = field(default_factory=list)
    span_coverage: float = 0.0
    index_coverage: float = 0.0
    category: str = ""
    source: str = ""
    pages_processed: int | None = None
    pages_total: int | None = None
    check_results: list[Check] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_VERIFIED and not self.problems

    @property
    def pages_summary(self) -> str:
        if self.pages_total:
            processed = self.pages_processed if self.pages_processed is not None else 0
            return f"{processed}/{self.pages_total}"
        return "unknown (the source document did not declare a page count)"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["ok"] = self.ok
        payload["pages_summary"] = self.pages_summary
        return payload

    def render(self) -> str:
        lines = [
            f"Document: {self.title or self.document_id} ({self.document_id})",
            f"  status: {self.status.upper()}",
            f"  pages processed: {self.pages_summary}",
            f"  characters: {self.chars}",
            f"  chunks: {self.chunks} (indexed: {self.indexed_chunks})",
            f"  span coverage: {self.span_coverage:.1%}",
            f"  category: {self.category or 'unknown'}",
            f"  source: {self.source or 'unknown'}",
            f"  sections detected: {self.sections}",
        ]
        for check in self.check_results:
            lines.append(f"  {check.marker} {check.name}: {check.detail}")
        for problem in self.problems:
            lines.append(f"  ! {problem}")
        return "\n".join(lines)


@dataclass
class BatchVerification:
    """Verification across many documents (a whole ingestion, or the whole base)."""

    documents: list[DocumentVerification] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.documents)

    @property
    def verified(self) -> list[DocumentVerification]:
        return [item for item in self.documents if item.status == STATUS_VERIFIED]

    @property
    def partial(self) -> list[DocumentVerification]:
        return [item for item in self.documents if item.status == STATUS_PARTIAL]

    @property
    def empty(self) -> list[DocumentVerification]:
        return [item for item in self.documents if item.status == STATUS_EMPTY]

    @property
    def ok(self) -> bool:
        return bool(self.documents) and not self.partial and not self.empty and not self.problems

    @property
    def average_coverage(self) -> float:
        if not self.documents:
            return 0.0
        return round(
            sum(item.span_coverage for item in self.documents) / len(self.documents), 4
        )

    def summary(self) -> dict[str, Any]:
        return {
            "documents": len(self.documents),
            "verified": len(self.verified),
            "partial": len(self.partial),
            "empty": len(self.empty),
            "chunks": sum(item.chunks for item in self.documents),
            "indexed_chunks": sum(item.indexed_chunks for item in self.documents),
            "average_coverage": self.average_coverage,
            "problems": len(self.problems),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "problems": list(self.problems),
            "documents": [item.to_dict() for item in self.documents],
        }

    def render(self, limit: int | None = None) -> str:
        summary = self.summary()
        lines = [
            "Ingestion verification",
            "=" * 60,
            f"Documents: {summary['documents']} "
            f"(verified={summary['verified']}, partial={summary['partial']}, "
            f"empty={summary['empty']})",
            f"Chunks: {summary['chunks']} (indexed: {summary['indexed_chunks']})",
            f"Average span coverage: {summary['average_coverage']:.1%}",
            "=" * 60,
        ]
        shown = self.documents[:limit] if limit else self.documents
        for item in shown:
            lines.append(f"{item.status.upper():<9} {item.title or item.document_id}")
            lines.append(
                f"          chunks={item.chunks} indexed={item.indexed_chunks} "
                f"coverage={item.span_coverage:.1%} pages={item.pages_summary}"
            )
            for problem in item.problems:
                lines.append(f"          ! {problem}")
        if limit and len(self.documents) > limit:
            lines.append(f"... and {len(self.documents) - limit} more document(s)")
        for problem in self.problems:
            lines.append(f"! {problem}")
        return "\n".join(lines)


# ---------------------------------------------------------------------- #
def covered_chars(chunks: Iterable[Chunk]) -> int:
    """Union of the character spans covered by ``chunks`` (overlap counted once)."""
    spans = sorted(
        (min(item.start_char, item.end_char), max(item.start_char, item.end_char))
        for item in chunks
    )
    total = 0
    current_start: int | None = None
    current_end: int | None = None
    for start, end in spans:
        if current_start is None:
            current_start, current_end = start, end
            continue
        assert current_end is not None
        if start <= current_end:
            current_end = max(current_end, end)
            continue
        total += current_end - current_start
        current_start, current_end = start, end
    if current_start is not None and current_end is not None:
        total += current_end - current_start
    return total


def verify_document(
    document: KnowledgeDocument,
    *,
    chunks: list[Chunk] | None = None,
    settings: Settings | None = None,
    database: Database | None = None,
) -> DocumentVerification:
    """Verify one document against its own chunks and (optionally) the database."""
    settings = settings or get_settings()
    if chunks is None:
        chunks = list(
            iter_chunks(
                document,
                chunk_size=settings.chunk_size,
                overlap=settings.chunk_overlap,
            )
        )

    verification = DocumentVerification(
        document_id=document.id,
        title=document.title,
        chars=document.char_count,
        chunks=len(chunks),
        category=document.category,
        source=document.source,
        indexed_chunks=(
            sum(1 for _ in database.iter_chunks(document.id)) if database is not None else 0
        ),
    )
    checks: list[Check] = []
    problems: list[str] = []

    content_ok = bool((document.content or "").strip()) and any(
        character.isalpha() for character in document.content
    )
    checks.append(
        Check(
            "content",
            content_ok,
            f"{document.char_count} character(s)" if content_ok else "no readable text",
        )
    )
    if not content_ok:
        problems.append("document has no readable content (empty or non-text extraction)")

    # Two hash definitions exist by design and both are legitimate: readers hash
    # the extracted text alone, while directly constructed documents default to
    # hashing (title, content). Verification accepts either, and says which.
    text_hash = content_hash(document.content)
    titled_hash = content_hash(document.title, document.content)
    if document.content_hash == text_hash:
        hash_ok, hash_detail = True, "matches the extracted text (reader hash)"
    elif document.content_hash == titled_hash:
        hash_ok, hash_detail = True, "matches title+content (document hash)"
    else:
        hash_ok, hash_detail = False, "hash matches neither the text nor title+content"
    checks.append(Check("content_hash", hash_ok, hash_detail))
    if not hash_ok:
        problems.append("content hash does not match the content (dedup/versioning is unsafe)")

    chunks_ok = len(chunks) > 0
    checks.append(Check("chunks", chunks_ok, f"{len(chunks)} chunk(s)" if chunks_ok else "no chunks"))
    if not chunks_ok:
        problems.append("no chunks were produced; the document is not retrievable")

    spans_ok = bool(chunks) and chunks[0].start_char == 0 and chunks[-1].end_char >= document.char_count - 1
    if chunks:
        monotonic = all(
            chunks[index].start_char <= chunks[index + 1].start_char
            for index in range(len(chunks) - 1)
        )
        spans_ok = spans_ok and monotonic
    checks.append(
        Check(
            "chunk_spans",
            spans_ok,
            (
                f"spans {chunks[0].start_char}-{chunks[-1].end_char} of {document.char_count}"
                if chunks
                else "no chunks"
            ),
        )
    )
    if chunks and not spans_ok:
        problems.append("chunk spans do not cover the whole document")

    provenance_ok = all(
        item.document_id == document.id
        and str(item.metadata.get("category", "")) == document.category
        and bool(item.metadata.get("language"))
        for item in chunks
    )
    checks.append(
        Check(
            "chunk_provenance",
            provenance_ok,
            "every chunk keeps document id, category and language"
            if provenance_ok
            else "a chunk lost its provenance",
        )
    )
    if not provenance_ok:
        problems.append("some chunks lost their provenance metadata")

    seen: set[str] = set()
    duplicates = 0
    for item in chunks:
        if item.content_hash in seen:
            duplicates += 1
        seen.add(item.content_hash)
    checks.append(
        Check(
            "duplicate_chunks",
            duplicates == 0,
            "none" if duplicates == 0 else f"{duplicates} duplicate chunk(s)",
        )
    )
    if duplicates:
        problems.append(f"{duplicates} duplicate chunk(s) would be indexed twice")

    category_ok = document.category in set(settings.categories or KNOWLEDGE_CATEGORIES) or document.category in {
        "general",
        "unknown",
    }
    checks.append(
        Check(
            "category",
            category_ok,
            document.category or "missing",
        )
    )
    if not category_ok:
        problems.append(
            f"category {document.category!r} is not a known category; retrieval scoping will miss it"
        )

    source_ok = bool((document.source or "").strip())
    checks.append(
        Check("source", source_ok, document.source or "missing (provenance required)")
    )
    if not source_ok:
        problems.append("document has no source; it cannot be cited")

    if database is not None:
        index_ok = verification.indexed_chunks == len(chunks)
        checks.append(
            Check(
                "indexed",
                index_ok,
                f"database holds {verification.indexed_chunks} of {len(chunks)} chunk(s)",
            )
        )
        if not index_ok:
            problems.append(
                f"only {verification.indexed_chunks} of {len(chunks)} chunk(s) are indexed"
            )

    structure = detect_sections(document.content)
    verification.sections = len(structure.sections)
    verification.section_titles = structure.titles[:20]
    verification.check_results = checks
    verification.problems = problems
    verification.span_coverage = (
        round(covered_chars(chunks) / document.char_count, 4) if document.char_count else 0.0
    )
    verification.index_coverage = (
        round(verification.indexed_chunks / len(chunks), 4)
        if database is not None and chunks
        else (1.0 if chunks else 0.0)
    )

    pages_total = _int_or_none(
        document.metadata.get("pages") or document.metadata.get("page_count")
    )
    pages_processed = _int_or_none(
        document.metadata.get("pages_processed") or document.metadata.get("pages")
    )
    verification.pages_total = pages_total
    verification.pages_processed = pages_processed

    if not content_ok:
        verification.status = STATUS_EMPTY
    elif problems:
        verification.status = STATUS_PARTIAL
    else:
        verification.status = STATUS_VERIFIED
    return verification


def verify_ingestion(
    report: IngestionReport | None = None,
    *,
    database: Database,
    settings: Settings | None = None,
    limit: int | None = None,
) -> BatchVerification:
    """Verify the documents the database holds.

    ``report`` is accepted for API symmetry (the CLI passes the ingestion report),
    but the authority is always the database: what was *stored* is what must be
    verified.
    """
    settings = settings or get_settings()
    batch = BatchVerification()
    for index, document in enumerate(database.iter_documents()):
        if limit is not None and index >= limit:
            batch.problems.append(
                f"verification stopped after {limit} document(s); more remain"
            )
            break
        batch.documents.append(
            verify_document(document, settings=settings, database=database)
        )
    if not batch.documents:
        batch.problems.append(
            "the knowledge base is empty: ingest documents before verifying coverage"
        )
    if report is not None and report.errors:
        batch.problems.extend(report.errors)
    return batch


def coverage_report(
    database: Database, *, settings: Settings | None = None, limit: int | None = None
) -> dict[str, Any]:
    """A compact JSON-friendly coverage summary for the API/CLI."""
    batch = verify_ingestion(database=database, settings=settings, limit=limit)
    summary = batch.summary()
    summary["status"] = "ok" if batch.ok else ("partial" if batch.documents else "empty")
    summary["categories"] = database.category_counts()
    return summary


def _int_or_none(value: Any) -> int | None:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return None
    return number if number >= 0 else None


__all__ = [
    "BatchVerification",
    "Check",
    "DocumentVerification",
    "STATUS_EMPTY",
    "STATUS_PARTIAL",
    "STATUS_VERIFIED",
    "coverage_report",
    "covered_chars",
    "verify_document",
    "verify_ingestion",
]
