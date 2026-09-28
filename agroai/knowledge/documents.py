"""Document and chunk models: the retrievable units of knowledge.

These dataclasses are the *contract* between ingestion, storage, retrieval and
the agent. They intentionally carry full provenance (source, URL, author, hash)
so that provenance is never lost during chunking or embedding.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from agroai.knowledge.metadata import (
    DocumentMetadata,
    TrustLevel,
    content_hash,
    infer_trust_level,
    utcnow_iso,
)


@dataclass
class Chunk:
    """A retrievable slice of a document.

    ``document_id`` links every chunk back to its parent ``KnowledgeDocument``
    so provenance survives chunking.
    """

    id: str
    document_id: str
    chunk_index: int
    content: str
    content_hash: str = ""
    start_char: int = 0
    end_char: int = 0
    created_at: str = field(default_factory=utcnow_iso)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.content)

    @property
    def char_count(self) -> int:
        return len(self.content)


@dataclass
class KnowledgeDocument:
    """A single unit of knowledge, with mandatory provenance metadata."""

    id: str
    title: str
    content: str
    source: str = "local"
    source_url: str = ""
    category: str = "general"
    language: str = "uz"
    publication_date: str = ""
    author: str = ""
    content_hash: str = ""
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    # Original filesystem path (if imported from a file), kept for re-ingestion.
    path: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.content_hash:
            self.content_hash = content_hash(self.title, self.content)

    @property
    def char_count(self) -> int:
        return len(self.content)

    @property
    def word_count(self) -> int:
        return len(self.content.split())

    @property
    def trust_level(self) -> TrustLevel:
        """Trust level recorded by ingestion, or inferred from the source."""
        raw = self.metadata.get("trust_level")
        if raw:
            try:
                return TrustLevel(str(raw))
            except ValueError:
                return TrustLevel.UNKNOWN
        return infer_trust_level(self.source, str(self.metadata.get("source_type", "")))

    def organization(self) -> str:
        return str(self.metadata.get("organization") or self.source)

    def region(self) -> str:
        return str(self.metadata.get("region", ""))

    def country(self) -> str:
        return str(self.metadata.get("country", ""))

    def to_document_metadata(self) -> DocumentMetadata:
        """Project this document onto the normalized metadata schema."""
        return DocumentMetadata(
            title=self.title,
            author=self.author,
            organization=self.organization(),
            source_url=self.source_url,
            publication_date=self.publication_date,
            language=self.language,
            country=self.country(),
            region=self.region(),
            category=self.category,
            license=str(self.metadata.get("license", "")),
            content_hash=self.content_hash,
            version=str(self.metadata.get("version", "1")),
            trust_level=self.trust_level,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "source": self.source,
            "source_url": self.source_url,
            "category": self.category,
            "language": self.language,
            "publication_date": self.publication_date,
            "author": self.author,
            "content_hash": self.content_hash,
            "trust_level": self.trust_level.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "path": self.path,
            "char_count": self.char_count,
            "metadata": dict(self.metadata),
        }


@dataclass
class RetrievedChunk:
    """A chunk returned by a retriever, together with its score and document."""

    chunk: Chunk
    score: float
    document: KnowledgeDocument | None = None

    @property
    def source_line(self) -> str:
        doc = self.document
        if doc is None:
            return f"document {self.chunk.document_id}"
        title = doc.title or doc.id
        parts = [title]
        if doc.source:
            parts.append(f"source: {doc.source}")
        if doc.source_url:
            parts.append(doc.source_url)
        return " | ".join(parts)


@dataclass
class IngestionReport:
    """Outcome of ingesting one or more documents."""

    documents_seen: int = 0
    documents_ingested: int = 0
    documents_skipped: int = 0
    documents_failed: int = 0
    chunks_created: int = 0
    #: Previous versions archived because the content changed (or duplicated).
    versions_archived: int = 0
    errors: list[str] = field(default_factory=list)

    def merge(self, other: "IngestionReport") -> "IngestionReport":
        self.documents_seen += other.documents_seen
        self.documents_ingested += other.documents_ingested
        self.documents_skipped += other.documents_skipped
        self.documents_failed += other.documents_failed
        self.chunks_created += other.chunks_created
        self.versions_archived += other.versions_archived
        self.errors.extend(other.errors)
        return self


def chunk_text(
    text: str,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> list[tuple[int, int, str]]:
    """Split ``text`` into overlapping character windows.

    Returns a list of ``(start, end, content)`` tuples. The strategy is a
    simple sliding window with paragraph preference; it is deliberately
    streaming-friendly so it can be applied to very large documents without
    holding more than one window in memory beyond the source text.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    overlap = max(0, min(overlap, chunk_size - 1))

    text = text or ""
    if not text.strip():
        return []
    if len(text) <= chunk_size:
        return [(0, len(text), text)]

    chunks: list[tuple[int, int, str]] = []
    start = 0
    length = len(text)
    while start < length:
        end = min(start + chunk_size, length)
        # Prefer to break on a paragraph/sentence boundary near the end.
        if end < length:
            window = text[start:end]
            for sep in ("\n\n", "\n", ". "):
                idx = window.rfind(sep)
                if idx > chunk_size // 2:
                    end = start + idx + len(sep)
                    break
        content = text[start:end]
        chunks.append((start, end, content))
        if end >= length:
            break
        start = max(end - overlap, start + 1)
    return chunks


def iter_chunks(
    document: KnowledgeDocument,
    *,
    chunk_size: int = 800,
    overlap: int = 120,
) -> Iterable[Chunk]:
    """Yield :class:`Chunk` objects for a document without loading extras.

    Chunk metadata merges the document's own metadata (format, file name, ...)
    with category/language so provenance survives chunking.
    """
    base_metadata: dict[str, Any] = {
        "category": document.category,
        "language": document.language,
        "trust_level": document.trust_level.value,
        **document.metadata,
    }
    for index, (start, end, content) in enumerate(
        chunk_text(document.content, chunk_size=chunk_size, overlap=overlap)
    ):
        yield Chunk(
            id=f"{document.id}:{index}",
            document_id=document.id,
            chunk_index=index,
            content=content,
            start_char=start,
            end_char=end,
            metadata=dict(base_metadata),
        )


__all__ = [
    "Chunk",
    "IngestionReport",
    "KnowledgeDocument",
    "RetrievedChunk",
    "chunk_text",
    "iter_chunks",
]
