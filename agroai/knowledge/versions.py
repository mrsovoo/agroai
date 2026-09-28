"""Knowledge versioning: a replaced document never erases what was known.

The roadmap requires that when an important document changes, **both** the old
and the new version are kept. This module is that bookkeeping:

* :class:`DocumentVersion` - one archived snapshot, with its hash and timestamp;
* :class:`VersionStore` - the interface (SQLite-backed and in-memory impls);
* :func:`changes_between` - an honest diff summary between two versions.

Ingestion archives the previous version automatically whenever content changes,
so a new FAO edition or a corrected rate leaves an auditable trail.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable

from agroai.knowledge.documents import KnowledgeDocument
from agroai.knowledge.metadata import content_hash

if TYPE_CHECKING:  # pragma: no cover - avoids an import cycle with database.py
    from agroai.database.database import Database


@dataclass(frozen=True)
class DocumentVersion:
    """One archived version of a knowledge document."""

    version_id: str
    document_id: str
    version: str
    title: str
    content_hash: str
    archived_at: str
    reason: str = ""
    source: str = ""
    category: str = "general"
    content: str = ""

    @property
    def char_count(self) -> int:
        return len(self.content)

    def to_dict(self) -> dict[str, Any]:
        return {
            "version_id": self.version_id,
            "document_id": self.document_id,
            "version": self.version,
            "title": self.title,
            "content_hash": self.content_hash,
            "archived_at": self.archived_at,
            "reason": self.reason,
            "source": self.source,
            "category": self.category,
        }


def version_of(document: KnowledgeDocument) -> DocumentVersion | None:
    """Return a document's *archived* version info, if it came from the archive."""
    archived_at = document.metadata.get("archived_at")
    if not archived_at:
        return None
    return DocumentVersion(
        version_id=str(document.metadata.get("version_id", "")),
        document_id=document.id,
        version=str(document.metadata.get("version", "1")),
        title=document.title,
        content_hash=document.content_hash,
        archived_at=str(archived_at),
        reason=str(document.metadata.get("archive_reason", "")),
        source=document.source,
        category=document.category,
        content=document.content,
    )


class VersionStore(ABC):
    """Interface for archiving and reading document versions."""

    @abstractmethod
    def archive(
        self,
        document: KnowledgeDocument,
        *,
        reason: str = "replaced",
        version: str | None = None,
    ) -> DocumentVersion:
        """Store one version of ``document``."""

    @abstractmethod
    def history(self, document_id: str) -> list[DocumentVersion]:
        """Return versions of a document, newest first."""

    @abstractmethod
    def count(self) -> int:
        """Total number of archived versions."""

    def latest(self, document_id: str) -> DocumentVersion | None:
        versions = self.history(document_id)
        return versions[0] if versions else None


def _to_version(document: KnowledgeDocument) -> DocumentVersion:
    info = version_of(document)
    if info is not None:
        return info
    return DocumentVersion(
        version_id="",
        document_id=document.id,
        version=str(document.metadata.get("version", "1")),
        title=document.title,
        content_hash=document.content_hash,
        archived_at=document.updated_at,
        source=document.source,
        category=document.category,
        content=document.content,
    )


class SQLiteVersionStore(VersionStore):
    """Version store backed by the local SQLite database (the default).

    The database is used structurally (duck-typed), so this module never imports
    :mod:`agroai.database` at runtime - that would create a circular import.
    """

    def __init__(self, database: "Database") -> None:
        self._db = database

    @property
    def database(self) -> "Database":
        return self._db

    def archive(
        self,
        document: KnowledgeDocument,
        *,
        reason: str = "replaced",
        version: str | None = None,
    ) -> DocumentVersion:
        self._db.add_document_version(document, reason=reason, version=version)
        return _to_version(document)

    def history(self, document_id: str) -> list[DocumentVersion]:
        return [_to_version(document) for document in self._db.list_document_versions(document_id)]

    def count(self) -> int:
        return self._db.count_document_versions()


class MemoryVersionStore(VersionStore):
    """In-process version store (tests, ephemeral sessions, dry runs)."""

    def __init__(self) -> None:
        self._versions: list[DocumentVersion] = []

    def archive(
        self,
        document: KnowledgeDocument,
        *,
        reason: str = "replaced",
        version: str | None = None,
    ) -> DocumentVersion:
        from agroai.knowledge.metadata import utcnow_iso

        stored = DocumentVersion(
            version_id=f"{document.id}@{version or '1'}",
            document_id=document.id,
            version=str(version or document.metadata.get("version", "1")),
            title=document.title,
            content_hash=document.content_hash,
            archived_at=utcnow_iso(),
            reason=reason,
            source=document.source,
            category=document.category,
            content=document.content,
        )
        self._versions.append(stored)
        return stored

    def history(self, document_id: str) -> list[DocumentVersion]:
        return [item for item in reversed(self._versions) if item.document_id == document_id]

    def count(self) -> int:
        return len(self._versions)


def changes_between(old: DocumentVersion, new: DocumentVersion) -> dict[str, Any]:
    """Summarise how two versions differ (hashes, sizes, title)."""
    return {
        "document_id": new.document_id,
        "same_content": old.content_hash == new.content_hash,
        "old_hash": old.content_hash,
        "new_hash": new.content_hash,
        "old_chars": old.char_count,
        "new_chars": new.char_count,
        "char_delta": new.char_count - old.char_count,
        "title_changed": old.title != new.title,
        "old_version": old.version,
        "new_version": new.version,
    }


def version_history_summary(versions: Iterable[DocumentVersion]) -> dict[str, Any]:
    """Summarise a version history for the CLI and doctor output."""
    versions = list(versions)
    return {
        "versions": len(versions),
        "latest": versions[0].to_dict() if versions else None,
        "oldest": versions[-1].to_dict() if versions else None,
        "content_hashes": [version.content_hash[:12] for version in versions],
    }


def content_digest(document: KnowledgeDocument) -> str:
    """Return a document's content digest (recomputed, for verification)."""
    return content_hash(document.title, document.content)


__all__ = [
    "DocumentVersion",
    "MemoryVersionStore",
    "SQLiteVersionStore",
    "VersionStore",
    "changes_between",
    "content_digest",
    "version_history_summary",
    "version_of",
]
