"""SQLite storage backend.

Design goals
------------
* **Scalable**: never load the whole knowledge base into memory. All reads are
  cursors that can be iterated lazily and all writes are batched.
* **Provenance**: documents, chunks and sources are separate tables joined by
  ids, so an answer can always cite its origin.
* **Portable**: only the Python standard library (``sqlite3``) is used.

When SQLite was compiled with FTS5 the ``chunks_fts`` virtual table provides
real full-text search; otherwise the retriever transparently falls back to a
``LIKE`` scan, so the project always works.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

from agroai.knowledge.documents import Chunk, KnowledgeDocument
from agroai.knowledge.metadata import utcnow_iso
from agroai.knowledge.sources import SourceRecord

_SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id               TEXT PRIMARY KEY,
    title            TEXT NOT NULL,
    content          TEXT NOT NULL,
    source           TEXT NOT NULL DEFAULT 'local',
    source_url       TEXT NOT NULL DEFAULT '',
    category         TEXT NOT NULL DEFAULT 'general',
    language         TEXT NOT NULL DEFAULT 'uz',
    publication_date TEXT NOT NULL DEFAULT '',
    author           TEXT NOT NULL DEFAULT '',
    content_hash     TEXT NOT NULL,
    path             TEXT NOT NULL DEFAULT '',
    metadata         TEXT NOT NULL DEFAULT '{}',
    created_at       TEXT NOT NULL,
    updated_at       TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_category ON documents(category);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source);

CREATE TABLE IF NOT EXISTS chunks (
    id           TEXT PRIMARY KEY,
    document_id  TEXT NOT NULL,
    chunk_index  INTEGER NOT NULL,
    content      TEXT NOT NULL,
    content_hash TEXT NOT NULL DEFAULT '',
    start_char   INTEGER NOT NULL DEFAULT 0,
    end_char     INTEGER NOT NULL DEFAULT 0,
    metadata     TEXT NOT NULL DEFAULT '{}',
    created_at   TEXT NOT NULL,
    FOREIGN KEY (document_id) REFERENCES documents(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);

CREATE TABLE IF NOT EXISTS sources (
    name             TEXT PRIMARY KEY,
    source_type      TEXT NOT NULL DEFAULT 'local',
    url              TEXT NOT NULL DEFAULT '',
    document_count   INTEGER NOT NULL DEFAULT 0,
    last_ingested_at TEXT NOT NULL DEFAULT '',
    metadata         TEXT NOT NULL DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

-- Knowledge versioning: every replaced document keeps its previous version so
-- a change (new FAO edition, corrected rate) never erases what was known.
CREATE TABLE IF NOT EXISTS document_versions (
    version_id    TEXT PRIMARY KEY,
    document_id   TEXT NOT NULL,
    version       TEXT NOT NULL DEFAULT '1',
    title         TEXT NOT NULL DEFAULT '',
    content       TEXT NOT NULL DEFAULT '',
    content_hash  TEXT NOT NULL,
    source        TEXT NOT NULL DEFAULT '',
    source_url    TEXT NOT NULL DEFAULT '',
    category      TEXT NOT NULL DEFAULT 'general',
    language      TEXT NOT NULL DEFAULT 'uz',
    metadata      TEXT NOT NULL DEFAULT '{}',
    archived_at   TEXT NOT NULL,
    reason        TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_versions_document ON document_versions(document_id);
CREATE INDEX IF NOT EXISTS idx_versions_hash ON document_versions(content_hash);

-- Knowledge learning: user reports and other unverified material live here as
-- candidates until a human verifies or rejects them. They are never promoted to
-- the knowledge base automatically.
CREATE TABLE IF NOT EXISTS knowledge_candidates (
    id             TEXT PRIMARY KEY,
    statement      TEXT NOT NULL,
    category       TEXT NOT NULL DEFAULT 'general',
    crop           TEXT NOT NULL DEFAULT '',
    animal         TEXT NOT NULL DEFAULT '',
    region         TEXT NOT NULL DEFAULT '',
    country        TEXT NOT NULL DEFAULT '',
    language       TEXT NOT NULL DEFAULT 'uz',
    source         TEXT NOT NULL DEFAULT '',
    source_url     TEXT NOT NULL DEFAULT '',
    author         TEXT NOT NULL DEFAULT '',
    reported_at    TEXT NOT NULL DEFAULT '',
    evidence_level TEXT NOT NULL DEFAULT 'observation',
    confidence     REAL NOT NULL DEFAULT 0.0,
    status         TEXT NOT NULL DEFAULT 'pending',
    review_notes   TEXT NOT NULL DEFAULT '',
    conflicts      TEXT NOT NULL DEFAULT '[]',
    metadata       TEXT NOT NULL DEFAULT '{}',
    created_at     TEXT NOT NULL,
    updated_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_candidates_status ON knowledge_candidates(status);

-- User memory: farm profile and observations, stored only with consent.
CREATE TABLE IF NOT EXISTS user_memory (
    id         TEXT PRIMARY KEY,
    user_id    TEXT NOT NULL,
    kind       TEXT NOT NULL DEFAULT 'profile',
    key        TEXT NOT NULL DEFAULT '',
    value      TEXT NOT NULL DEFAULT '',
    metadata   TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_user_memory_user ON user_memory(user_id, kind);
"""

_FTS_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    content,
    title,
    category,
    chunk_id UNINDEXED,
    document_id UNINDEXED,
    tokenize = 'unicode61'
);
"""


def _loads(raw: str) -> dict:
    try:
        value = json.loads(raw) if raw else {}
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


class Database:
    """Thin, explicit wrapper around a single SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._conn: sqlite3.Connection | None = None
        self.fts_enabled = False

    # ------------------------------------------------------------------ #
    # Connection lifecycle
    # ------------------------------------------------------------------ #
    @property
    def connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self.connect()
        assert self._conn is not None
        return self._conn

    def connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # check_same_thread=False: the HTTP server is threaded and hands the
        # connection across request threads. All access stays serialized by
        # the GIL plus short, single transactions; WAL keeps readers/writers
        # independent. (Cross-thread sqlite misuse was the previous crash.)
        conn = sqlite3.connect(str(self.path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.execute("PRAGMA synchronous = NORMAL")
        self._conn = conn
        return conn

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def __enter__(self) -> "Database":
        self.connect()
        self.initialize()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Schema
    # ------------------------------------------------------------------ #
    def initialize(self) -> None:
        conn = self.connection
        conn.executescript(_SCHEMA)
        try:
            conn.executescript(_FTS_SCHEMA)
            self.fts_enabled = True
        except sqlite3.OperationalError:
            # SQLite built without FTS5 -> retriever falls back to LIKE.
            self.fts_enabled = False
        conn.commit()

    def ping(self) -> bool:
        """Return True if the database answers a trivial query."""
        try:
            self.connection.execute("SELECT 1").fetchone()
            return True
        except sqlite3.Error:
            return False

    # ------------------------------------------------------------------ #
    # Writes
    # ------------------------------------------------------------------ #
    def upsert_document(self, document: KnowledgeDocument) -> None:
        conn = self.connection
        now = utcnow_iso()
        conn.execute(
            """
            INSERT INTO documents (
                id, title, content, source, source_url, category, language,
                publication_date, author, content_hash, path, metadata,
                created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                title=excluded.title,
                content=excluded.content,
                source=excluded.source,
                source_url=excluded.source_url,
                category=excluded.category,
                language=excluded.language,
                publication_date=excluded.publication_date,
                author=excluded.author,
                content_hash=excluded.content_hash,
                path=excluded.path,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            (
                document.id,
                document.title,
                document.content,
                document.source,
                document.source_url,
                document.category,
                document.language,
                document.publication_date,
                document.author,
                document.content_hash,
                document.path,
                json.dumps(document.metadata, ensure_ascii=False),
                document.created_at or now,
                now,
            ),
        )
        conn.commit()

    def delete_document(self, document_id: str) -> None:
        conn = self.connection
        conn.execute("DELETE FROM chunks WHERE document_id = ?", (document_id,))
        if self.fts_enabled:
            conn.execute("DELETE FROM chunks_fts WHERE document_id = ?", (document_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (document_id,))
        conn.commit()

    def add_chunks(self, chunks: Iterable[Chunk], title_map: dict[str, str] | None = None) -> int:
        """Insert chunks in one transaction. Returns the number inserted."""
        conn = self.connection
        title_map = title_map or {}
        rows = []
        fts_rows = []
        count = 0
        for chunk in chunks:
            rows.append(
                (
                    chunk.id,
                    chunk.document_id,
                    chunk.chunk_index,
                    chunk.content,
                    chunk.content_hash,
                    chunk.start_char,
                    chunk.end_char,
                    json.dumps(chunk.metadata, ensure_ascii=False),
                    chunk.created_at,
                )
            )
            if self.fts_enabled:
                fts_rows.append(
                    (
                        chunk.content,
                        title_map.get(chunk.document_id, ""),
                        str(chunk.metadata.get("category", "")),
                        chunk.id,
                        chunk.document_id,
                    )
                )
            count += 1

        conn.executemany(
            """
            INSERT OR REPLACE INTO chunks (
                id, document_id, chunk_index, content, content_hash,
                start_char, end_char, metadata, created_at
            ) VALUES (?,?,?,?,?,?,?,?,?)
            """,
            rows,
        )
        if self.fts_enabled and fts_rows:
            conn.executemany(
                "INSERT INTO chunks_fts (content, title, category, chunk_id, document_id) "
                "VALUES (?,?,?,?,?)",
                fts_rows,
            )
        conn.commit()
        return count

    def upsert_source(self, source: SourceRecord) -> None:
        conn = self.connection
        conn.execute(
            """
            INSERT INTO sources (name, source_type, url, document_count, last_ingested_at, metadata)
            VALUES (?,?,?,?,?,?)
            ON CONFLICT(name) DO UPDATE SET
                source_type=excluded.source_type,
                url=excluded.url,
                document_count=excluded.document_count,
                last_ingested_at=excluded.last_ingested_at,
                metadata=excluded.metadata
            """,
            (
                source.name,
                source.source_type,
                source.url,
                source.document_count,
                source.last_ingested_at or utcnow_iso(),
                json.dumps(source.metadata, ensure_ascii=False),
            ),
        )
        conn.commit()

    # ------------------------------------------------------------------ #
    # Knowledge versioning (old versions are never erased)
    # ------------------------------------------------------------------ #
    def add_document_version(
        self,
        document: KnowledgeDocument,
        *,
        reason: str = "replaced",
        version: str | None = None,
        archived_at: str | None = None,
    ) -> str:
        """Archive one version of a document. Returns the version id."""
        conn = self.connection
        stamp = archived_at or utcnow_iso()
        chosen_version = version or str(document.metadata.get("version", "1"))
        version_id = f"{document.id}@{chosen_version}#{stamp}"
        conn.execute(
            """
            INSERT OR REPLACE INTO document_versions (
                version_id, document_id, version, title, content, content_hash,
                source, source_url, category, language, metadata, archived_at, reason
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                version_id,
                document.id,
                chosen_version,
                document.title,
                document.content,
                document.content_hash,
                document.source,
                document.source_url,
                document.category,
                document.language,
                json.dumps(document.metadata, ensure_ascii=False),
                stamp,
                reason,
            ),
        )
        conn.commit()
        return version_id

    def list_document_versions(self, document_id: str | None = None) -> list[KnowledgeDocument]:
        """Yield archived versions (newest first) as documents with provenance."""
        if document_id:
            cursor = self.connection.execute(
                "SELECT * FROM document_versions WHERE document_id = ? ORDER BY archived_at DESC",
                (document_id,),
            )
        else:
            cursor = self.connection.execute(
                "SELECT * FROM document_versions ORDER BY archived_at DESC"
            )
        return [self._row_to_version(row) for row in cursor]

    def latest_document_version(self, document_id: str) -> KnowledgeDocument | None:
        versions = self.list_document_versions(document_id)
        return versions[0] if versions else None

    def count_document_versions(self) -> int:
        row = self.connection.execute("SELECT COUNT(*) AS c FROM document_versions").fetchone()
        return int(row["c"])

    @staticmethod
    def _row_to_version(row: sqlite3.Row) -> KnowledgeDocument:
        metadata = _loads(row["metadata"])
        metadata.update(
            {
                "archived_at": row["archived_at"],
                "archive_reason": row["reason"],
                "version_id": row["version_id"],
                "version": row["version"],
            }
        )
        return KnowledgeDocument(
            id=row["document_id"],
            title=row["title"],
            content=row["content"],
            source=row["source"],
            source_url=row["source_url"],
            category=row["category"],
            language=row["language"],
            content_hash=row["content_hash"],
            metadata=metadata,
        )

    # ------------------------------------------------------------------ #
    # Knowledge candidates (unverified material awaiting review)
    # ------------------------------------------------------------------ #
    def upsert_candidate(self, candidate: Any) -> None:
        from agroai.knowledge.candidates import KnowledgeCandidate  # local: no cycle

        if not isinstance(candidate, KnowledgeCandidate):
            raise TypeError("upsert_candidate expects a KnowledgeCandidate")
        conn = self.connection
        conn.execute(
            """
            INSERT INTO knowledge_candidates (
                id, statement, category, crop, animal, region, country, language,
                source, source_url, author, reported_at, evidence_level, confidence,
                status, review_notes, conflicts, metadata, created_at, updated_at
            ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                statement=excluded.statement,
                category=excluded.category,
                crop=excluded.crop,
                animal=excluded.animal,
                region=excluded.region,
                country=excluded.country,
                language=excluded.language,
                source=excluded.source,
                source_url=excluded.source_url,
                author=excluded.author,
                reported_at=excluded.reported_at,
                evidence_level=excluded.evidence_level,
                confidence=excluded.confidence,
                status=excluded.status,
                review_notes=excluded.review_notes,
                conflicts=excluded.conflicts,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            (
                candidate.id,
                candidate.statement,
                candidate.category,
                candidate.crop,
                candidate.animal,
                candidate.region,
                candidate.country,
                candidate.language,
                candidate.source,
                candidate.source_url,
                candidate.author,
                candidate.reported_at,
                candidate.evidence_level.value,
                candidate.confidence,
                candidate.status.value,
                candidate.review_notes,
                json.dumps(candidate.conflicts, ensure_ascii=False),
                json.dumps(candidate.metadata, ensure_ascii=False),
                candidate.created_at,
                candidate.updated_at,
            ),
        )
        conn.commit()

    def get_candidate(self, candidate_id: str) -> Any | None:
        row = self.connection.execute(
            "SELECT * FROM knowledge_candidates WHERE id = ?", (candidate_id,)
        ).fetchone()
        return self._row_to_candidate(row) if row else None

    def list_candidates(self, status: str | None = None, limit: int | None = None) -> list[Any]:
        sql = "SELECT * FROM knowledge_candidates"
        params: list[object] = []
        if status:
            sql += " WHERE status = ?"
            params.append(status)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        return [self._row_to_candidate(row) for row in self.connection.execute(sql, params)]

    def count_candidates(self, status: str | None = None) -> int:
        if status:
            row = self.connection.execute(
                "SELECT COUNT(*) AS c FROM knowledge_candidates WHERE status = ?", (status,)
            ).fetchone()
        else:
            row = self.connection.execute(
                "SELECT COUNT(*) AS c FROM knowledge_candidates"
            ).fetchone()
        return int(row["c"])

    def delete_candidate(self, candidate_id: str) -> None:
        conn = self.connection
        conn.execute("DELETE FROM knowledge_candidates WHERE id = ?", (candidate_id,))
        conn.commit()

    @staticmethod
    def _row_to_candidate(row: sqlite3.Row) -> Any:
        from agroai.knowledge.candidates import CandidateStatus, EvidenceLevel, KnowledgeCandidate

        return KnowledgeCandidate(
            id=row["id"],
            statement=row["statement"],
            category=row["category"],
            crop=row["crop"],
            animal=row["animal"],
            region=row["region"],
            country=row["country"],
            language=row["language"],
            source=row["source"],
            source_url=row["source_url"],
            author=row["author"],
            reported_at=row["reported_at"],
            evidence_level=EvidenceLevel(row["evidence_level"]),
            confidence=float(row["confidence"]),
            status=CandidateStatus(row["status"]),
            review_notes=row["review_notes"],
            conflicts=json.loads(row["conflicts"] or "[]"),
            metadata=_loads(row["metadata"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    # ------------------------------------------------------------------ #
    # User memory (stored only with explicit consent)
    # ------------------------------------------------------------------ #
    def set_user_memory(
        self,
        user_id: str,
        key: str,
        value: str,
        *,
        kind: str = "profile",
        metadata: dict | None = None,
    ) -> None:
        conn = self.connection
        now = utcnow_iso()
        entry_id = f"{user_id}:{kind}:{key}"
        conn.execute(
            """
            INSERT INTO user_memory (id, user_id, kind, key, value, metadata, created_at, updated_at)
            VALUES (?,?,?,?,?,?,?,?)
            ON CONFLICT(id) DO UPDATE SET
                value=excluded.value,
                metadata=excluded.metadata,
                updated_at=excluded.updated_at
            """,
            (
                entry_id,
                user_id,
                kind,
                key,
                value,
                json.dumps(metadata or {}, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.commit()

    def list_user_memory(self, user_id: str, kind: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT * FROM user_memory WHERE user_id = ?"
        params: list[object] = [user_id]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        sql += " ORDER BY kind, key"
        return [
            {
                "kind": row["kind"],
                "key": row["key"],
                "value": row["value"],
                "metadata": _loads(row["metadata"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in self.connection.execute(sql, params)
        ]

    def delete_user_memory(self, user_id: str, *, kind: str | None = None, key: str | None = None) -> int:
        """Forget memory entries. Returns how many rows were removed."""
        sql = "DELETE FROM user_memory WHERE user_id = ?"
        params: list[object] = [user_id]
        if kind:
            sql += " AND kind = ?"
            params.append(kind)
        if key:
            sql += " AND key = ?"
            params.append(key)
        cursor = self.connection.execute(sql, params)
        self.connection.commit()
        return int(cursor.rowcount or 0)

    # ------------------------------------------------------------------ #
    # Reads (streaming)
    # ------------------------------------------------------------------ #
    def get_document(self, document_id: str) -> KnowledgeDocument | None:
        row = self.connection.execute(
            "SELECT * FROM documents WHERE id = ?", (document_id,)
        ).fetchone()
        return self._row_to_document(row) if row else None

    def find_by_hash(self, digest: str) -> str | None:
        row = self.connection.execute(
            "SELECT id FROM documents WHERE content_hash = ? LIMIT 1", (digest,)
        ).fetchone()
        return row["id"] if row else None

    def iter_documents(self, category: str | None = None, limit: int | None = None) -> Iterator[KnowledgeDocument]:
        """Yield documents lazily (never materialises the full table)."""
        sql = "SELECT * FROM documents"
        params: list[object] = []
        if category:
            sql += " WHERE category = ?"
            params.append(category)
        sql += " ORDER BY created_at DESC"
        if limit is not None:
            sql += " LIMIT ?"
            params.append(limit)
        for row in self.connection.execute(sql, params):
            yield self._row_to_document(row)

    def iter_chunks(self, document_id: str | None = None) -> Iterator[Chunk]:
        if document_id:
            cursor = self.connection.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,),
            )
        else:
            cursor = self.connection.execute("SELECT * FROM chunks ORDER BY document_id, chunk_index")
        for row in cursor:
            yield self._row_to_chunk(row)

    def count_documents(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) AS c FROM documents").fetchone()["c"])

    def count_chunks(self) -> int:
        return int(self.connection.execute("SELECT COUNT(*) AS c FROM chunks").fetchone()["c"])

    def category_counts(self) -> dict[str, int]:
        rows = self.connection.execute(
            "SELECT category, COUNT(*) AS c FROM documents GROUP BY category ORDER BY c DESC"
        )
        return {row["category"]: int(row["c"]) for row in rows}

    def list_sources(self) -> list[SourceRecord]:
        rows = self.connection.execute(
            "SELECT * FROM sources ORDER BY document_count DESC, name ASC"
        )
        return [
            SourceRecord(
                name=row["name"],
                source_type=row["source_type"],
                url=row["url"],
                document_count=int(row["document_count"]),
                last_ingested_at=row["last_ingested_at"],
                metadata=_loads(row["metadata"]),
            )
            for row in rows
        ]

    # ------------------------------------------------------------------ #
    # Search
    # ------------------------------------------------------------------ #
    def search(
        self,
        query: str,
        top_k: int = 5,
        categories: Sequence[str] | None = None,
    ) -> list[tuple[Chunk, KnowledgeDocument, float]]:
        """Return ``(chunk, document, score)`` tuples, best match first.

        ``categories`` optionally scopes the search to knowledge categories (used
        by the skills layer to focus retrieval on the relevant domains).
        """
        query = (query or "").strip()
        if not query:
            return []
        wanted = [category for category in (categories or []) if category]
        if self.fts_enabled:
            results = self._search_fts(query, top_k, wanted)
            if results:
                return results
            # FTS may reject the query (e.g. only stop-words) -> LIKE fallback.
        return self._search_like(query, top_k, wanted)

    def _search_fts(
        self, query: str, top_k: int, categories: Sequence[str]
    ) -> list[tuple[Chunk, KnowledgeDocument, float]]:
        match_expr = _to_fts_query(query)
        if not match_expr:
            return []
        placeholders = ",".join("?" for _ in categories)
        category_clause = f" AND d.category IN ({placeholders})" if categories else ""
        try:
            rows = self.connection.execute(
                f"""
                SELECT c.*, bm25(chunks_fts) AS rank
                FROM chunks_fts
                JOIN chunks c ON c.id = chunks_fts.chunk_id
                JOIN documents d ON d.id = c.document_id
                WHERE chunks_fts MATCH ?{category_clause}
                ORDER BY rank
                LIMIT ?
                """,
                (match_expr, *categories, top_k),
            ).fetchall()
        except sqlite3.OperationalError:
            return []

        out: list[tuple[Chunk, KnowledgeDocument, float]] = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            doc = self.get_document(chunk.document_id)
            if doc is not None:
                # bm25 returns a negative score (lower = better); invert it.
                out.append((chunk, doc, -float(row["rank"] or 0.0)))
        return out

    def _search_like(
        self, query: str, top_k: int, categories: Sequence[str] = ()
    ) -> list[tuple[Chunk, KnowledgeDocument, float]]:
        terms = [t for t in query.lower().split() if t]
        if not terms:
            return []
        where = " OR ".join("LOWER(c.content) LIKE ?" for _ in terms)
        params: list[object] = [f"%{term}%" for term in terms]
        category_clause = ""
        if categories:
            placeholders = ",".join("?" for _ in categories)
            category_clause = f" AND d.category IN ({placeholders})"
            params.extend(categories)
        rows = self.connection.execute(
            f"SELECT c.* FROM chunks c JOIN documents d ON d.id = c.document_id "
            f"WHERE {where}{category_clause} LIMIT ?",
            (*params, top_k * 5),
        ).fetchall()

        scored: list[tuple[Chunk, KnowledgeDocument, float]] = []
        for row in rows:
            chunk = self._row_to_chunk(row)
            doc = self.get_document(chunk.document_id)
            if doc is None:
                continue
            lowered = chunk.content.lower()
            score = sum(lowered.count(term) for term in terms)
            scored.append((chunk, doc, float(score)))
        scored.sort(key=lambda item: item[2], reverse=True)
        return scored[:top_k]

    # ------------------------------------------------------------------ #
    # Row mapping
    # ------------------------------------------------------------------ #
    @staticmethod
    def _row_to_document(row: sqlite3.Row) -> KnowledgeDocument:
        return KnowledgeDocument(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            source=row["source"],
            source_url=row["source_url"],
            category=row["category"],
            language=row["language"],
            publication_date=row["publication_date"],
            author=row["author"],
            content_hash=row["content_hash"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            path=row["path"],
            metadata=_loads(row["metadata"]),
        )

    @staticmethod
    def _row_to_chunk(row: sqlite3.Row) -> Chunk:
        return Chunk(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=int(row["chunk_index"]),
            content=row["content"],
            content_hash=row["content_hash"],
            start_char=int(row["start_char"]),
            end_char=int(row["end_char"]),
            created_at=row["created_at"],
            metadata=_loads(row["metadata"]),
        )


def _to_fts_query(query: str) -> str:
    """Turn free text into a safe FTS5 OR query of quoted tokens."""
    tokens = [t for t in _tokenize(query) if len(t) > 1]
    if not tokens:
        return ""
    return " OR ".join(f'"{t}"' for t in tokens)


def _tokenize(text: str) -> list[str]:
    out: list[str] = []
    current: list[str] = []
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        else:
            if current:
                out.append("".join(current))
                current = []
    if current:
        out.append("".join(current))
    return out
