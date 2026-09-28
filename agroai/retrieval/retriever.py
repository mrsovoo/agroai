"""Retrieval abstraction.

The retriever is deliberately independent of the LLM: it answers
``search(query, top_k)`` and nothing else. Vector backends (FAISS, Qdrant,
Chroma, sqlite-vec) plug in by implementing :class:`Retriever` and can be
combined with the vector store in :mod:`agroai.retrieval.vector_store`.

Two real implementations exist:

* :class:`SQLiteRetriever` - working lexical (FTS5/BM25) search, the default.
* :class:`VectorRetriever` - semantic search over a
  :class:`~agroai.retrieval.vector_store.VectorStore`. Its machinery is real and
  tested, but it stays *unready* until an embedding provider is configured, and
  says so instead of returning invented matches.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass

from agroai.config.settings import Settings, get_settings
from agroai.database.database import Database
from agroai.knowledge.documents import Chunk, RetrievedChunk
from agroai.retrieval.embeddings import EmbeddingProvider
from agroai.retrieval.vector_store import (
    NullVectorStore,
    VectorStore,
    get_vector_store,
)

logger = logging.getLogger(__name__)


class Retriever(ABC):
    """Interface for knowledge retrieval. Must not depend on any LLM."""

    name: str = "base"

    @abstractmethod
    def search(
        self,
        query: str,
        top_k: int = 5,
        categories: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        """Return the most relevant chunks for ``query``.

        ``categories`` optionally scopes the search to knowledge categories, as
        chosen by the skills layer. Implementations that cannot filter must still
        accept the argument (and may ignore it) so the agent stays uniform.
        """

    @abstractmethod
    def health_check(self) -> "RetrievalHealth":
        """Report readiness of the retrieval backend."""


@dataclass
class RetrievalHealth:
    backend: str
    ready: bool
    chunks: int
    documents: int
    search_mode: str
    detail: str


class SQLiteRetriever(Retriever):
    """Working lexical retriever using SQLite FTS5 (LIKE fallback)."""

    name = "sqlite"

    def __init__(self, database: Database) -> None:
        self._db = database

    def search(
        self,
        query: str,
        top_k: int = 5,
        categories: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0:
            return []
        rows = self._db.search(query, top_k=top_k, categories=categories)
        if not rows and categories:
            # A scoped search that found nothing must not hide the rest of the
            # knowledge base: fall back to an unscoped search instead.
            rows = self._db.search(query, top_k=top_k)
        return [
            RetrievedChunk(chunk=chunk, score=score, document=doc)
            for chunk, doc, score in rows
        ]

    def health_check(self) -> RetrievalHealth:
        chunks = self._db.count_chunks()
        documents = self._db.count_documents()
        mode = "fts5" if self._db.fts_enabled else "like"
        if documents == 0:
            detail = "Backend ready, but the knowledge base is empty. Run 'agroai ingest'."
        else:
            detail = "Backend ready."
        return RetrievalHealth(
            backend=self.name,
            ready=True,
            chunks=chunks,
            documents=documents,
            search_mode=mode,
            detail=detail,
        )


class VectorRetriever(Retriever):
    """Semantic retriever over a vector store.

    Ready only when all three pieces exist: a real (non-null) vector store, a
    configured embedding provider, and a database to resolve chunk/document
    provenance. Otherwise :meth:`search` returns nothing and
    :meth:`health_check` explains precisely what is missing.
    """

    name = "vector"

    def __init__(
        self,
        backend: str = "none",
        store: VectorStore | None = None,
        embeddings: EmbeddingProvider | None = None,
        database: Database | None = None,
    ) -> None:
        self._backend = backend
        self._store = store
        self._embeddings = embeddings
        self._db = database

    @property
    def is_ready(self) -> bool:
        return (
            self._store is not None
            and not isinstance(self._store, NullVectorStore)
            and self._embeddings is not None
            and self._embeddings.is_configured
            and self._store.count() > 0
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
        categories: Sequence[str] | None = None,
    ) -> list[RetrievedChunk]:
        if top_k <= 0 or not self.is_ready:
            return []
        assert self._store is not None and self._embeddings is not None
        vector = self._embeddings.embed_query(query)
        if not vector:
            return []
        results: list[RetrievedChunk] = []
        for match in self._store.search(vector, top_k=top_k):
            record = match.record
            chunk = Chunk(
                id=record.chunk_id or record.id,
                document_id=record.document_id,
                chunk_index=int(record.metadata.get("chunk_index", 0) or 0),
                content=record.content,
                metadata=dict(record.metadata),
            )
            document = (
                self._db.get_document(record.document_id) if self._db is not None else None
            )
            results.append(RetrievedChunk(chunk=chunk, score=match.score, document=document))
        if categories:
            scoped = [
                result
                for result in results
                if str(result.chunk.metadata.get("category", "")) in set(categories)
            ]
            # Never return nothing just because a skill scope was wrong.
            return scoped or results
        return results

    def health_check(self) -> RetrievalHealth:
        vectors = self._store.count() if self._store is not None else 0
        if self.is_ready:
            detail = f"Vector backend '{self._backend}' ready with {vectors} vectors."
        else:
            missing: list[str] = []
            if self._store is None or isinstance(self._store, NullVectorStore):
                missing.append("no vector store configured")
            if self._embeddings is None or not self._embeddings.is_configured:
                missing.append("no embedding provider configured")
            elif vectors == 0:
                missing.append("vector index is empty (nothing has been indexed)")
            detail = (
                f"Vector backend '{self._backend}' is not configured: "
                + ", ".join(missing)
                + ". Lexical SQLite FTS retrieval is active instead."
            )
        return RetrievalHealth(
            backend=self.name,
            ready=self.is_ready,
            chunks=vectors,
            documents=self._db.count_documents() if self._db is not None else 0,
            search_mode="vector" if self.is_ready else "none",
            detail=detail,
        )


def get_retriever(
    database: Database,
    settings: Settings | None = None,
    embeddings: EmbeddingProvider | None = None,
) -> Retriever:
    """Build the retriever for the configured backend.

    Lexical backends (and every unknown name) use the working SQLite retriever.
    A vector backend is only selected when a real embedding provider *and* the
    configured vector store are usable - otherwise falling back to lexical
    search is the honest choice, and the doctor command reports the gap.
    """
    settings = settings or get_settings()
    backend = (settings.retrieval_backend or "sqlite").lower()
    if backend in {"sqlite", "fts", "fts5", "lexical"}:
        return SQLiteRetriever(database)

    store = get_vector_store(settings, embeddings)
    usable_embeddings = embeddings is not None and embeddings.is_configured
    candidate = VectorRetriever(
        backend=backend, store=store, embeddings=embeddings, database=database
    )
    if candidate.is_ready:
        return candidate
    if usable_embeddings and isinstance(store, NullVectorStore):
        logger.warning("Embeddings are configured but the vector store is not; using lexical search.")
    # Unknown/vector backend without usable embeddings -> working lexical retriever.
    return SQLiteRetriever(database)


__all__ = [
    "RetrievalHealth",
    "Retriever",
    "SQLiteRetriever",
    "VectorRetriever",
    "get_retriever",
]

