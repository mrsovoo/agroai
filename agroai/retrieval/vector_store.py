"""Vector stores: the semantic half of retrieval.

Three things are deliberately separated:

* :class:`VectorStore` - the storage/search interface (FAISS, Qdrant, Chroma,
  sqlite-vec all implement it later).
* :class:`InMemoryVectorStore` - a **real**, standard-library cosine-similarity
  store. It makes no claim about embeddings: you feed it real vectors and it
  returns real nearest neighbours. This is what makes the vector path testable
  today, with no dependency and no faked similarity.
* :class:`NullVectorStore` - what you get when nothing is configured. It says so
  instead of returning invented neighbours.

Embeddings themselves remain the missing piece (no embedding provider is
configured), so :class:`~agroai.retrieval.retriever.VectorRetriever` stays
unready until a real provider exists - and reports exactly that.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

from agroai.config.settings import Settings, get_settings
from agroai.retrieval.embeddings import EmbeddingProvider

BACKEND_MEMORY = "memory"
BACKEND_NONE = "none"

#: Backends named by the configuration that are still unimplemented.
KNOWN_UNIMPLEMENTED_BACKENDS = ("faiss", "qdrant", "chroma", "sqlite-vec", "sqlite_vec")


class VectorStoreNotConfigured(RuntimeError):
    """Raised when a vector store operation is attempted without a store."""


@dataclass(frozen=True)
class VectorRecord:
    """One embedded chunk, still linked to the chunk and document it came from.

    ``content`` is the chunk payload: storing it alongside the vector is normal
    for vector databases and keeps retrieval from depending on a second lookup.
    """

    id: str
    vector: tuple[float, ...]
    chunk_id: str = ""
    document_id: str = ""
    content: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def dimensions(self) -> int:
        return len(self.vector)


@dataclass
class VectorMatch:
    """A search result: the stored record plus its similarity score."""

    record: VectorRecord
    score: float


@dataclass
class VectorStoreHealth:
    backend: str
    ready: bool
    vectors: int
    dimensions: int
    detail: str


def cosine_similarity(left: Sequence[float], right: Sequence[float]) -> float:
    """Cosine similarity, ``0.0`` for degenerate input (never a fake match)."""
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


class VectorStore(ABC):
    """Interface for vector storage and nearest-neighbour search."""

    name: str = "base"

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensionality of the store (0 when unknown/empty)."""

    @abstractmethod
    def add(self, records: Iterable[VectorRecord]) -> int:
        """Insert or replace records. Returns the number stored."""

    @abstractmethod
    def search(self, vector: Sequence[float], top_k: int = 5) -> list[VectorMatch]:
        """Return the ``top_k`` most similar records, best first."""

    @abstractmethod
    def count(self) -> int:
        """Number of stored vectors."""

    @abstractmethod
    def clear(self) -> None:
        """Remove every stored vector."""

    @abstractmethod
    def health_check(self) -> VectorStoreHealth:
        """Report readiness and size."""

    def replace_document(self, document_id: str, records: Iterable[VectorRecord]) -> int:
        """Re-index one document: drop its old vectors, then add the new ones."""
        self.remove_document(document_id)
        return self.add(records)

    def remove_document(self, document_id: str) -> int:
        """Remove every vector belonging to ``document_id``. Returns the count."""
        raise NotImplementedError(
            f"{self.name} does not support per-document removal yet."
        )


class InMemoryVectorStore(VectorStore):
    """Real cosine-similarity store, standard library only.

    Exact (brute-force) search: perfectly correct, and honest about scaling -
    fine for tens of thousands of chunks, at which point a dedicated index
    (FAISS/Qdrant/sqlite-vec) should take over.
    """

    name = BACKEND_MEMORY

    def __init__(self, dimensions: int = 0) -> None:
        self._dimensions = max(0, dimensions)
        self._records: dict[str, VectorRecord] = {}

    @property
    def dimensions(self) -> int:
        if self._records:
            return next(iter(self._records.values())).dimensions
        return self._dimensions

    def add(self, records: Iterable[VectorRecord]) -> int:
        stored = 0
        for record in records:
            if not record.vector:
                raise ValueError(f"record {record.id!r} has an empty vector")
            if self._dimensions == 0:
                self._dimensions = record.dimensions
            elif record.dimensions != self._dimensions:
                raise ValueError(
                    f"dimension mismatch for {record.id!r}: expected "
                    f"{self._dimensions}, got {record.dimensions}. Refusing to mix "
                    "vectors from different embedding models."
                )
            self._records[record.id] = record
            stored += 1
        return stored

    def search(self, vector: Sequence[float], top_k: int = 5) -> list[VectorMatch]:
        if top_k <= 0 or not self._records or not vector:
            return []
        matches = [
            VectorMatch(record=record, score=cosine_similarity(vector, record.vector))
            for record in self._records.values()
        ]
        matches.sort(key=lambda match: match.score, reverse=True)
        return matches[:top_k]

    def count(self) -> int:
        return len(self._records)

    def clear(self) -> None:
        self._records.clear()

    def remove_document(self, document_id: str) -> int:
        doomed = [
            key for key, record in self._records.items() if record.document_id == document_id
        ]
        for key in doomed:
            del self._records[key]
        return len(doomed)

    def get(self, record_id: str) -> VectorRecord | None:
        return self._records.get(record_id)

    def health_check(self) -> VectorStoreHealth:
        return VectorStoreHealth(
            backend=self.name,
            ready=True,
            vectors=self.count(),
            dimensions=self.dimensions,
            detail=(
                f"In-memory exact (cosine) search, {self.count()} vectors. "
                "Nothing to search until embeddings are configured and indexed."
                if self.count() == 0
                else f"In-memory exact (cosine) search, {self.count()} vectors ready."
            ),
        )


class NullVectorStore(VectorStore):
    """Explicit "no vector store configured" backend."""

    name = BACKEND_NONE

    def __init__(self, requested: str = "none", detail: str = "") -> None:
        self._requested = requested
        self._detail = detail

    @property
    def dimensions(self) -> int:
        return 0

    def add(self, records: Iterable[VectorRecord]) -> int:
        raise VectorStoreNotConfigured(
            f"No vector store configured (requested: '{self._requested}'). "
            "Set AGROAI_VECTOR_STORE_BACKEND=memory or implement the backend."
        )

    def search(self, vector: Sequence[float], top_k: int = 5) -> list[VectorMatch]:
        # Empty is the truthful answer: there is no index to search.
        return []

    def count(self) -> int:
        return 0

    def clear(self) -> None:
        return None

    def health_check(self) -> VectorStoreHealth:
        detail = self._detail or (
            f"Vector store '{self._requested}' is not configured. Semantic "
            "retrieval stays unavailable; lexical SQLite FTS search is active."
        )
        return VectorStoreHealth(
            backend=self.name,
            ready=False,
            vectors=0,
            dimensions=0,
            detail=detail,
        )


def get_vector_store(
    settings: Settings | None = None,
    embeddings: EmbeddingProvider | None = None,
) -> VectorStore:
    """Return the configured vector store.

    ``none`` (the default) yields :class:`NullVectorStore`. ``memory`` yields the
    real in-memory store. Named third-party backends (faiss, qdrant, chroma,
    sqlite-vec) are reported as unimplemented rather than silently substituted.
    """
    settings = settings or get_settings()
    backend = (settings.vector_store_backend or BACKEND_NONE).lower()

    if backend in {"", BACKEND_NONE, "disabled"}:
        return NullVectorStore(BACKEND_NONE)
    if backend in {BACKEND_MEMORY, "in-memory", "inmemory"}:
        dimensions = embeddings.dimensions if embeddings is not None else 0
        return InMemoryVectorStore(dimensions=dimensions)
    if backend in KNOWN_UNIMPLEMENTED_BACKENDS:
        return NullVectorStore(
            backend,
            detail=(
                f"Vector store '{backend}' is not implemented yet (planned for the "
                "embeddings/vector phase). Use 'memory' to exercise the vector path "
                "locally, or keep 'none' for lexical search."
            ),
        )
    return NullVectorStore(
        backend,
        detail=(
            f"Unknown vector store backend '{backend}'. Valid values: none, memory. "
            "Lexical SQLite FTS search remains active."
        ),
    )


class VectorIndexer:
    """Embeds stored chunks and writes them into a vector store.

    It is real code with a dependency it cannot fake: without a configured
    embedding provider, indexing stops immediately with
    :class:`~agroai.retrieval.embeddings.EmbeddingsNotConfigured`.
    """

    def __init__(
        self,
        store: VectorStore,
        embeddings: EmbeddingProvider,
        database: Any | None = None,
    ) -> None:
        self._store = store
        self._embeddings = embeddings
        self._db = database

    @property
    def is_ready(self) -> bool:
        return self._embeddings.is_configured and not isinstance(self._store, NullVectorStore)

    def index_chunks(self, chunks: Iterable[Any], *, batch_size: int = 64) -> int:
        """Embed and store the given chunks. Returns the number indexed."""
        if isinstance(self._store, NullVectorStore):
            raise VectorStoreNotConfigured(
                "Indexing needs a real vector store; configure "
                "AGROAI_VECTOR_STORE_BACKEND (for example 'memory')."
            )

        indexed = 0
        batch: list[Any] = []
        for chunk in chunks:
            batch.append(chunk)
            if len(batch) >= batch_size:
                indexed += self._index_batch(batch)
                batch = []
        if batch:
            indexed += self._index_batch(batch)
        return indexed

    def index_database(self, *, batch_size: int = 64) -> int:
        """Index every chunk currently stored in the database."""
        if self._db is None:
            raise RuntimeError("VectorIndexer needs a database to index from.")
        return self.index_chunks(self._db.iter_chunks(), batch_size=batch_size)

    def _index_batch(self, batch: list[Any]) -> int:
        vectors = self._embeddings.embed([chunk.content for chunk in batch])
        if len(vectors) != len(batch):
            raise RuntimeError(
                f"embedding provider returned {len(vectors)} vectors for "
                f"{len(batch)} chunks"
            )
        records = [
            VectorRecord(
                id=f"vec:{chunk.id}",
                vector=tuple(vector),
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                metadata={**chunk.metadata, "chunk_index": chunk.chunk_index},
            )
            for chunk, vector in zip(batch, vectors)
        ]
        return self._store.add(records)


__all__ = [
    "BACKEND_MEMORY",
    "BACKEND_NONE",
    "KNOWN_UNIMPLEMENTED_BACKENDS",
    "InMemoryVectorStore",
    "NullVectorStore",
    "VectorIndexer",
    "VectorMatch",
    "VectorRecord",
    "VectorStore",
    "VectorStoreHealth",
    "VectorStoreNotConfigured",
    "cosine_similarity",
    "get_vector_store",
]
