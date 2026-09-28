"""Tests for the vector store: real cosine search plus honest null behaviour."""

from __future__ import annotations

import math

import pytest

from agroai.knowledge.documents import Chunk, KnowledgeDocument
from agroai.retrieval.embeddings import EmbeddingsNotConfigured, NullEmbeddingProvider
from agroai.retrieval.retriever import VectorRetriever
from agroai.retrieval.vector_store import (
    InMemoryVectorStore,
    NullVectorStore,
    VectorIndexer,
    VectorRecord,
    VectorStoreNotConfigured,
    cosine_similarity,
    get_vector_store,
)


class StubEmbedder:
    """Deterministic 3-dimensional embedder used to exercise the real path."""

    name = "stub"
    _TABLE = {"tuproq": (1.0, 0.0, 0.0), "pomidor": (0.0, 1.0, 0.0), "sug'orish": (0.0, 0.0, 1.0)}

    def __init__(self, dimensions: int = 3) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def is_configured(self) -> bool:
        return True

    def _vector(self, text: str) -> list[float]:
        for key, vector in self._TABLE.items():
            if key in text.lower():
                return list(vector)
        return [0.1, 0.1, 0.1]

    def embed(self, texts):
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)

    def health_check(self):
        raise NotImplementedError


# --------------------------------------------------------------------- #
# Cosine similarity
# --------------------------------------------------------------------- #
def test_cosine_similarity_basics() -> None:
    assert cosine_similarity([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine_similarity([1, 0], [2, 0]) == pytest.approx(1.0)
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)
    # Degenerate input never produces a fake match.
    assert cosine_similarity([], [1]) == 0.0
    assert cosine_similarity([1, 2], [1]) == 0.0
    assert cosine_similarity([0, 0], [1, 1]) == 0.0
    assert cosine_similarity([1, 1], [1, 1]) == pytest.approx(1.0 / math.sqrt(2) * math.sqrt(2))


# --------------------------------------------------------------------- #
# In-memory store
# --------------------------------------------------------------------- #
def test_in_memory_store_adds_searches_and_ranks() -> None:
    store = InMemoryVectorStore()
    stored = store.add(
        [
            VectorRecord(id="v1", vector=(1.0, 0.0), chunk_id="c1", document_id="d1", content="tuproq"),
            VectorRecord(id="v2", vector=(0.0, 1.0), chunk_id="c2", document_id="d2", content="pomidor"),
        ]
    )
    assert stored == 2
    assert store.count() == 2
    assert store.dimensions == 2

    matches = store.search((0.9, 0.1), top_k=2)
    assert [match.record.id for match in matches] == ["v1", "v2"]
    assert matches[0].score > matches[1].score

    assert store.search((1.0, 0.0), top_k=0) == []
    assert store.search([], top_k=2) == []


def test_in_memory_store_rejects_mixed_dimensions_and_empty_vectors() -> None:
    store = InMemoryVectorStore()
    store.add([VectorRecord(id="v1", vector=(1.0, 0.0))])
    with pytest.raises(ValueError, match="dimension mismatch"):
        store.add([VectorRecord(id="v2", vector=(1.0, 0.0, 0.0))])
    with pytest.raises(ValueError, match="empty vector"):
        store.add([VectorRecord(id="v3", vector=())])


def test_in_memory_store_removal_and_clear() -> None:
    store = InMemoryVectorStore()
    store.add(
        [
            VectorRecord(id="v1", vector=(1.0,), document_id="d1"),
            VectorRecord(id="v2", vector=(1.0,), document_id="d1"),
            VectorRecord(id="v3", vector=(1.0,), document_id="d2"),
        ]
    )
    assert store.remove_document("d1") == 2
    assert store.count() == 1
    assert store.get("v1") is None
    store.clear()
    assert store.count() == 0

    health = store.health_check()
    assert health.ready is True
    assert "Nothing to search" in health.detail


def test_in_memory_store_replace_document() -> None:
    store = InMemoryVectorStore()
    store.add([VectorRecord(id="old", vector=(1.0,), document_id="d1")])
    store.replace_document("d1", [VectorRecord(id="new", vector=(1.0,), document_id="d1")])
    assert store.get("old") is None
    assert store.get("new") is not None


# --------------------------------------------------------------------- #
# Null store + factory
# --------------------------------------------------------------------- #
def test_null_store_is_explicit() -> None:
    store = NullVectorStore("faiss")
    assert store.count() == 0
    assert store.search((1.0,), top_k=3) == []
    with pytest.raises(VectorStoreNotConfigured):
        store.add([VectorRecord(id="v1", vector=(1.0,))])
    health = store.health_check()
    assert health.ready is False
    assert "faiss" in health.detail
    store.clear()  # must not raise


def test_get_vector_store_resolves_configured_backends(settings) -> None:
    assert isinstance(get_vector_store(settings), NullVectorStore)

    settings.vector_store_backend = "memory"
    assert isinstance(get_vector_store(settings), InMemoryVectorStore)

    settings.vector_store_backend = "faiss"
    store = get_vector_store(settings)
    assert isinstance(store, NullVectorStore)
    assert "not implemented" in store.health_check().detail

    settings.vector_store_backend = "nonsense"
    assert "Unknown vector store" in get_vector_store(settings).health_check().detail


# --------------------------------------------------------------------- #
# Indexer + vector retriever
# --------------------------------------------------------------------- #
def _chunks() -> list[Chunk]:
    return [
        Chunk(id="d1:0", document_id="d1", chunk_index=0, content="tuproq sho'rlanishi"),
        Chunk(id="d2:0", document_id="d2", chunk_index=0, content="pomidor kasalligi"),
    ]


def test_indexer_needs_a_real_store_and_real_embeddings(settings) -> None:
    null_store_indexer = VectorIndexer(NullVectorStore(), StubEmbedder())
    assert null_store_indexer.is_ready is False
    with pytest.raises(VectorStoreNotConfigured):
        null_store_indexer.index_chunks(_chunks())

    store = InMemoryVectorStore()
    without_embeddings = VectorIndexer(store, NullEmbeddingProvider())
    assert without_embeddings.is_ready is False
    with pytest.raises(EmbeddingsNotConfigured):
        without_embeddings.index_chunks(_chunks())


def test_indexer_indexes_batches_and_reports_errors(database, settings) -> None:
    store = InMemoryVectorStore()
    indexer = VectorIndexer(store, StubEmbedder(), database)
    assert indexer.is_ready is True

    assert indexer.index_chunks(_chunks(), batch_size=1) == 2
    assert store.count() == 2

    class BrokenEmbedder(StubEmbedder):
        def embed(self, texts):
            return [self._vector("x")]  # wrong length on purpose

    with pytest.raises(RuntimeError, match="vectors"):
        VectorIndexer(InMemoryVectorStore(), BrokenEmbedder(), database).index_chunks(_chunks())


def test_indexer_indexes_the_database(database, settings) -> None:
    database.upsert_document(KnowledgeDocument(id="d1", title="T", content="c"))
    database.add_chunks([Chunk(id="d1:0", document_id="d1", chunk_index=0, content="tuproq")])
    store = InMemoryVectorStore()
    indexed = VectorIndexer(store, StubEmbedder(), database).index_database()
    assert indexed == 1

    with pytest.raises(RuntimeError):
        VectorIndexer(store, StubEmbedder(), None).index_database()


def test_vector_retriever_searches_real_index(database, settings) -> None:
    database.upsert_document(KnowledgeDocument(id="d1", title="Tuproq", content="c"))
    database.upsert_document(KnowledgeDocument(id="d2", title="Pomidor", content="c"))
    store = InMemoryVectorStore()
    VectorIndexer(store, StubEmbedder(), database).index_chunks(_chunks())

    retriever = VectorRetriever(
        backend="memory", store=store, embeddings=StubEmbedder(), database=database
    )
    assert retriever.is_ready is True
    health = retriever.health_check()
    assert health.ready is True
    assert health.search_mode == "vector"

    results = retriever.search("pomidor haqida", top_k=1)
    assert len(results) == 1
    assert results[0].chunk.document_id == "d2"
    assert results[0].document is not None
    assert results[0].document.title == "Pomidor"


def test_vector_retriever_is_unready_without_embeddings(database, settings) -> None:
    store = InMemoryVectorStore()
    store.add([VectorRecord(id="v1", vector=(1.0,), document_id="d1")])
    retriever = VectorRetriever(
        backend="memory", store=store, embeddings=NullEmbeddingProvider(), database=database
    )
    assert retriever.is_ready is False
    assert retriever.search("tuproq") == []
    assert "embedding provider" in retriever.health_check().detail


def test_vector_indexer_metadata_keeps_provenance() -> None:
    store = InMemoryVectorStore()
    VectorIndexer(store, StubEmbedder()).index_chunks(
        [Chunk(id="d1:0", document_id="d1", chunk_index=4, content="tuproq", metadata={"category": "soil"})]
    )
    record = store.get("vec:d1:0")
    assert record is not None
    assert record.metadata["chunk_index"] == 4
    assert record.metadata["category"] == "soil"
    assert record.content == "tuproq"
