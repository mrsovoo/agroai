"""Tests for the retriever abstraction and embedding interface honesty."""

from __future__ import annotations

import pytest

from agroai.knowledge.models import Chunk, KnowledgeDocument
from agroai.retrieval.embeddings import (
    EmbeddingProvider,
    EmbeddingsNotConfigured,
    NullEmbeddingProvider,
    get_embedding_provider,
)
from agroai.retrieval.retriever import (
    Retriever,
    SQLiteRetriever,
    VectorRetriever,
    get_retriever,
)


def _seed(database) -> None:
    database.upsert_document(
        KnowledgeDocument(id="d1", title="Pomidor", content="c", category="plant_diseases")
    )
    database.add_chunks(
        [
            Chunk(id="d1:0", document_id="d1", chunk_index=0, content="Pomidor kech quri yopish kasalligi"),
            Chunk(id="d1:1", document_id="d1", chunk_index=1, content="Sug'orish jadvali va me'yorlari"),
        ]
    )


def test_sqlite_retriever_search(database) -> None:
    _seed(database)
    retriever = SQLiteRetriever(database)
    results = retriever.search("pomidor kasallik", top_k=2)
    assert 0 < len(results) <= 2
    assert results[0].chunk.document_id == "d1"
    assert results[0].score > 0
    assert results[0].document is not None
    assert "Pomidor" in results[0].source_line


def test_sqlite_retriever_top_k_and_empty(database) -> None:
    _seed(database)
    retriever = SQLiteRetriever(database)
    assert retriever.search("pomidor", top_k=1).__len__() == 1
    assert retriever.search("pomidor", top_k=0) == []
    assert retriever.search("xyzzy_notfound", top_k=3) == []


def test_retriever_is_independent_from_llm(database) -> None:
    """The retriever must work with no LLM involved at all."""
    _seed(database)
    retriever = get_retriever(database)
    assert isinstance(retriever, Retriever)
    assert retriever.search("sug'orish", top_k=2)


def test_health_check_reports_state(database) -> None:
    retriever = SQLiteRetriever(database)
    health = retriever.health_check()
    assert health.ready is True
    assert health.documents == 0
    assert "empty" in health.detail
    _seed(database)
    health = retriever.health_check()
    assert health.documents == 1
    assert health.chunks == 2


def test_vector_retriever_is_honest_placeholder() -> None:
    retriever = VectorRetriever(backend="faiss")
    assert retriever.search("anything", top_k=3) == []
    health = retriever.health_check()
    assert health.ready is False
    assert "not configured" in health.detail


def test_get_retriever_unknown_backend_falls_back_to_sqlite(database) -> None:
    from agroai.config.settings import Settings

    settings = Settings.from_env()
    settings.retrieval_backend = "faiss"
    retriever = get_retriever(database, settings)
    assert isinstance(retriever, SQLiteRetriever)


def test_null_embeddings_raise_on_use(settings) -> None:
    provider = get_embedding_provider(settings)
    assert isinstance(provider, EmbeddingProvider)
    assert provider.is_configured is False
    with pytest.raises(EmbeddingsNotConfigured):
        provider.embed(["matn"])


def test_null_embeddings_health_is_explicit(settings) -> None:
    provider = NullEmbeddingProvider(dimensions=768)
    health = provider.health_check()
    assert health.configured is False
    assert health.dimensions == 768
    assert "lexical" in health.detail.lower()
