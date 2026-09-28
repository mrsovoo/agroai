"""Tests for knowledge data models: hashing, chunking, provenance."""

from __future__ import annotations

from agroai.knowledge.models import (
    Chunk,
    IngestionReport,
    KnowledgeDocument,
    RetrievedChunk,
    chunk_text,
    content_hash,
    iter_chunks,
)


def test_content_hash_is_stable_and_sensitive() -> None:
    first = content_hash("title", "body")
    assert first == content_hash("title", "body")
    assert first != content_hash("title ", "body")
    assert first != content_hash("title", "body ")
    assert len(first) == 64


def test_document_computes_hash_when_missing() -> None:
    document = KnowledgeDocument(id="d1", title="T", content="C")
    assert document.content_hash == content_hash("T", "C")


def test_document_word_and_char_count() -> None:
    document = KnowledgeDocument(id="d1", title="T", content="one two three")
    assert document.word_count == 3
    assert document.char_count == len("one two three")


def test_chunk_text_short_document_single_chunk() -> None:
    chunks = chunk_text("salom dunyo", chunk_size=800, overlap=120)
    assert len(chunks) == 1
    start, end, text = chunks[0]
    assert (start, end, text) == (0, len("salom dunyo"), "salom dunyo")


def test_chunk_text_splits_long_text() -> None:
    paragraph = "Bu test paragrafi. " * 100  # ~1900 chars
    chunks = chunk_text(paragraph, chunk_size=800, overlap=100)
    assert len(chunks) > 1
    for start, end, text in chunks:
        assert end - start <= 800
        assert text == paragraph[start:end]


def test_chunk_text_empty_and_invalid() -> None:
    assert chunk_text("") == []
    assert chunk_text("   \n  ") == []
    try:
        chunk_text("abc", chunk_size=0)
    except ValueError:
        pass
    else:
        raise AssertionError("chunk_size=0 must raise ValueError")


def test_iter_chunks_preserves_provenance() -> None:
    document = KnowledgeDocument(
        id="doc_1", title="Pomidor", content="so'z " * 400, category="plant_diseases", language="uz"
    )
    chunks = list(iter_chunks(document, chunk_size=500, overlap=50))
    assert chunks
    for index, chunk in enumerate(chunks):
        assert chunk.document_id == document.id
        assert chunk.id == f"doc_1:{index}"
        assert chunk.chunk_index == index
        assert chunk.metadata["category"] == "plant_diseases"
        assert chunk.metadata["language"] == "uz"
        assert chunk.start_char <= chunk.end_char


def test_chunk_hash_autocomputed() -> None:
    chunk = Chunk(id="c1", document_id="d1", chunk_index=0, content="matn")
    assert chunk.content_hash == content_hash("matn")


def test_retrieved_chunk_source_line() -> None:
    document = KnowledgeDocument(
        id="d1", title="Pomidor kasalliklari", content="c", source="FAO", source_url="https://fao.org"
    )
    chunk = Chunk(id="d1:0", document_id="d1", chunk_index=0, content="c")
    retrieved = RetrievedChunk(chunk=chunk, score=1.5, document=document)
    assert "Pomidor kasalliklari" in retrieved.source_line
    assert "FAO" in retrieved.source_line
    assert "https://fao.org" in retrieved.source_line


def test_ingestion_report_merge() -> None:
    first = IngestionReport(documents_seen=1, documents_ingested=1, chunks_created=3)
    second = IngestionReport(documents_seen=1, documents_skipped=1, errors=["x"])
    first.merge(second)
    assert first.documents_seen == 2
    assert first.documents_ingested == 1
    assert first.documents_skipped == 1
    assert first.chunks_created == 3
    assert first.errors == ["x"]
