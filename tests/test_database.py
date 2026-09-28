"""Tests for the SQLite storage layer: schema, CRUD, FTS search, fallback."""

from __future__ import annotations

import sqlite3

import pytest

from agroai.database.database import Database
from agroai.knowledge.models import Chunk, KnowledgeDocument, SourceRecord


def _doc(doc_id: str, title: str, content: str, **kwargs) -> KnowledgeDocument:
    return KnowledgeDocument(id=doc_id, title=title, content=content, **kwargs)


def test_ping_and_counts(database) -> None:
    assert database.ping() is True
    assert database.count_documents() == 0
    assert database.count_chunks() == 0


def test_upsert_and_get_document(database) -> None:
    document = _doc("d1", "Pomidor", "Pomidor parvarishi", category="crops")
    database.upsert_document(document)
    loaded = database.get_document("d1")
    assert loaded is not None
    assert loaded.title == "Pomidor"
    assert loaded.category == "crops"
    assert loaded.metadata == {}
    # Upsert updates in place.
    database.upsert_document(_doc("d1", "Pomidor 2", "Yangi matn"))
    assert database.count_documents() == 1
    assert database.get_document("d1").title == "Pomidor 2"


def test_find_by_hash(database) -> None:
    document = _doc("d1", "T", "C")
    database.upsert_document(document)
    assert database.find_by_hash(document.content_hash) == "d1"
    assert database.find_by_hash("0" * 64) is None


def test_delete_document_cascades_chunks(database) -> None:
    document = _doc("d1", "T", "bir iki uch to'rt besh")
    database.upsert_document(document)
    database.add_chunks(
        [Chunk(id="d1:0", document_id="d1", chunk_index=0, content="bir iki uch")]
    )
    assert database.count_chunks() == 1
    database.delete_document("d1")
    assert database.count_documents() == 0
    assert database.count_chunks() == 0


def test_add_chunks_and_search_fts(database) -> None:
    database.upsert_document(_doc("d1", "Pomidor kasalliklari", "x"))
    database.add_chunks(
        [Chunk(id="d1:0", document_id="d1", chunk_index=0, content="Pomidor barglarida sariq dog'lar")]
    )
    results = database.search("pomidor sariq", top_k=5)
    assert results
    chunk, document, score = results[0]
    assert chunk.id == "d1:0"
    assert document.id == "d1"
    assert score > 0


def test_search_ranks_best_match_first(database) -> None:
    database.upsert_document(_doc("d1", "A", "x"))
    database.upsert_document(_doc("d2", "B", "y"))
    database.add_chunks(
        [
            Chunk(id="d1:0", document_id="d1", chunk_index=0, content="Bug'doy urug'ini tayyorlash"),
            Chunk(id="d2:0", document_id="d2", chunk_index=0, content="Pomidor kasalligi sariq dog'lar bilan kechadi"),
        ]
    )
    results = database.search("pomidor", top_k=2)
    assert results[0][1].id == "d2"


def test_search_empty_query_returns_empty(database) -> None:
    assert database.search("", top_k=5) == []
    assert database.search("   ", top_k=5) == []


def test_search_without_fts_falls_back_to_like(tmp_path) -> None:
    db = Database(tmp_path / "nolike.db")
    db.connect()
    db.initialize()
    try:
        # Simulate a build without FTS5.
        db.fts_enabled = False
        db.upsert_document(_doc("d1", "T", "c"))
        db.add_chunks([Chunk(id="d1:0", document_id="d1", chunk_index=0, content="Zamburug'li kasallik davolash")])
        results = db.search("kasallik", top_k=3)
        assert results and results[0][1].id == "d1"
    finally:
        db.close()


def test_fts_special_characters_do_not_crash(database) -> None:
    database.upsert_document(_doc("d1", "T", "oddiy matn"))
    database.add_chunks([Chunk(id="d1:0", document_id="d1", chunk_index=0, content="oddiy matn")])
    # Should not raise, even with FTS operator characters.
    database.search('pomidor AND (yoki "drop")*', top_k=3)


def test_sources_and_meta_roundtrip(database) -> None:
    database.upsert_source(SourceRecord(name="FAO", source_type="international_organization", document_count=2))
    rows = database.list_sources()
    assert rows[0].name == "FAO"
    assert rows[0].document_count == 2


def test_iter_documents_lazy_and_category_filter(database) -> None:
    database.upsert_document(_doc("d1", "A", "x", category="crops"))
    database.upsert_document(_doc("d2", "B", "y", category="soil"))
    assert [doc.id for doc in database.iter_documents(category="soil")] == ["d2"]
    assert len(list(database.iter_documents())) == 2


def test_context_manager_opens_and_closes(tmp_path) -> None:
    with Database(tmp_path / "ctx.db") as db:
        db.upsert_document(_doc("d1", "T", "C"))
        assert db.count_documents() == 1


def test_connect_creates_parent_directory(tmp_path) -> None:
    db = Database(tmp_path / "deep" / "nested" / "agroai.db")
    db.connect()
    try:
        assert (tmp_path / "deep" / "nested" / "agroai.db").exists()
    finally:
        db.close()
