"""Tests for the knowledge manager: dedup, incremental ingestion, sources."""

from __future__ import annotations

from pathlib import Path

from agroai.knowledge.manager import KnowledgeManager


def _make_doc_file(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


def test_ingest_file_and_retrieve_roundtrip(database, settings) -> None:
    manager = KnowledgeManager(database, settings)
    path = _make_doc_file(settings.knowledge_dir / "soil", "soil_note.txt", "Tuproq azotiga tekshirish bahorda qilinadi.")
    report = manager.ingest_file(path)
    assert report.documents_seen == 1
    assert report.documents_ingested == 1
    assert report.documents_skipped == 0
    assert report.chunks_created >= 1
    assert database.count_documents() == 1
    assert database.count_chunks() >= 1


def test_duplicate_ingestion_is_skipped(database, settings) -> None:
    manager = KnowledgeManager(database, settings)
    path = _make_doc_file(settings.knowledge_dir / "soil", "note.txt", "Bir xil matn.")
    first = manager.ingest_file(path)
    second = manager.ingest_file(path)
    assert first.documents_ingested == 1
    assert second.documents_skipped == 1
    assert database.count_documents() == 1


def test_overwrite_reingests_duplicate(database, settings) -> None:
    manager = KnowledgeManager(database, settings)
    path = _make_doc_file(settings.knowledge_dir / "soil", "note.txt", "Bir xil matn.")
    manager.ingest_file(path)
    forced = manager.ingest_file(path, overwrite=True)
    assert forced.documents_ingested == 1
    assert database.count_documents() == 1


def test_ingest_directory_incremental(database, settings) -> None:
    manager = KnowledgeManager(database, settings)
    directory = settings.knowledge_dir / "livestock"
    _make_doc_file(directory, "a.txt", "Mol boqish bo'yicha maslahat.")
    _make_doc_file(directory, "b.txt", "Qo'ylarni parvarishlash qoidalari.")

    first = manager.ingest_directory(settings.knowledge_dir)
    assert first.documents_ingested == 2
    second = manager.ingest_directory(settings.knowledge_dir)
    assert second.documents_ingested == 0
    assert second.documents_skipped == 2
    assert database.count_documents() == 2


def test_ingest_directory_skips_unsupported(database, settings) -> None:
    manager = KnowledgeManager(database, settings)
    directory = settings.knowledge_dir / "crops"
    _make_doc_file(directory, "ok.txt", "Makkajo'xori haqida.")
    _make_doc_file(directory, "skip.pdf", "%PDF not parsed")
    report = manager.ingest_directory(directory)
    assert report.documents_ingested == 1
    assert report.documents_failed == 1
    assert any("not implemented" in error for error in report.errors)


def test_ingest_missing_file_reported_as_failure(database, settings) -> None:
    manager = KnowledgeManager(database, settings)
    report = manager.ingest_file(settings.knowledge_dir / "ghost.txt")
    assert report.documents_failed == 1
    assert report.errors


def test_source_record_created(database, settings) -> None:
    manager = KnowledgeManager(database, settings)
    path = _make_doc_file(settings.knowledge_dir / "veterinary", "vet.txt", "Mollar uchun emlash jadvali.")
    manager.ingest_file(path)
    sources = database.list_sources()
    assert len(sources) == 1
    assert sources[0].name == "vet.txt"
    assert sources[0].document_count == 1
