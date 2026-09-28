"""Tests for the ingestion pipeline: cleaning, language, metadata, dedupe."""

from __future__ import annotations

from pathlib import Path

import pytest

from agroai.config.settings import Settings
from agroai.ingestion.pipeline import (
    STAGES,
    IngestionPipeline,
    clean_text,
    detect_language,
)
from agroai.knowledge.metadata import TrustLevel


def _write(directory: Path, name: str, text: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_text(text, encoding="utf-8")
    return path


# --------------------------------------------------------------------- #
# Stage: clean
# --------------------------------------------------------------------- #
def test_clean_text_normalizes_noise() -> None:
    raw = "\ufeffBirinchi\r\n\r\nIkkinchi   \n\n\n\nUchinchi\x07"
    assert clean_text(raw) == "Birinchi\n\nIkkinchi\n\nUchinchi"


def test_clean_text_handles_empty() -> None:
    assert clean_text("") == ""
    assert clean_text("   \n\t  ") == ""


# --------------------------------------------------------------------- #
# Stage: detect language
# --------------------------------------------------------------------- #
def test_detect_language_uzbek_latin_from_apostrophes() -> None:
    guess = detect_language("Pomidor barglarida sariq dog'lar paydo bo'ldi")
    assert guess.language == "uz"
    assert guess.is_confident is True


def test_detect_language_english_from_lexicon() -> None:
    text = (
        "The soil and the water for the crop and the plant disease of the field "
        "with the irrigation and the harvest of the farm is important."
    )
    assert detect_language(text).language == "en"


def test_detect_language_russian_cyrillic() -> None:
    guess = detect_language("Заболевания растений важны для урожая пшеницы")
    assert guess.language == "ru"
    assert guess.method == "cyrillic-ru"


def test_detect_language_uzbek_cyrillic() -> None:
    guess = detect_language("Буғдой ўсимлиги суғориш ва ҳосил ҳақида маълумот")
    assert guess.language == "uz"
    assert guess.method == "cyrillic-uz"


def test_detect_language_falls_back_to_default_when_unsure() -> None:
    assert detect_language("", default="uz").language == "uz"
    assert detect_language("12345 ... 6789", default="ru").language == "ru"
    assert detect_language("Salom", default="en").language == "en"


# --------------------------------------------------------------------- #
# Stages: metadata + chunking
# --------------------------------------------------------------------- #
def test_prepare_builds_metadata_and_chunks(tmp_path: Path, settings: Settings) -> None:
    path = _write(
        settings.knowledge_dir / "soil",
        "shorlanish.txt",
        "---\n"
        "organization: FAO\n"
        "source_type: government\n"
        "publication_date: 2024/3/5\n"
        "region: Sirdaryo\n"
        "country: UZ\n"
        "license: CC-BY\n"
        "---\n"
        "Tuproq sho'rlanishi bo'yicha qisqa eslatma. " * 20,
    )
    prepared = IngestionPipeline(settings=settings).prepare(path)
    document = prepared.document

    assert document.source == "FAO"
    assert document.category == "soil"
    assert document.publication_date == "2024-03-05"
    assert document.metadata["organization"] == "FAO"
    assert document.trust_level is TrustLevel.TIER_1
    assert document.metadata["trust_level"] == "tier_1"
    assert document.metadata["region"] == "Sirdaryo"
    assert prepared.chunks
    # prepare() runs every stage except the two that touch the database.
    assert prepared.stages_run == [
        stage for stage in STAGES if stage not in {"deduplicate", "store"}
    ]
    assert prepared.metadata_problems == []


def test_prepare_reports_unknown_source_as_tier_3(tmp_path: Path, settings: Settings) -> None:
    path = _write(settings.knowledge_dir / "soil", "note.txt", "Tuproq haqida oddiy eslatma.")
    document = IngestionPipeline(settings=settings).prepare(path).document
    assert document.trust_level is TrustLevel.TIER_3
    assert document.source == "note.txt"


def test_prepare_rejects_empty_and_unsupported(tmp_path: Path, settings: Settings) -> None:
    from agroai.ingestion.base import DocumentReadError, UnsupportedFormatError

    empty = _write(tmp_path, "empty.txt", "   \n\n  ")
    with pytest.raises(DocumentReadError):
        IngestionPipeline(settings=settings).prepare(empty)

    pdf = _write(tmp_path, "report.pdf", "%PDF-1.4")
    with pytest.raises(NotImplementedError):
        IngestionPipeline(settings=settings).prepare(pdf)

    xlsx = _write(tmp_path, "table.xlsx", "binary")
    with pytest.raises(UnsupportedFormatError):
        IngestionPipeline(settings=settings).prepare(xlsx)


# --------------------------------------------------------------------- #
# Stages: deduplicate + store
# --------------------------------------------------------------------- #
def test_pipeline_ingest_deduplicates_by_hash(database, settings: Settings) -> None:
    pipeline = IngestionPipeline(database, settings)
    path = _write(settings.knowledge_dir / "soil", "note.txt", "Tuproq tahlili matni.")

    first = pipeline.ingest_file(path)
    assert first.documents_ingested == 1
    assert first.chunks_created >= 1

    second = pipeline.ingest_file(path)
    assert second.documents_skipped == 1
    assert database.count_documents() == 1

    forced = pipeline.ingest_file(path, overwrite=True)
    assert forced.documents_ingested == 1
    assert database.count_documents() == 1


def test_pipeline_ingest_directory_is_incremental(database, settings: Settings) -> None:
    pipeline = IngestionPipeline(database, settings)
    _write(settings.knowledge_dir / "crops", "a.txt", "Bug'doy yetishtirish.")
    _write(settings.knowledge_dir / "pests", "b.md", "# Shira\n\nShira haqida.")

    first = pipeline.ingest_directory(settings.knowledge_dir)
    assert first.documents_ingested == 2
    assert first.documents_failed == 0

    second = pipeline.ingest_directory(settings.knowledge_dir)
    assert second.documents_ingested == 0
    assert second.documents_skipped == 2


def test_pipeline_ingest_directory_reports_missing_directory(database, settings: Settings) -> None:
    report = IngestionPipeline(database, settings).ingest_directory(settings.knowledge_dir / "nope")
    assert report.documents_failed == 1
    assert report.errors


def test_pipeline_store_requires_a_database(settings: Settings, tmp_path: Path) -> None:
    pipeline = IngestionPipeline(None, settings)
    path = _write(tmp_path, "note.txt", "Matn.")
    with pytest.raises(RuntimeError):
        pipeline.ingest_file(path)


def test_pipeline_source_record_carries_trust_level(database, settings: Settings) -> None:
    pipeline = IngestionPipeline(database, settings)
    path = _write(settings.knowledge_dir / "veterinary", "vet.txt", "Emlash jadvali.")
    pipeline.ingest_file(path)
    sources = database.list_sources()
    assert sources[0].metadata.get("trust_level") == TrustLevel.TIER_3.value
    assert sources[0].metadata.get("category") == "veterinary"
