"""Tests for the trusted source registry (offline metadata only)."""

from __future__ import annotations

from agroai.knowledge.models import SourceRecord
from agroai.sources import TRUSTED_SOURCES, build_catalog, is_trusted


def test_registry_is_populated_and_typed() -> None:
    assert TRUSTED_SOURCES
    assert all(source.name for source in TRUSTED_SOURCES)
    assert all(source.source_type for source in TRUSTED_SOURCES)


def test_is_trusted_matches_registry() -> None:
    assert is_trusted("FAO (Food and Agriculture Organization of the United Nations)")
    assert is_trusted("fao report 2024")
    assert is_trusted("USDA")
    assert not is_trusted("random_blog.ru")
    assert not is_trusted("")


def test_build_catalog_marks_trusted() -> None:
    records = [
        SourceRecord(name="FAO", source_type="international_organization", document_count=5),
        SourceRecord(name="some_blog", source_type="file", document_count=2),
    ]
    catalog = build_catalog(records)
    assert catalog[0].name == "FAO"  # sorted by document count first
    assert catalog[0].trusted is True
    assert catalog[1].trusted is False


def test_build_catalog_empty() -> None:
    assert build_catalog([]) == []
