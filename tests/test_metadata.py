"""Tests for provenance metadata: trust levels, dates, front matter, citations."""

from __future__ import annotations

import pytest

from agroai.knowledge.documents import KnowledgeDocument
from agroai.knowledge.metadata import (
    DocumentMetadata,
    TrustLevel,
    infer_trust_level,
    normalize_date,
    parse_front_matter,
)
from agroai.knowledge.sources import SourceCitation, citation_for


# --------------------------------------------------------------------- #
# Trust levels
# --------------------------------------------------------------------- #
def test_trust_level_rank_and_authority() -> None:
    assert TrustLevel.TIER_1.rank < TrustLevel.TIER_2.rank < TrustLevel.TIER_3.rank
    assert TrustLevel.TIER_1.rank < TrustLevel.UNKNOWN.rank
    assert TrustLevel.TIER_1.is_authoritative is True
    assert TrustLevel.TIER_3.is_authoritative is False


def test_infer_trust_level_from_registry_and_type() -> None:
    assert infer_trust_level("FAO") is TrustLevel.TIER_1
    assert infer_trust_level("fao report 2024") is TrustLevel.TIER_1
    assert infer_trust_level("USDA") is TrustLevel.TIER_1
    # Declared source_type wins and is looked up offline.
    assert infer_trust_level("Some extension service", "extension_service") is TrustLevel.TIER_2
    # Unknown local notes are useful but not authority.
    assert infer_trust_level("random_blog.ru") is TrustLevel.TIER_3
    assert infer_trust_level("") is TrustLevel.TIER_3


# --------------------------------------------------------------------- #
# Dates
# --------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("", ""),
        ("2024-03-05", "2024-03-05"),
        ("2024/3/5", "2024-03-05"),
        ("05.03.2024", "2024-03-05"),
        ("2024", "2024-01-01"),
        ("2024-13-40", "2024-13-40"),  # impossible date is left untouched
        ("unknown", "unknown"),
    ],
)
def test_normalize_date(raw: str, expected: str) -> None:
    assert normalize_date(raw) == expected


# --------------------------------------------------------------------- #
# Document metadata schema
# --------------------------------------------------------------------- #
def test_metadata_problems_report_missing_provenance() -> None:
    metadata = DocumentMetadata(title="Tuproq sho'rlanishi", category="soil")
    assert metadata.problems() == []
    assert metadata.is_complete() is True

    incomplete = DocumentMetadata(title="", content_hash="abc")
    problems = incomplete.problems()
    assert any("title" in problem for problem in problems)
    assert any("content_hash" in problem for problem in problems)
    assert incomplete.is_complete() is False


def test_metadata_rejects_unnormalized_date_and_bad_url() -> None:
    metadata = DocumentMetadata(
        title="T",
        category="soil",
        publication_date="05.03.2024",
        source_url="fao.org",
    )
    problems = metadata.problems()
    assert any("publication_date" in problem for problem in problems)
    assert any("source_url" in problem for problem in problems)


def test_metadata_roundtrip_and_unknown_trust_level() -> None:
    metadata = DocumentMetadata(
        title="T", category="soil", country="Uzbekistan", region="Sirdaryo",
        trust_level=TrustLevel.TIER_1,
    )
    data = metadata.to_dict()
    assert data["trust_level"] == "tier_1"
    assert DocumentMetadata.from_dict(data) == metadata

    data["trust_level"] = "not-a-tier"
    assert DocumentMetadata.from_dict(data).trust_level is TrustLevel.UNKNOWN
    # Unknown keys are ignored rather than exploding.
    assert DocumentMetadata.from_dict({"title": "x", "bogus": 1}).title == "x"


def test_metadata_for_source_infers_organization_and_trust() -> None:
    metadata = DocumentMetadata.for_source(
        title="Guideline", source_name="FAO", category="soil"
    )
    assert metadata.organization == "FAO"
    assert metadata.trust_level is TrustLevel.TIER_1
    assert metadata.is_uzbekistan_specific is False


def test_document_projects_to_metadata_schema() -> None:
    document = KnowledgeDocument(
        id="d1",
        title="Sho'rlanish",
        content="matn",
        source="FAO",
        source_url="https://fao.org/x",
        category="soil",
        metadata={"region": "Sirdaryo", "country": "UZ"},
    )
    metadata = document.to_document_metadata()
    assert metadata.title == "Sho'rlanish"
    assert metadata.organization == "FAO"
    assert metadata.region == "Sirdaryo"
    assert metadata.trust_level is TrustLevel.TIER_1


# --------------------------------------------------------------------- #
# Front matter
# --------------------------------------------------------------------- #
def test_parse_front_matter_extracts_declared_provenance() -> None:
    text = (
        "---\n"
        "# a comment\n"
        "title: Tuproq sho'rlanishing ta'siri\n"
        'organization: "FAO"\n'
        "source_type: government\n"
        "publication_date: 2024/3/5\n"
        "region: Sirdaryo\n"
        "not_a_field: ignored\n"
        "---\n"
        "Asosiy matn.\n"
    )
    fields, body = parse_front_matter(text)
    assert fields["organization"] == "FAO"
    assert fields["source_type"] == "government"
    assert fields["publication_date"] == "2024/3/5"
    assert "not_a_field" not in fields
    assert body.strip() == "Asosiy matn."


def test_parse_front_matter_absent_or_unclosed_is_not_metadata() -> None:
    assert parse_front_matter("Oddiy matn") == ({}, "Oddiy matn")
    assert parse_front_matter("") == ({}, "")
    unclosed = "---\ntitle: X\nmatn davom etadi"
    fields, body = parse_front_matter(unclosed)
    assert fields == {}
    assert body == unclosed


# --------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------- #
def test_citation_is_built_only_from_a_document() -> None:
    document = KnowledgeDocument(
        id="d1",
        title="Pomidor kasalliklari",
        content="c",
        source="FAO",
        source_url="https://fao.org/pomidor",
        category="plant_diseases",
        publication_date="2024-03-05",
        metadata={"region": "Sirdaryo", "country": "UZ"},
    )
    citation = citation_for(document)
    assert isinstance(citation, SourceCitation)
    assert "Pomidor kasalliklari" in citation.label
    assert "Sirdaryo" in citation.label
    assert "https://fao.org/pomidor" in citation.label
    assert citation.to_dict()["trust_level"] == "tier_1"


def test_citation_label_falls_back_to_source_name() -> None:
    document = KnowledgeDocument(id="d1", title="", content="c", source="local note")
    citation = citation_for(document)
    assert citation.title == "d1"
    assert "local note" in citation.label
