"""Trusted source registry.

This module ships a *metadata registry* of the kinds of organizations whose
material may later be imported (FAO, USDA, ministries, universities, ...).
No downloads, no scraping, no network calls: this is provenance
infrastructure only. Automatic updates live in
:mod:`agroai.sources.updater` and are likewise opt-in and offline until a
network phase is explicitly enabled.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass(frozen=True)
class TrustedSource:
    """A known, trusted origin for agricultural/veterinary knowledge."""

    name: str
    source_type: str
    url: str = ""
    notes: str = ""


# Categories of sources the architecture trusts *in principle*.
TRUSTED_SOURCE_TYPES: tuple[str, ...] = (
    "international_organization",
    "government",
    "university",
    "research",
    "veterinary_authority",
    "local_organization",
)


# Curated registry (metadata only - AgroAI fetches nothing from these now).
TRUSTED_SOURCES: tuple[TrustedSource, ...] = (
    TrustedSource(
        "FAO (Food and Agriculture Organization of the United Nations)",
        "international_organization",
        "https://www.fao.org",
        "Global agriculture, food security and livestock guidance.",
    ),
    TrustedSource(
        "USDA (United States Department of Agriculture)",
        "government",
        "https://www.usda.gov",
        "Crop production, plant health and soil references.",
    ),
    TrustedSource(
        "OIE / WOAH (World Organisation for Animal Health)",
        "veterinary_authority",
        "https://www.woah.org",
        "Animal disease standards and veterinary guidance.",
    ),
    TrustedSource(
        "Uzbekistan Ministry of Agriculture",
        "government",
        "https://agro.uz",
        "National agricultural policy and extension materials.",
    ),
    TrustedSource(
        "Uzbekistan Veterinary Committee",
        "veterinary_authority",
        "",
        "National veterinary regulation and animal health programs.",
    ),
    TrustedSource(
        "Tashkent State Agrarian University",
        "university",
        "https://tdau.uz",
        "Agronomy and veterinary research and teaching materials.",
    ),
    TrustedSource(
        "Samarkand State University of Veterinary Medicine",
        "university",
        "",
        "Veterinary medicine education and research.",
    ),
    TrustedSource(
        "CIMMYT / CGIAR research centers",
        "research",
        "https://www.cimmyt.org",
        "Peer-reviewed crop and farming-systems research.",
    ),
)


@dataclass
class SourceCatalogEntry:
    """A source aggregated from ingested documents (runtime view)."""

    name: str
    source_type: str = "local"
    url: str = ""
    document_count: int = 0
    last_ingested_at: str = ""
    trusted: bool = False
    categories: set[str] = field(default_factory=set)


def _acronyms(name: str) -> set[str]:
    """Extract uppercase acronym tokens (FAO, USDA, ...) from a name."""
    return {
        token for token in re.findall(r"[A-Za-z]+", name) if token.isupper() and len(token) >= 2
    }


def _matches(source: TrustedSource, lowered_name: str) -> bool:
    """Conservative match: full-name substring or a known acronym as a word.

    ``"FAO report 2024"`` matches, ``"afo"`` does not.
    """
    source_lower = source.name.lower()
    if source_lower in lowered_name or lowered_name in source_lower:
        return True
    return any(
        re.search(rf"\b{re.escape(acronym.lower())}\b", lowered_name)
        for acronym in _acronyms(source.name)
    )


def get_source(source_name: str) -> TrustedSource | None:
    """Return the matching registry entry for ``source_name``, if any.

    Offline and dependency-free: used to infer trust levels, never to fetch.
    """
    lowered = (source_name or "").strip().lower()
    if not lowered:
        return None
    for source in TRUSTED_SOURCES:
        if _matches(source, lowered):
            return source
    return None


def is_trusted(source_name: str) -> bool:
    """Return True when ``source_name`` matches a registry entry (offline)."""
    return get_source(source_name) is not None


def build_catalog(source_records) -> list[SourceCatalogEntry]:
    """Build a display catalog from database source records.

    ``source_records`` is any iterable of
    :class:`agroai.knowledge.models.SourceRecord` (as returned by
    :meth:`Database.list_sources`).
    """
    entries: list[SourceCatalogEntry] = []
    for record in source_records:
        entries.append(
            SourceCatalogEntry(
                name=record.name,
                source_type=record.source_type,
                url=record.url,
                document_count=record.document_count,
                last_ingested_at=record.last_ingested_at,
                trusted=is_trusted(record.name),
            )
        )
    entries.sort(key=lambda entry: (-entry.document_count, entry.name.lower()))
    return entries


__all__ = [
    "SourceCatalogEntry",
    "TRUSTED_SOURCES",
    "TRUSTED_SOURCE_TYPES",
    "TrustedSource",
    "build_catalog",
    "get_source",
    "is_trusted",
]
