"""Source bookkeeping for the knowledge base.

Two things live here:

* :class:`SourceRecord` - the runtime record of an ingested origin (what
  ``agroai sources`` lists and the database stores).
* :class:`SourceCitation` - a rendered, provenance-carrying reference to a
  document. The agent only ever cites sources through this type, and it can
  only be built *from a stored document*, so a citation cannot be invented.

The curated registry of *trusted* organizations (FAO, WOAH, USDA, ...) lives in
:mod:`agroai.sources.registry`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agroai.knowledge.documents import KnowledgeDocument
from agroai.knowledge.metadata import TrustLevel


@dataclass
class SourceRecord:
    """A tracked origin of imported knowledge (for ``agroai sources``)."""

    name: str
    source_type: str = "local"
    url: str = ""
    document_count: int = 0
    last_ingested_at: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class SourceCitation:
    """A reference to one stored document, with its provenance."""

    title: str
    source: str = ""
    source_url: str = ""
    organization: str = ""
    publication_date: str = ""
    category: str = ""
    region: str = ""
    country: str = ""
    trust_level: TrustLevel = TrustLevel.TIER_3

    @property
    def label(self) -> str:
        """One-line, human readable provenance label."""
        parts = [self.title or "(untitled)"]
        if self.organization and self.organization != self.source:
            parts.append(self.organization)
        elif self.source:
            parts.append(self.source)
        if self.publication_date:
            parts.append(self.publication_date)
        if self.region or self.country:
            parts.append(", ".join(part for part in (self.region, self.country) if part))
        if self.source_url:
            parts.append(self.source_url)
        return " | ".join(parts)

    def to_dict(self) -> dict[str, str]:
        return {
            "title": self.title,
            "source": self.source,
            "source_url": self.source_url,
            "organization": self.organization,
            "publication_date": self.publication_date,
            "category": self.category,
            "region": self.region,
            "country": self.country,
            "trust_level": self.trust_level.value,
        }


def citation_for(document: KnowledgeDocument) -> SourceCitation:
    """Build a citation from a stored document (never from free text)."""
    return SourceCitation(
        title=document.title or document.id,
        source=document.source,
        source_url=document.source_url,
        organization=document.organization(),
        publication_date=document.publication_date,
        category=document.category,
        region=document.region(),
        country=document.country(),
        trust_level=document.trust_level,
    )


__all__ = ["SourceCitation", "SourceRecord", "citation_for"]
