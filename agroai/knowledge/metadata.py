"""Provenance metadata: hashing, timestamps and the document metadata schema.

Everything that describes *where a piece of knowledge came from* lives here.
The roadmap requires three things from a knowledge document before it may enter
the base:

* a **source** (organization, URL, author, publication date),
* a **category** (crops, veterinary, soil, ...),
* a **trust level** derived from the source - never invented, only inferred
  from the curated registry in :mod:`agroai.sources.registry`.

Hashing lives here too, because deduplication *is* provenance bookkeeping: the
content hash is what lets the pipeline recognise the same FAO document served
from ten different places.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from agroai.sources.registry import get_source


def utcnow_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def content_hash(*parts: str) -> str:
    """Stable SHA-256 hash used for deduplication and change detection."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update((part or "").encode("utf-8"))
        digest.update(b"\x1f")  # unit separator to avoid ambiguity
    return digest.hexdigest()


class TrustLevel(str, Enum):
    """How much weight a source's material may carry (roadmap "manbalar" tiers).

    * ``TIER_1`` - peer-reviewed research, FAO/WOAH/USDA, government bodies,
      universities: authoritative, quotable.
    * ``TIER_2`` - professional and extension organizations, research centers:
      reliable but usually secondary.
    * ``TIER_3`` - everything else, including blogs, forums and social posts.
      Such material is **never** treated as fact automatically.
    """

    TIER_1 = "tier_1"
    TIER_2 = "tier_2"
    TIER_3 = "tier_3"
    UNKNOWN = "unknown"

    @property
    def rank(self) -> int:
        """Lower is more authoritative; ``UNKNOWN`` sorts last."""
        return {TrustLevel.TIER_1: 1, TrustLevel.TIER_2: 2, TrustLevel.TIER_3: 3}[
            self
        ] if self is not TrustLevel.UNKNOWN else 4

    @property
    def is_authoritative(self) -> bool:
        return self is TrustLevel.TIER_1


#: Source *types* (from the trusted registry) mapped to a trust level.
SOURCE_TYPE_TRUST: dict[str, TrustLevel] = {
    "international_organization": TrustLevel.TIER_1,
    "government": TrustLevel.TIER_1,
    "university": TrustLevel.TIER_1,
    "research": TrustLevel.TIER_1,
    "veterinary_authority": TrustLevel.TIER_1,
    "local_organization": TrustLevel.TIER_2,
    "professional_organization": TrustLevel.TIER_2,
    "extension_service": TrustLevel.TIER_2,
}


def infer_trust_level(source_name: str, source_type: str = "") -> TrustLevel:
    """Infer the trust level of a source from the curated registry (offline).

    Unknown sources get ``TIER_3``: local notes and imported files are useful,
    but they are not treated as authority until a human says so.
    """
    if source_type:
        level = SOURCE_TYPE_TRUST.get(source_type.strip().lower())
        if level is not None:
            return level
    trusted = get_source(source_name) if source_name else None
    if trusted is not None:
        return SOURCE_TYPE_TRUST.get(trusted.source_type, TrustLevel.TIER_2)
    return TrustLevel.TIER_3


_DATE_PATTERNS = (
    re.compile(r"^(?P<y>\d{4})-(?P<m>\d{2})-(?P<d>\d{2})$"),
    re.compile(r"^(?P<y>\d{4})/(?P<m>\d{1,2})/(?P<d>\d{1,2})$"),
    re.compile(r"^(?P<d>\d{1,2})\.(?P<m>\d{1,2})\.(?P<y>\d{4})$"),
    re.compile(r"^(?P<y>\d{4})$"),
)


def normalize_date(value: str) -> str:
    """Normalize a publication date to ``YYYY-MM-DD`` (``""`` if unusable).

    Only formats that can be read unambiguously are converted; anything else is
    returned unchanged-but-flagged by :meth:`DocumentMetadata.problems`, because
    silently guessing a date would corrupt provenance.
    """
    value = (value or "").strip()
    if not value:
        return ""
    for pattern in _DATE_PATTERNS:
        match = pattern.match(value)
        if not match:
            continue
        parts = match.groupdict()
        year = int(parts["y"])
        month = int(parts.get("m") or 1)
        day = int(parts.get("d") or 1)
        try:
            return datetime(year, month, day).date().isoformat()
        except ValueError:
            return value
    return value


#: Metadata keys accepted from a document's front matter block.
FRONT_MATTER_FIELDS: tuple[str, ...] = (
    "title",
    "author",
    "organization",
    "source_type",
    "source_url",
    "publication_date",
    "language",
    "country",
    "region",
    "category",
    "crop",
    "animal",
    "license",
    "version",
)


def parse_front_matter(text: str) -> tuple[dict[str, str], str]:
    """Split an optional ``---`` front matter block off a text document.

    Returns ``(fields, body)``. The block must be the first thing in the file
    and is delimited by a line containing only ``---``. Only ``key: value``
    lines whose key is in :data:`FRONT_MATTER_FIELDS` are kept, so a stray
    colon in prose cannot silently become provenance.
    """
    text = text or ""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text

    fields: dict[str, str] = {}
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[index + 1 :])
            return fields, body
        if ":" not in line or line.lstrip().startswith("#"):
            continue
        key, value = line.split(":", 1)
        key = key.strip().lower()
        if key not in FRONT_MATTER_FIELDS:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        fields[key] = value
    # No closing delimiter: this was not a front matter block after all.
    return {}, text


@dataclass
class DocumentMetadata:
    """The provenance record every knowledge document carries.

    Field names follow the roadmap's document metadata list so that export,
    citation and deduplication all speak the same vocabulary.
    """

    title: str = ""
    author: str = ""
    organization: str = ""
    source_url: str = ""
    publication_date: str = ""
    language: str = "uz"
    country: str = ""
    region: str = ""
    category: str = "general"
    license: str = ""
    content_hash: str = ""
    version: str = "1"
    trust_level: TrustLevel = TrustLevel.TIER_3
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    @property
    def is_uzbekistan_specific(self) -> bool:
        return (self.country or "").strip().lower() in {"uz", "uzbekistan", "o'zbekiston"}

    def problems(self) -> list[str]:
        """Return the list of reasons this metadata is not yet complete.

        An empty list means the record may be stored without caveats.
        """
        found: list[str] = []
        if not self.title.strip():
            found.append("missing title")
        if not self.language.strip():
            found.append("missing language")
        if not self.category.strip():
            found.append("missing category")
        if self.content_hash and len(self.content_hash) != 64:
            found.append("content_hash is not a SHA-256 hex digest")
        if self.publication_date and not re.match(r"^\d{4}(-\d{2}-\d{2})?$", self.publication_date):
            found.append(f"publication_date not normalized: {self.publication_date!r}")
        if self.source_url and not self.source_url.startswith(("http://", "https://")):
            found.append("source_url is not an http(s) URL")
        return found

    def is_complete(self) -> bool:
        return not self.problems()

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["trust_level"] = self.trust_level.value
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DocumentMetadata":
        known = {field_name for field_name in cls.__dataclass_fields__}
        clean = {key: value for key, value in (data or {}).items() if key in known}
        raw_level = clean.get("trust_level")
        if raw_level is not None and not isinstance(raw_level, TrustLevel):
            try:
                clean["trust_level"] = TrustLevel(str(raw_level))
            except ValueError:
                clean["trust_level"] = TrustLevel.UNKNOWN
        return cls(**clean)

    @classmethod
    def for_source(
        cls,
        *,
        title: str,
        source_name: str,
        source_type: str = "",
        category: str = "general",
        language: str = "uz",
        **overrides: Any,
    ) -> "DocumentMetadata":
        """Build metadata for a document and infer its trust level."""
        return cls(
            title=title,
            organization=overrides.pop("organization", source_name),
            category=category,
            language=language,
            trust_level=infer_trust_level(source_name, source_type),
            **overrides,
        )


__all__ = [
    "FRONT_MATTER_FIELDS",
    "DocumentMetadata",
    "SOURCE_TYPE_TRUST",
    "TrustLevel",
    "content_hash",
    "infer_trust_level",
    "normalize_date",
    "parse_front_matter",
    "utcnow_iso",
]
