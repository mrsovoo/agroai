"""Trusted sources: curated registry, provenance helpers and update interface.

Nothing in this package touches the network. :mod:`agroai.sources.registry`
holds the curated metadata of trusted organizations; :mod:`agroai.sources.updater`
defines how a *future*, opt-in, source-controlled update phase will behave.
"""

from __future__ import annotations

from agroai.sources.registry import (
    SourceCatalogEntry,
    TRUSTED_SOURCES,
    TRUSTED_SOURCE_TYPES,
    TrustedSource,
    build_catalog,
    get_source,
    is_trusted,
)
from agroai.sources.updater import (
    DisabledUpdater,
    KnowledgeUpdater,
    UpdateResult,
    get_updater,
)

__all__ = [
    "DisabledUpdater",
    "KnowledgeUpdater",
    "SourceCatalogEntry",
    "TRUSTED_SOURCES",
    "TRUSTED_SOURCE_TYPES",
    "TrustedSource",
    "UpdateResult",
    "build_catalog",
    "get_source",
    "get_updater",
    "is_trusted",
]
