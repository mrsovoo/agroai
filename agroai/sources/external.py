"""Trusted external search - policy now, provider later.

```text
local knowledge -> enough? --no--> external search -> trusted sources
                                 -> source validation -> knowledge candidate
                                 -> answer
```

The rule that matters is the last one: material fetched from the internet **never
becomes knowledge automatically**, and only registry-grade sources may be quoted
in an answer. Everything untrusted becomes a *candidate* awaiting review (see
:mod:`agroai.knowledge.candidates`).

What is implemented today:

* :func:`classify_url` / :func:`is_trusted_url` - deterministic domain policy,
  validated against the curated registry domains
  (:mod:`agroai.sources.registry`);
* :class:`TrustedSourcePolicy` - admits, filters and ranks results by trust;
* :class:`DisabledExternalSearch` - the shipped provider. It reports
  ``not_configured`` and performs **no network call**, so nothing can leak out of
  an offline deployment by accident.

A real provider implements :class:`ExternalSearchProvider`; the policy, ranking
and the ``search -> candidate`` pipeline stay unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable
from urllib.parse import urlparse

from agroai.config.settings import Settings, get_settings
from agroai.knowledge.metadata import TrustLevel
from agroai.sources.registry import TRUSTED_SOURCES

SEARCH_OK = "ok"
SEARCH_NOT_CONFIGURED = "not_configured"
SEARCH_NOT_IMPLEMENTED = "not_implemented"
SEARCH_ERROR = "error"

#: Explicitly trusted hosts (from the curated registry, plus national domains).
TRUSTED_HOSTS: frozenset[str] = frozenset(
    {
        "fao.org",
        "www.fao.org",
        "usda.gov",
        "www.usda.gov",
        "woah.org",
        "www.woah.org",
        "oie.int",
        "agro.uz",
        "www.agro.uz",
        "tdau.uz",
        "cimmyt.org",
        "www.cimmyt.org",
        "cgiar.org",
        "www.cgiar.org",
        "uzbektom.uz",
    }
    | {urlparse(source.url).netloc for source in TRUSTED_SOURCES if source.url}
)

#: Government/education suffixes that are trusted in principle (TIER_1/TIER_2).
TRUSTED_SUFFIXES: tuple[str, ...] = (
    ".gov",
    ".gov.uz",
    ".gov.uk",
    ".edu",
    ".ac.uk",
    ".edu.au",
)

#: Suffixes that are never authoritative on their own (TIER_3).
UNTRUSTED_SUFFIXES: tuple[str, ...] = (
    ".blogspot.com",
    ".wordpress.com",
    ".medium.com",
    ".facebook.com",
    ".t.me",
    ".pinterest.com",
)

MINIMUM_ANSWER_TRUST = TrustLevel.TIER_2


def extract_host(url: str) -> str:
    """Return the lowercase host of a URL (empty when unparseable)."""
    try:
        parsed = urlparse(url or "")
    except ValueError:
        return ""
    return (parsed.hostname or "").lower()


def is_https(url: str) -> bool:
    """True when the URL is HTTPS. Production traffic must be HTTPS."""
    try:
        return urlparse(url or "").scheme.lower() == "https"
    except ValueError:
        return False


def classify_url(url: str) -> TrustLevel:
    """Trust level implied by a URL's host, using the curated policy."""
    host = extract_host(url)
    if not host:
        return TrustLevel.UNKNOWN
    if host in TRUSTED_HOSTS:
        return TrustLevel.TIER_1
    if host.endswith(TRUSTED_SUFFIXES):
        return TrustLevel.TIER_1 if host.endswith((".edu", ".ac.uk")) else TrustLevel.TIER_2
    if host.endswith(UNTRUSTED_SUFFIXES):
        return TrustLevel.TIER_3
    # A generic organization domain: plausible, unverified.
    return TrustLevel.TIER_3 if "." in host else TrustLevel.UNKNOWN


def is_trusted_url(url: str, *, minimum: TrustLevel = MINIMUM_ANSWER_TRUST) -> bool:
    """True when ``url`` may be quoted in an answer (host trust >= ``minimum``)."""
    return classify_url(url).rank <= minimum.rank


@dataclass
class ExternalResult:
    """One external search hit, with its trust verdict attached."""

    title: str
    url: str
    snippet: str = ""
    source_type: str = ""
    trust_level: str = TrustLevel.UNKNOWN.value
    published_at: str = ""
    #: Trusted results may be cited; untrusted ones may only seed a candidate.
    quotable: bool = False
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "url": self.url,
            "snippet": self.snippet,
            "source_type": self.source_type,
            "trust_level": self.trust_level,
            "published_at": self.published_at,
            "quotable": self.quotable,
            "detail": self.detail,
        }


@dataclass
class ExternalSearchStatus:
    """Outcome of one external search attempt."""

    status: str
    detail: str
    query: str = ""
    provider: str = ""
    results: list[ExternalResult] = field(default_factory=list)
    network_used: bool = False

    @property
    def ok(self) -> bool:
        return self.status == SEARCH_OK

    @property
    def trusted(self) -> list[ExternalResult]:
        return [item for item in self.results if item.quotable]

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "query": self.query,
            "provider": self.provider,
            "network_used": self.network_used,
            "results": [item.to_dict() for item in self.results],
        }


class TrustedSourcePolicy:
    """Admits, filters and ranks external results by trust."""

    def __init__(self, *, minimum: TrustLevel = MINIMUM_ANSWER_TRUST) -> None:
        self.minimum = minimum

    def admit(self, result: ExternalResult) -> ExternalResult:
        """Attach a trust verdict to ``result`` (never upgrades silently)."""
        level = classify_url(result.url)
        result.trust_level = level.value
        result.quotable = level.rank <= self.minimum.rank
        if not result.quotable:
            result.detail = (
                f"host trust {level.value} is below the quotable minimum "
                f"({self.minimum.value}): usable only as a knowledge candidate"
            )
        return result

    def filter(self, results: Iterable[ExternalResult], *, quotable_only: bool = True):
        """Admit every result, then optionally keep only the quotable ones."""
        admitted = [self.admit(item) for item in results]
        if not quotable_only:
            return admitted
        return [item for item in admitted if item.quotable]

    def rank(self, results: Iterable[ExternalResult]) -> list[ExternalResult]:
        """Rank by trust, then by whether the host is the curated registry."""
        return sorted(
            (self.admit(item) for item in results),
            key=lambda item: (
                TrustLevel(item.trust_level).rank,
                extract_host(item.url) not in TRUSTED_HOSTS,
                item.title.lower(),
            ),
        )


class ExternalSearchProvider(ABC):
    """Interface every external search backend implements."""

    name: str = "base"

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True only when a real, reachable backend is configured."""

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """True only when the user explicitly enabled external search."""

    @abstractmethod
    def search(self, query: str, *, top_k: int = 5) -> ExternalSearchStatus:
        """Search trusted sources (no network when disabled)."""

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Report whether external search could run."""


class DisabledExternalSearch(ExternalSearchProvider):
    """The shipped provider: policy-complete, network-free, truthful."""

    name = "disabled"

    def __init__(self, settings: Settings | None = None, reason: str = "") -> None:
        self._settings = settings or get_settings()
        self._reason = reason or (
            "External search is not implemented in this build, and AgroAI Core "
            "never opens a network connection on its own. Answers come from the "
            "local knowledge base; the trusted-source policy below is already "
            "enforced for when a provider is added."
        )

    @property
    def is_configured(self) -> bool:
        return False

    @property
    def is_enabled(self) -> bool:
        return bool(getattr(self._settings, "external_search_enabled", False))

    def search(self, query: str, *, top_k: int = 5) -> ExternalSearchStatus:
        return ExternalSearchStatus(
            status=SEARCH_NOT_CONFIGURED,
            detail=self._reason,
            query=query,
            provider=self.name,
            results=[],
            network_used=False,
        )

    def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "configured": False,
            "enabled": self.is_enabled,
            "available": False,
            "network_used": False,
            "detail": self._reason,
            "trusted_hosts": len(TRUSTED_HOSTS),
        }


class UnimplementedExternalSearch(ExternalSearchProvider):
    """Answers for a provider name that is configured but not built."""

    def __init__(self, provider: str, settings: Settings | None = None) -> None:
        self.name = provider
        self._settings = settings or get_settings()
        self._reason = (
            f"External search provider '{provider}' is not implemented in this "
            "build. No network call was made and no results were fabricated."
        )

    @property
    def is_configured(self) -> bool:
        return False

    @property
    def is_enabled(self) -> bool:
        return bool(getattr(self._settings, "external_search_enabled", False))

    def search(self, query: str, *, top_k: int = 5) -> ExternalSearchStatus:
        return ExternalSearchStatus(
            status=SEARCH_NOT_IMPLEMENTED,
            detail=self._reason,
            query=query,
            provider=self.name,
            results=[],
            network_used=False,
        )

    def health_check(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "configured": False,
            "enabled": self.is_enabled,
            "available": False,
            "network_used": False,
            "detail": self._reason,
        }


def get_external_search(settings: Settings | None = None) -> ExternalSearchProvider:
    """Return the configured external search provider.

    Every configuration resolves to a provider that makes no network call: the
    offline default, or an explicit "not implemented" answer.
    """
    settings = settings or get_settings()
    provider = (
        getattr(settings, "external_search_provider", "none") or "none"
    ).strip().lower()
    if provider in {"", "none", "disabled", "off"}:
        return DisabledExternalSearch(settings)
    return UnimplementedExternalSearch(provider, settings)


__all__ = [
    "DisabledExternalSearch",
    "ExternalResult",
    "ExternalSearchProvider",
    "ExternalSearchStatus",
    "MINIMUM_ANSWER_TRUST",
    "SEARCH_ERROR",
    "SEARCH_NOT_CONFIGURED",
    "SEARCH_NOT_IMPLEMENTED",
    "SEARCH_OK",
    "TRUSTED_HOSTS",
    "TRUSTED_SUFFIXES",
    "TrustedSourcePolicy",
    "UNTRUSTED_SUFFIXES",
    "UnimplementedExternalSearch",
    "classify_url",
    "extract_host",
    "get_external_search",
    "is_https",
    "is_trusted_url",
]
