"""Automatic knowledge updates - interface and honest default.

The roadmap's Phase 10 is "trusted sources -> crawler/API -> new documents? ->
hash comparison -> cleaning -> validation -> knowledge base update -> embedding
update", with the rule that a changed document keeps **both** the old and the new
version.

That phase is deliberately **not** implemented: AgroAI Core never opens a
network connection on its own. What exists here is the contract it will be built
against, plus a :class:`DisabledUpdater` whose answers are always truthful.

Enable-ability is explicit: an updater only ever runs when a provider is
configured *and* the caller asks for it. :func:`get_updater` returns the
disabled updater for every configuration that exists today.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agroai.config.settings import Settings, get_settings
from agroai.sources.registry import TRUSTED_SOURCES, TrustedSource

#: Status values an update run can report.
NOT_IMPLEMENTED = "not_implemented"
NOT_CONFIGURED = "not_configured"
OK = "ok"

#: Default opt-in flag for any future updater.
DEFAULT_ENABLED = False


@dataclass
class UpdateResult:
    """Honest outcome of an update attempt."""

    status: str
    detail: str
    checked: int = 0
    added: int = 0
    updated: int = 0
    skipped: int = 0
    network_used: bool = False
    sources: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return self.status == OK

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "detail": self.detail,
            "checked": self.checked,
            "added": self.added,
            "updated": self.updated,
            "skipped": self.skipped,
            "network_used": self.network_used,
            "sources": list(self.sources),
        }


class KnowledgeUpdater(ABC):
    """Interface for keeping the knowledge base in step with trusted sources.

    Implementations must be: opt-in, incremental, deduplicated by content hash,
    provenance-preserving (a changed document keeps its previous version) and
    fully offline unless the user has explicitly configured a provider.
    """

    name: str = "base"

    @property
    @abstractmethod
    def is_enabled(self) -> bool:
        """True only when updates are configured *and* explicitly enabled."""

    @abstractmethod
    def candidate_sources(self) -> list[TrustedSource]:
        """The registry entries this updater would consult."""

    @abstractmethod
    def check_updates(self) -> UpdateResult:
        """Report what new material appears to exist, without importing it."""

    @abstractmethod
    def apply_updates(self, *, dry_run: bool = True) -> UpdateResult:
        """Import validated new material (``dry_run`` never writes)."""


class DisabledUpdater(KnowledgeUpdater):
    """The only updater that exists today: it refuses to pretend.

    It still answers the useful, offline questions - which sources *would* be
    consulted, and why nothing is happening yet - so the CLI and doctor can
    report a real state instead of a vague one.
    """

    name = "disabled"

    def __init__(self, settings: Settings | None = None, reason: str = "") -> None:
        self._settings = settings or get_settings()
        self._reason = reason or (
            "Automatic knowledge updates are not implemented in this phase. "
            "The knowledge base is only changed by explicit 'agroai ingest' "
            "commands, and AgroAI Core never reaches the network on its own."
        )

    @property
    def is_enabled(self) -> bool:
        return DEFAULT_ENABLED

    def candidate_sources(self) -> list[TrustedSource]:
        return list(TRUSTED_SOURCES)

    def check_updates(self) -> UpdateResult:
        return UpdateResult(
            status=NOT_IMPLEMENTED,
            detail=self._reason,
            checked=0,
            sources=[source.name for source in self.candidate_sources()],
            network_used=False,
        )

    def apply_updates(self, *, dry_run: bool = True) -> UpdateResult:
        result = self.check_updates()
        result.detail += (
            " Nothing was written (dry run, no updater configured)."
            if dry_run
            else " Nothing was written: no updater is configured."
        )
        return result


def get_updater(settings: Settings | None = None) -> KnowledgeUpdater:
    """Return the configured knowledge updater.

    Only the disabled implementation exists; any other requested provider is
    reported through the returned object's ``reason`` instead of silently
    pretending an updater exists.
    """
    settings = settings or get_settings()
    provider = (getattr(settings, "updater_provider", "none") or "none").lower()
    if provider in {"none", "disabled", ""}:
        return DisabledUpdater(settings)
    return DisabledUpdater(
        settings,
        reason=(
            f"Updater provider '{provider}' is not implemented. Only 'none' is "
            "available; the knowledge base changes only via explicit ingestion."
        ),
    )


__all__ = [
    "DEFAULT_ENABLED",
    "DisabledUpdater",
    "KnowledgeUpdater",
    "NOT_CONFIGURED",
    "NOT_IMPLEMENTED",
    "OK",
    "UpdateResult",
    "get_updater",
]
