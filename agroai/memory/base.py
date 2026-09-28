"""Long-term memory: the contract, consent gating and entry types.

The roadmap distinguishes two memories, and the distinction is enforced by types
rather than by convention:

* **Knowledge memory** - verified facts, sources and provenance. That is the
  knowledge base (:mod:`agroai.knowledge`); it is never written by a chat.
* **User memory** - the farm profile and the user's own experiences
  (:mod:`agroai.memory.user_memory`, :mod:`agroai.memory.episodic`). It is
  personal data, so it is **opt-in**: every write requires explicit consent, and
  forgetting is a first-class operation.

Short-term conversation context is a third, separate thing and lives in
:mod:`agroai.agent.memory` (in-process, bounded, discarded with the session).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from agroai.knowledge.metadata import utcnow_iso


class MemoryKind(str, Enum):
    """What kind of entry is being stored."""

    PROFILE = "profile"          # "farm size 12 ha", "region Sirdaryo"
    OBSERVATION = "observation"  # "I saw yellow leaves after the rain"
    PREFERENCE = "preference"    # "answer in Uzbek", "no chemicals if possible"


class ConsentRequired(RuntimeError):
    """Raised when memory is written without the user's consent."""


@dataclass
class MemoryEntry:
    """One remembered fact about a user."""

    user_id: str
    key: str
    value: str
    kind: MemoryKind = MemoryKind.PROFILE
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)

    @property
    def id(self) -> str:
        return f"{self.user_id}:{self.kind.value}:{self.key}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "kind": self.kind.value,
            "key": self.key,
            "value": self.value,
            "metadata": dict(self.metadata),
            "updated_at": self.updated_at,
        }


class MemoryStore(ABC):
    """Interface for consent-gated per-user memory."""

    def __init__(self, *, consent: bool = False) -> None:
        self._consent = bool(consent)

    @property
    def consent_granted(self) -> bool:
        return self._consent

    def grant_consent(self) -> None:
        """Record the user's explicit permission to remember their data."""
        self._consent = True

    def revoke_consent(self, *, forget_all: bool = True, owner: str = "") -> int:
        """Withdraw consent; optionally erase what was stored."""
        self._consent = False
        if forget_all and owner:
            return self.forget(owner)
        return 0

    def _require_consent(self) -> None:
        if not self._consent:
            raise ConsentRequired(
                "User memory is opt-in: no data is stored without explicit "
                "consent. Call grant_consent() (or set AGROAI_MEMORY_CONSENT=true) "
                "after asking the user."
            )

    # ------------------------------------------------------------------ #
    @abstractmethod
    def remember(
        self,
        user_id: str,
        key: str,
        value: str,
        *,
        kind: MemoryKind = MemoryKind.PROFILE,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        """Store one entry (requires consent)."""

    @abstractmethod
    def recall(
        self, user_id: str, *, kind: MemoryKind | None = None, key: str | None = None
    ) -> list[MemoryEntry]:
        """Read stored entries (never raises when empty)."""

    @abstractmethod
    def forget(self, user_id: str, *, kind: MemoryKind | None = None, key: str | None = None) -> int:
        """Erase entries. Returns how many were removed."""


def summarize_entries(entries: list[MemoryEntry], limit: int = 12) -> str:
    """Render memory entries as a short, bounded context block."""
    if not entries:
        return ""
    lines = ["FOYDALANUVCHI KONTEKSTI / USER CONTEXT (from user memory):"]
    for entry in entries[:limit]:
        lines.append(f"- {entry.key}: {entry.value}")
    return "\n".join(lines)


__all__ = [
    "ConsentRequired",
    "MemoryEntry",
    "MemoryKind",
    "MemoryStore",
    "summarize_entries",
]
