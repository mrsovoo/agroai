"""User memory: the farm profile and what we are allowed to remember.

Two backends implement :class:`~agroai.memory.base.MemoryStore`:

* :class:`InMemoryUserMemory` - the whole session's memory, dropped on exit;
* :class:`SQLiteUserMemory` - persists locally, still only with consent.

:class:`UserProfile` is the structured view of that memory: location, farm size,
crops, livestock, irrigation and soil. It is what makes the second question
("what should I do for my tomato?") work without the user repeating everything -
and it is exactly the data the user must agree to store, so it lives behind the
consent gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from agroai.memory.base import (
    MemoryEntry,
    MemoryKind,
    MemoryStore,
)

#: Profile keys AgroAI understands (anything else is stored under "extra").
PROFILE_KEYS: tuple[str, ...] = (
    "location",
    "region",
    "country",
    "farm_size_ha",
    "crops",
    "livestock",
    "irrigation",
    "soil",
    "language",
    "notes",
)


def _split_list(value: str) -> list[str]:
    return [item.strip() for item in (value or "").replace(";", ",").split(",") if item.strip()]


@dataclass
class UserProfile:
    """Structured farm context for one user."""

    user_id: str
    location: str = ""
    region: str = ""
    country: str = ""
    farm_size_ha: float | None = None
    crops: list[str] = field(default_factory=list)
    livestock: list[str] = field(default_factory=list)
    irrigation: str = ""
    soil: str = ""
    language: str = ""
    notes: str = ""
    extra: dict[str, str] = field(default_factory=dict)

    @property
    def is_empty(self) -> bool:
        return not any(
            (
                self.location,
                self.region,
                self.country,
                self.farm_size_ha,
                self.crops,
                self.livestock,
                self.irrigation,
                self.soil,
                self.notes,
            )
        )

    def as_facts(self) -> dict[str, str]:
        """Return the profile as flat key/value facts (what gets stored)."""
        facts: dict[str, str] = {}
        for key in PROFILE_KEYS:
            value = getattr(self, key, None)
            if value in (None, "", []):
                continue
            facts[key] = ", ".join(value) if isinstance(value, list) else str(value)
        facts.update({key: value for key, value in self.extra.items() if value})
        return facts

    @classmethod
    def from_entries(cls, user_id: str, entries: Iterable[MemoryEntry]) -> "UserProfile":
        """Build a profile from stored memory entries."""
        facts: dict[str, str] = {}
        extra: dict[str, str] = {}
        for entry in entries:
            if entry.kind is not MemoryKind.PROFILE:
                continue
            if entry.key in PROFILE_KEYS:
                facts[entry.key] = entry.value
            else:
                extra[entry.key] = entry.value
        size = facts.get("farm_size_ha")
        return cls(
            user_id=user_id,
            location=facts.get("location", ""),
            region=facts.get("region", ""),
            country=facts.get("country", ""),
            farm_size_ha=float(size) if size and _is_number(size) else None,
            crops=_split_list(facts.get("crops", "")),
            livestock=_split_list(facts.get("livestock", "")),
            irrigation=facts.get("irrigation", ""),
            soil=facts.get("soil", ""),
            language=facts.get("language", ""),
            notes=facts.get("notes", ""),
            extra=extra,
        )

    def prompt_context(self) -> str:
        """Bounded summary of the profile for the prompt (empty when unknown)."""
        if self.is_empty:
            return ""
        lines = ["FERMER KONTEKSTI / FARM CONTEXT (remembered, user-provided):"]
        for key, value in self.as_facts().items():
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "user_id": self.user_id,
            "location": self.location,
            "region": self.region,
            "country": self.country,
            "farm_size_ha": self.farm_size_ha,
            "crops": list(self.crops),
            "livestock": list(self.livestock),
            "irrigation": self.irrigation,
            "soil": self.soil,
            "language": self.language,
            "notes": self.notes,
            "extra": dict(self.extra),
        }


def _is_number(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


class InMemoryUserMemory(MemoryStore):
    """Session-scoped user memory (nothing is written to disk)."""

    def __init__(self, *, consent: bool = False) -> None:
        super().__init__(consent=consent)
        self._entries: dict[str, MemoryEntry] = {}

    def remember(
        self,
        user_id: str,
        key: str,
        value: str,
        *,
        kind: MemoryKind = MemoryKind.PROFILE,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        self._require_consent()
        entry = MemoryEntry(
            user_id=user_id,
            key=key.strip(),
            value=str(value).strip(),
            kind=kind,
            metadata=dict(metadata or {}),
        )
        if not entry.key:
            raise ValueError("memory key must not be empty")
        self._entries[entry.id] = entry
        return entry

    def recall(
        self, user_id: str, *, kind: MemoryKind | None = None, key: str | None = None
    ) -> list[MemoryEntry]:
        return [
            entry
            for entry in self._entries.values()
            if entry.user_id == user_id
            and (kind is None or entry.kind is kind)
            and (key is None or entry.key == key)
        ]

    def forget(self, user_id: str, *, kind: MemoryKind | None = None, key: str | None = None) -> int:
        doomed = [
            entry_id
            for entry_id, entry in self._entries.items()
            if entry.user_id == user_id
            and (kind is None or entry.kind is kind)
            and (key is None or entry.key == key)
        ]
        for entry_id in doomed:
            del self._entries[entry_id]
        return len(doomed)

    def profile(self, user_id: str) -> UserProfile:
        return UserProfile.from_entries(user_id, self.recall(user_id, kind=MemoryKind.PROFILE))

    def save_profile(self, profile: UserProfile) -> UserProfile:
        """Write every non-empty profile fact (requires consent)."""
        for key, value in profile.as_facts().items():
            self.remember(profile.user_id, key, value, kind=MemoryKind.PROFILE)
        return profile


class SQLiteUserMemory(MemoryStore):
    """Local, consent-gated user memory persisted in SQLite."""

    def __init__(self, database: Any, *, consent: bool = False) -> None:
        super().__init__(consent=consent)
        self._db = database

    def remember(
        self,
        user_id: str,
        key: str,
        value: str,
        *,
        kind: MemoryKind = MemoryKind.PROFILE,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry:
        self._require_consent()
        clean_key = key.strip()
        if not clean_key:
            raise ValueError("memory key must not be empty")
        self._db.set_user_memory(
            user_id, clean_key, str(value).strip(), kind=kind.value, metadata=metadata
        )
        return MemoryEntry(
            user_id=user_id,
            key=clean_key,
            value=str(value).strip(),
            kind=kind,
            metadata=dict(metadata or {}),
        )

    def recall(
        self, user_id: str, *, kind: MemoryKind | None = None, key: str | None = None
    ) -> list[MemoryEntry]:
        rows = self._db.list_user_memory(user_id, kind=kind.value if kind else None)
        entries: list[MemoryEntry] = []
        for row in rows:
            try:
                entry_kind = MemoryKind(row["kind"])
            except ValueError:
                entry_kind = MemoryKind.PROFILE
            if key is not None and row["key"] != key:
                continue
            entries.append(
                MemoryEntry(
                    user_id=user_id,
                    key=row["key"],
                    value=row["value"],
                    kind=entry_kind,
                    metadata=row["metadata"],
                    created_at=row["created_at"],
                    updated_at=row["updated_at"],
                )
            )
        return entries

    def forget(self, user_id: str, *, kind: MemoryKind | None = None, key: str | None = None) -> int:
        return self._db.delete_user_memory(
            user_id, kind=kind.value if kind else None, key=key
        )


def get_user_memory(settings: Any | None = None, database: Any | None = None) -> MemoryStore:
    """Build the user-memory store for the current configuration.

    Consent defaults to **off**: without `AGROAI_MEMORY_CONSENT=true` any write
    raises :class:`ConsentRequired`. With a database and consent, memory
    persists locally; otherwise it stays in the session.
    """
    consent = bool(getattr(settings, "memory_consent", False)) if settings is not None else False
    if database is not None:
        return SQLiteUserMemory(database, consent=consent)
    return InMemoryUserMemory(consent=consent)


__all__ = [
    "InMemoryUserMemory",
    "PROFILE_KEYS",
    "SQLiteUserMemory",
    "UserProfile",
    "get_user_memory",
]
