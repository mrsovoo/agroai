"""Memory: two long-term memories plus the short-term conversation buffer.

| Memory | Where | What it holds |
| --- | --- | --- |
| Knowledge memory | :mod:`agroai.knowledge` | verified facts, sources, provenance |
| User memory | :mod:`agroai.memory.user_memory` | farm profile (consent-gated) |
| Episodic memory | :mod:`agroai.memory.episodic` | the user's own observations |
| Short-term | :mod:`agroai.agent.memory` | the current conversation (in-process) |

The rule the code enforces: a user's statement can become a **pending knowledge
candidate** (:mod:`agroai.knowledge.candidates`) but never knowledge. Only a
reviewer's approval promotes it, and only then does it reach the knowledge base.
"""

from __future__ import annotations

from agroai.memory.base import (
    ConsentRequired,
    MemoryEntry,
    MemoryKind,
    MemoryStore,
    summarize_entries,
)
from agroai.memory.episodic import Observation, ObservationLog, RecordedObservation
from agroai.memory.user_memory import (
    PROFILE_KEYS,
    InMemoryUserMemory,
    SQLiteUserMemory,
    UserProfile,
    get_user_memory,
)

__all__ = [
    "ConsentRequired",
    "InMemoryUserMemory",
    "MemoryEntry",
    "MemoryKind",
    "MemoryStore",
    "Observation",
    "ObservationLog",
    "PROFILE_KEYS",
    "RecordedObservation",
    "SQLiteUserMemory",
    "UserProfile",
    "get_user_memory",
    "summarize_entries",
]
