"""Episodic memory: what the user experienced or reported.

The roadmap's rule is precise here: *"I applied this and the disease went away"*
is a useful **observation**, not the fact *"this product always cures this
disease"*. So an observation:

* is stored as user memory (scoped to the user, consent-gated),
* is turned into a **pending knowledge candidate** with an evidence level,
* is never written into the knowledge base by itself - a reviewer decides
  (:func:`agroai.knowledge.candidates.approve`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agroai.knowledge.candidates import (
    CandidatePipeline,
    EvidenceLevel,
    KnowledgeCandidate,
    guess_evidence_level,
)
from agroai.knowledge.metadata import content_hash, utcnow_iso
from agroai.memory.base import MemoryEntry, MemoryKind, MemoryStore


@dataclass
class Observation:
    """Something the user reported from their own farm."""

    user_id: str
    statement: str
    crop: str = ""
    animal: str = ""
    region: str = ""
    country: str = ""
    category: str = "general"
    language: str = "uz"
    reported_at: str = ""
    evidence_level: EvidenceLevel = EvidenceLevel.OBSERVATION
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return f"obs_{content_hash(self.user_id, self.statement)[:12]}"

    def to_candidate(self) -> KnowledgeCandidate:
        """Represent the observation as an unverified knowledge candidate."""
        return KnowledgeCandidate(
            id=f"cand_{self.id[4:]}",
            statement=self.statement,
            category=self.category,
            crop=self.crop,
            animal=self.animal,
            region=self.region,
            country=self.country,
            language=self.language,
            source="user observation",
            author=self.user_id,
            reported_at=self.reported_at or utcnow_iso(),
            evidence_level=self.evidence_level,
            metadata={
                **self.metadata,
                "origin": "episodic_memory",
                "user_id": self.user_id,
                "observation_id": self.id,
            },
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "user_id": self.user_id,
            "statement": self.statement,
            "crop": self.crop,
            "animal": self.animal,
            "region": self.region,
            "evidence_level": self.evidence_level.value,
            "reported_at": self.reported_at,
        }


@dataclass
class RecordedObservation:
    """The result of recording an observation."""

    observation: Observation
    entry: MemoryEntry | None = None
    candidate: KnowledgeCandidate | None = None

    @property
    def stored_in_memory(self) -> bool:
        return self.entry is not None

    @property
    def became_knowledge(self) -> bool:
        """Always False: an observation never enters the knowledge base alone."""
        return False


class ObservationLog:
    """Records observations and offers them for review as candidates."""

    def __init__(
        self,
        memory: MemoryStore,
        candidates: CandidatePipeline | None = None,
    ) -> None:
        self._memory = memory
        self._candidates = candidates

    @property
    def memory(self) -> MemoryStore:
        return self._memory

    def record(self, observation: Observation) -> RecordedObservation:
        """Store the observation and emit a pending candidate.

        Raises :class:`~agroai.memory.base.ConsentRequired` when the user has not
        allowed memory writes - nothing is stored silently.
        """
        statement = (observation.statement or "").strip()
        if not statement:
            raise ValueError("an observation needs a statement")
        observation.statement = statement
        if not observation.reported_at:
            observation.reported_at = utcnow_iso()
        if observation.evidence_level is EvidenceLevel.OBSERVATION:
            # Wording may justify a stronger level, but never a weaker one.
            observation.evidence_level = guess_evidence_level(statement)

        entry = self._memory.remember(
            observation.user_id,
            observation.id,
            statement,
            kind=MemoryKind.OBSERVATION,
            metadata=observation.to_dict(),
        )

        candidate = observation.to_candidate()
        if self._candidates is not None:
            candidate = self._candidates.submit(candidate)
        return RecordedObservation(observation=observation, entry=entry, candidate=candidate)

    def list(self, user_id: str) -> list[Observation]:
        """Return the observations remembered for one user."""
        observations: list[Observation] = []
        for entry in self._memory.recall(user_id, kind=MemoryKind.OBSERVATION):
            metadata = dict(entry.metadata)
            observations.append(
                Observation(
                    user_id=user_id,
                    statement=entry.value,
                    crop=str(metadata.get("crop", "")),
                    animal=str(metadata.get("animal", "")),
                    region=str(metadata.get("region", "")),
                    country=str(metadata.get("country", "")),
                    reported_at=str(metadata.get("reported_at", "")),
                    metadata=metadata,
                )
            )
        return observations

    def forget(self, user_id: str) -> int:
        """Erase a user's observations (memory writes stay consent-gated)."""
        return self._memory.forget(user_id, kind=MemoryKind.OBSERVATION)


__all__ = ["Observation", "ObservationLog", "RecordedObservation"]
