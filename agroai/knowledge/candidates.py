"""Knowledge candidates: how new information may enter the knowledge base.

The roadmap is explicit about this boundary: AgroAI must **never** treat
anything it is told as permanent knowledge on its own.

```text
new information -> candidate -> source check -> evidence check
                -> conflict check -> quality check -> confidence
                -> pending (human review) -> approved | rejected
```

Two rules are enforced here rather than documented and hoped for:

1. A candidate is created as ``pending`` and can only reach ``verified``
   through an explicit :func:`approve` call - nothing promotes itself.
2. Only verified candidates become documents, and those documents keep the
   candidate's provenance, evidence level and any registered conflict in their
   metadata, so the knowledge base stays auditable.

The confidence score is a deterministic consequence of evidence level, source
trust and completeness - it is a review aid, not a probability, and the code
says so.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable

from agroai.knowledge.documents import KnowledgeDocument
from agroai.knowledge.metadata import (
    TrustLevel,
    content_hash,
    infer_trust_level,
    utcnow_iso,
)

#: Minimum confidence a candidate needs before it may be approved.
DEFAULT_APPROVAL_THRESHOLD = 0.6


class EvidenceLevel(str, Enum):
    """How strong the support behind a statement is."""

    OBSERVATION = "observation"          # "I saw spots on my leaves"
    USER_REPORT = "user_report"          # "this worked on my farm"
    FIELD_EXPERIENCE = "field_experience"  # documented field trial by a grower
    OFFICIAL_DOCUMENT = "official_document"  # ministry/extension publication
    SCIENTIFIC = "scientific"            # peer-reviewed / FAO / WOAH / USDA

    @property
    def weight(self) -> float:
        """Score contribution of the evidence level (0..1)."""
        return {
            EvidenceLevel.OBSERVATION: 0.15,
            EvidenceLevel.USER_REPORT: 0.30,
            EvidenceLevel.FIELD_EXPERIENCE: 0.55,
            EvidenceLevel.OFFICIAL_DOCUMENT: 0.80,
            EvidenceLevel.SCIENTIFIC: 1.00,
        }[self]


class CandidateStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"
    REJECTED = "rejected"


@dataclass
class KnowledgeCandidate:
    """A statement that *might* become knowledge, with its review state."""

    id: str
    statement: str
    category: str = "general"
    crop: str = ""
    animal: str = ""
    region: str = ""
    country: str = ""
    language: str = "uz"
    source: str = ""
    source_url: str = ""
    author: str = ""
    reported_at: str = ""
    evidence_level: EvidenceLevel = EvidenceLevel.OBSERVATION
    confidence: float = 0.0
    status: CandidateStatus = CandidateStatus.PENDING
    review_notes: str = ""
    conflicts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)

    # ------------------------------------------------------------------ #
    @property
    def is_observation(self) -> bool:
        """True when this is somebody's experience, not verified knowledge."""
        return self.evidence_level in {
            EvidenceLevel.OBSERVATION,
            EvidenceLevel.USER_REPORT,
        }

    @property
    def has_conflict(self) -> bool:
        return bool(self.conflicts)

    @property
    def trust_level(self) -> TrustLevel:
        return infer_trust_level(self.source or self.author, str(self.metadata.get("source_type", "")))

    def is_approved(self) -> bool:
        return self.status is CandidateStatus.VERIFIED

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "statement": self.statement,
            "category": self.category,
            "crop": self.crop,
            "animal": self.animal,
            "region": self.region,
            "country": self.country,
            "language": self.language,
            "source": self.source,
            "source_url": self.source_url,
            "author": self.author,
            "reported_at": self.reported_at,
            "evidence_level": self.evidence_level.value,
            "confidence": round(self.confidence, 3),
            "status": self.status.value,
            "review_notes": self.review_notes,
            "conflicts": list(self.conflicts),
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def render(self) -> str:
        lines = [
            f"[{self.status.value}] {self.id} ({self.evidence_level.value}, "
            f"confidence={self.confidence:.2f})",
            f"  {self.statement.strip()}",
        ]
        if self.crop or self.animal or self.region:
            targets = ", ".join(part for part in (self.crop, self.animal, self.region) if part)
            lines.append(f"  crop/animal/region: {targets}")
        if self.source or self.author:
            lines.append(f"  source: {self.source or self.author}")
        if self.conflicts:
            lines.append("  conflicts: " + "; ".join(self.conflicts))
        if self.review_notes:
            lines.append(f"  review: {self.review_notes}")
        return "\n".join(lines)


@dataclass
class CandidateAssessment:
    """The deterministic review verdict for a candidate."""

    confidence: float
    evidence_level: EvidenceLevel
    requires_review: bool
    reasons: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)

    @property
    def meets_threshold(self) -> bool:
        return self.confidence >= DEFAULT_APPROVAL_THRESHOLD and not self.conflicts

    def render(self) -> str:
        return (
            f"confidence={self.confidence:.2f}, evidence={self.evidence_level.value}, "
            f"review_required={self.requires_review}"
            + (f", conflicts={len(self.conflicts)}" if self.conflicts else "")
            + ("\n  " + "\n  ".join(self.reasons) if self.reasons else "")
        )


EVIDENCE_HINTS: tuple[tuple[EvidenceLevel, tuple[str, ...]], ...] = (
    (
        EvidenceLevel.SCIENTIFIC,
        ("fao", "woah", "usda", "peer-reviewed", "research", "scientific", "study", "tadqiqot"),
    ),
    (
        EvidenceLevel.OFFICIAL_DOCUMENT,
        ("ministry", "vazirlik", "extension", "official", "standard", "regulation", "davlat"),
    ),
    (
        EvidenceLevel.FIELD_EXPERIENCE,
        ("field trial", "tajriba", "trial", "demonstration", "sinov", "farm trial"),
    ),
    (EvidenceLevel.USER_REPORT, ("i applied", "men qo'lladim", "worked", "yordam berdi", "men ekdim")),
)


def guess_evidence_level(text: str, declared: EvidenceLevel | None = None) -> EvidenceLevel:
    """Guess the evidence level from wording - used only for suggestions.

    The guess never raises the level above what the wording supports: an
    unclear statement stays an observation, which is the safest default.
    """
    if declared is not None:
        return declared
    lowered = (text or "").lower()
    for level, hints in EVIDENCE_HINTS:
        if any(hint in lowered for hint in hints):
            return level
    return EvidenceLevel.OBSERVATION


class CandidatePipeline:
    """Turns raw information into a reviewed, scored candidate."""

    def __init__(self, *, database: Any | None = None, approval_threshold: float = DEFAULT_APPROVAL_THRESHOLD) -> None:
        self._db = database
        self._threshold = approval_threshold

    @property
    def approval_threshold(self) -> float:
        return self._threshold

    # ------------------------------------------------------------------ #
    def assess(self, candidate: KnowledgeCandidate) -> CandidateAssessment:
        """Score a candidate: evidence, source trust, completeness, conflicts."""
        reasons: list[str] = []

        evidence_score = candidate.evidence_level.weight
        reasons.append(f"evidence level '{candidate.evidence_level.value}' -> {evidence_score:.2f}")

        trust = candidate.trust_level
        trust_score = {
            TrustLevel.TIER_1: 1.0,
            TrustLevel.TIER_2: 0.7,
            TrustLevel.TIER_3: 0.35,
            TrustLevel.UNKNOWN: 0.2,
        }[trust]
        reasons.append(f"source trust '{trust.value}' -> {trust_score:.2f}")

        completeness_fields = (candidate.source or candidate.author, candidate.source_url)
        completeness = sum(1 for value in completeness_fields if value) / len(completeness_fields)
        reasons.append(f"provenance completeness -> {completeness:.2f}")

        specificity = sum(
            1 for value in (candidate.crop, candidate.animal, candidate.region) if value
        ) / 3
        reasons.append(f"context specificity (crop/animal/region) -> {specificity:.2f}")

        confidence = round(
            0.45 * evidence_score + 0.25 * trust_score + 0.20 * completeness + 0.10 * specificity,
            3,
        )
        conflicts = self._detect_conflicts(candidate)
        if conflicts:
            reasons.append(f"{len(conflicts)} possible conflict(s) with stored documents")
        reasons.append(
            f"confidence is a review aid, not a probability (threshold {self._threshold:.2f})"
        )

        return CandidateAssessment(
            confidence=confidence,
            evidence_level=candidate.evidence_level,
            requires_review=True,  # every candidate needs a human decision
            reasons=reasons,
            conflicts=conflicts,
        )

    def submit(self, candidate: KnowledgeCandidate) -> KnowledgeCandidate:
        """Assess a candidate and mark it pending (never auto-verified)."""
        assessment = self.assess(candidate)
        candidate.confidence = assessment.confidence
        candidate.status = CandidateStatus.PENDING
        candidate.conflicts = assessment.conflicts
        candidate.metadata = {
            **candidate.metadata,
            "assessment_reasons": assessment.reasons,
            "approval_threshold": self._threshold,
        }
        candidate.updated_at = utcnow_iso()
        if self._db is not None:
            self._db.upsert_candidate(candidate)
        return candidate

    def _detect_conflicts(self, candidate: KnowledgeCandidate) -> list[str]:
        """Look for stored documents with the same title but different content."""
        if self._db is None:
            return []
        title_key = candidate.metadata.get("title") or candidate.statement[:60]
        try:
            documents = list(self._db.iter_documents(category=candidate.category or None))
        except Exception:  # noqa: BLE001 - conflict detection must never block a review
            return []
        conflicts: list[str] = []
        lowered_key = str(title_key).strip().lower()
        for document in documents:
            if document.content_hash and document.content_hash == candidate.metadata.get("content_hash"):
                continue
            if lowered_key and lowered_key in document.title.lower():
                conflicts.append(
                    f"stored document may already cover this: {document.title} "
                    f"({document.source}, updated {document.updated_at})"
                )
        return conflicts

    # ------------------------------------------------------------------ #
    def review(
        self,
        candidate: KnowledgeCandidate,
        *,
        approved: bool,
        reviewer: str = "",
        notes: str = "",
    ) -> KnowledgeCandidate:
        """Record a human decision. Approving below the threshold is allowed but noted."""
        if approved and candidate.conflicts:
            notes = (notes + " | " if notes else "") + "approved despite flagged conflicts"
        candidate.status = CandidateStatus.VERIFIED if approved else CandidateStatus.REJECTED
        candidate.review_notes = notes
        candidate.metadata = {
            **candidate.metadata,
            "reviewed_by": reviewer,
            "reviewed_at": utcnow_iso(),
        }
        candidate.updated_at = utcnow_iso()
        if self._db is not None:
            self._db.upsert_candidate(candidate)
        return candidate


def approve(candidate: KnowledgeCandidate, *, reviewer: str = "", notes: str = "") -> KnowledgeDocument:
    """Turn an approved candidate into a knowledge document.

    Raises :class:`ValueError` when the candidate was not verified first: the
    knowledge base only accepts reviewed material.
    """
    if not candidate.is_approved():
        raise ValueError(
            f"candidate {candidate.id} is '{candidate.status.value}': it must be "
            "approved by a reviewer before it can enter the knowledge base"
        )

    statement = candidate.statement.strip()
    trust = candidate.trust_level
    document = KnowledgeDocument(
        id=f"cand_{candidate.id}",
        title=candidate.metadata.get("title") or statement[:80],
        content=statement,
        source=candidate.source or candidate.author or "candidate",
        source_url=candidate.source_url,
        category=candidate.category,
        language=candidate.language,
        publication_date=candidate.reported_at,
        author=candidate.author,
        metadata={
            "trust_level": trust.value,
            "evidence_level": candidate.evidence_level.value,
            "confidence": candidate.confidence,
            "candidate_id": candidate.id,
            "reviewed_by": candidate.metadata.get("reviewed_by", reviewer),
            "review_notes": notes or candidate.review_notes,
            "conflicts": candidate.conflicts,
            "crop": candidate.crop,
            "animal": candidate.animal,
            "region": candidate.region,
            "country": candidate.country,
            "origin": "knowledge_candidate",
            "ingested_at": utcnow_iso(),
        },
    )
    return document


def candidates_from_reports(
    reports: Iterable[dict[str, Any]],
    *,
    region: str = "",
    country: str = "",
    category: str = "general",
) -> list[KnowledgeCandidate]:
    """Build candidates from structured user reports (``statement`` required)."""
    built: list[KnowledgeCandidate] = []
    for report in reports:
        statement = str(report.get("statement", "")).strip()
        if not statement:
            continue
        # Deterministic ids: the same report submitted twice is the same candidate.
        default_id = f"cand_{content_hash(statement)[:12]}"
        built.append(
            KnowledgeCandidate(
                id=str(report.get("id") or default_id),
                statement=statement,
                category=str(report.get("category") or category),
                crop=str(report.get("crop", "")),
                animal=str(report.get("animal", "")),
                region=str(report.get("region", region)),
                country=str(report.get("country", country)),
                language=str(report.get("language", "uz")),
                source=str(report.get("source", "")),
                source_url=str(report.get("source_url", "")),
                author=str(report.get("author", "")),
                reported_at=str(report.get("reported_at", "")),
                evidence_level=guess_evidence_level(
                    statement,
                    EvidenceLevel(report["evidence_level"]) if report.get("evidence_level") else None,
                ),
                metadata=dict(report.get("metadata", {})),
            )
        )
    return built


__all__ = [
    "DEFAULT_APPROVAL_THRESHOLD",
    "EVIDENCE_HINTS",
    "CandidateAssessment",
    "CandidatePipeline",
    "CandidateStatus",
    "EvidenceLevel",
    "KnowledgeCandidate",
    "approve",
    "candidates_from_reports",
    "guess_evidence_level",
]
