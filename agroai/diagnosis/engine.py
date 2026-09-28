"""The differential diagnosis engine.

```text
tomato leaf spots / yellowing
        |
        v
detect_symptoms()            <- what the user actually described
        |
        v
retriever.search(question)   <- evidence that exists in OUR knowledge base
        |
        v
score every retrieved document as a possible cause
        |
        v
rank -> observations that would discriminate between them
```

Two rules are enforced here:

1. **Nothing is invented.** A hypothesis is always a document that exists in the
   knowledge base; its excerpt is real text from that document.
2. **No verdict.** The result is a ranked differential with explicit confidence
   bands plus the observations that would sharpen it. When there is no evidence
   the engine refuses instead of guessing.

The scores combine evidence-level signals only (symptom overlap, source trust,
retrieval rank, provenance completeness). They are ranking aids - the docstring
of every public method says so, and :func:`render` prints it.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Sequence

from agroai.diagnosis.base import (
    MAX_HYPOTHESES,
    DifferentialDiagnosis,
    Hypothesis,
    detect_subject,
    detect_symptoms,
    required_observations,
)
from agroai.knowledge.metadata import TrustLevel
from agroai.retrieval.retriever import Retriever

logger = logging.getLogger(__name__)

#: Weight of each signal in the ranking score (documented, not a probability).
WEIGHT_SYMPTOM_MATCH = 0.35
WEIGHT_SOURCE_TRUST = 0.25
WEIGHT_RETRIEVAL_RANK = 0.20
WEIGHT_PROVENANCE = 0.20

#: Trust tier -> score contribution.
TRUST_WEIGHTS: dict[str, float] = {
    TrustLevel.TIER_1.value: 1.0,
    TrustLevel.TIER_2.value: 0.7,
    TrustLevel.TIER_3.value: 0.35,
    TrustLevel.UNKNOWN.value: 0.2,
}

EXCERPT_LENGTH = 320

NOTE_NOT_A_DIAGNOSIS = (
    "Bu ehtimolliklar reytingi; tasdiqlangan tashxis emas va laboratoriya "
    "tekshiruvi o'rnini bosmaydi."
)
NOTE_NO_EVIDENCE = (
    "AgroAI javobni faqat bilim bazasidagi dalilga asoslaydi; dalil topilmadi."
)


class DifferentialEngine(ABC):
    """Interface for ranking possible causes of an observed problem."""

    name: str = "base"

    @abstractmethod
    def differentiate(self, question: str, *, top_k: int = 8) -> DifferentialDiagnosis:
        """Return a ranked differential for ``question`` (never a verdict)."""


class RetrievalDifferentialEngine(DifferentialEngine):
    """Ranks hypotheses using retrievable knowledge only.

    Works with any :class:`~agroai.retrieval.retriever.Retriever`, including the
    default offline SQLite lexical retriever, so the differential is available
    today - the vector phase only makes it better, not possible.
    """

    name = "retrieval"

    def __init__(
        self,
        retriever: Retriever,
        *,
        max_hypotheses: int = MAX_HYPOTHESES,
        categories: Sequence[str] | None = None,
    ) -> None:
        self._retriever = retriever
        self._max_hypotheses = max(1, max_hypotheses)
        self._categories = list(categories or [])

    # ------------------------------------------------------------------ #
    def differentiate(self, question: str, *, top_k: int = 8) -> DifferentialDiagnosis:
        symptoms = detect_symptoms(question)
        subject = detect_subject(question)
        observations = required_observations(symptoms, subject)

        chunks = self._search(question, top_k)
        if not chunks:
            return DifferentialDiagnosis(
                question=question,
                subject=subject,
                symptoms=symptoms,
                hypotheses=[],
                required_observations=observations,
                refused=True,
                reason=(
                    "Bilim bazasida bu belgilarga mos hujjat topilmadi "
                    "(yoki baza bo'sh)."
                ),
                notes=[NOTE_NO_EVIDENCE],
            )

        hypotheses = self._rank(chunks, symptoms)
        if not hypotheses:
            return DifferentialDiagnosis(
                question=question,
                subject=subject,
                symptoms=symptoms,
                refused=True,
                reason="Topilgan hujjatlarda ehtimoliy sabab ajratilmadi.",
                required_observations=observations,
                notes=[NOTE_NO_EVIDENCE],
            )

        return DifferentialDiagnosis(
            question=question,
            subject=subject,
            symptoms=symptoms,
            hypotheses=hypotheses,
            required_observations=observations,
            refused=False,
            notes=[NOTE_NOT_A_DIAGNOSIS],
        )

    # ------------------------------------------------------------------ #
    def _search(self, question: str, top_k: int):
        try:
            return self._retriever.search(
                question, top_k=top_k, categories=self._categories or None
            )
        except TypeError:
            return self._retriever.search(question, top_k=top_k)
        except Exception as exc:  # noqa: BLE001 - a differential may never crash the agent
            logger.warning("Differential retrieval failed: %s", exc)
            return []

    def _rank(self, chunks, symptoms: list[str]) -> list[Hypothesis]:
        """Score each retrieved document as a possible cause (deduped by id)."""
        total = len(chunks)
        best: dict[str, Hypothesis] = {}
        for rank, retrieved in enumerate(chunks):
            document = retrieved.document
            document_id = retrieved.chunk.document_id
            cause = (
                (document.title if document is not None else "") or document_id
            ).strip()
            if not cause:
                continue

            content = retrieved.chunk.content or ""
            matched = [name for name in symptoms if self._mentions(content, name)]
            trust_level = (
                document.trust_level.value if document is not None else TrustLevel.UNKNOWN.value
            )
            source = ""
            if document is not None:
                source = document.source or document.organization() or ""
            if document is not None and document.source_url:
                source = f"{source} ({document.source_url})" if source else document.source_url

            confidence = self._score(
                matched_symptoms=len(matched),
                total_symptoms=len(symptoms),
                trust_level=trust_level,
                rank=rank,
                total=total,
                has_source=bool(source),
                has_document=document is not None,
            )
            candidate = Hypothesis(
                cause=cause,
                category=(
                    document.category if document is not None else ""
                ),
                confidence=confidence,
                document_id=document_id,
                trust_level=trust_level,
                source=source,
                matched_symptoms=matched,
                evidence=[retrieved.source_line],
                excerpt=self._excerpt(content),
            )
            existing = best.get(document_id)
            if existing is None or candidate.confidence > existing.confidence:
                best[document_id] = candidate

        ranked = sorted(best.values(), key=lambda item: (-item.confidence, item.cause))
        return ranked[: self._max_hypotheses]

    # ------------------------------------------------------------------ #
    @staticmethod
    def _mentions(content: str, symptom: str) -> bool:
        """True when ``content`` discusses ``symptom`` (lexicon-declared words)."""
        from agroai.diagnosis.base import SYMPTOM_LEXICON

        lowered = (content or "").lower()
        return any(word in lowered for word in SYMPTOM_LEXICON.get(symptom, ()))

    @staticmethod
    def _score(
        *,
        matched_symptoms: int,
        total_symptoms: int,
        trust_level: str,
        rank: int,
        total: int,
        has_source: bool,
        has_document: bool,
    ) -> float:
        """Ranking score in [0, 1]. Documented as an aid, not a probability."""
        symptom_score = (
            matched_symptoms / total_symptoms if total_symptoms else 0.0
        )
        trust_score = TRUST_WEIGHTS.get(trust_level, TRUST_WEIGHTS[TrustLevel.UNKNOWN.value])
        rank_score = 1.0 - (rank / total) if total else 0.0
        provenance_score = 0.0
        if has_source:
            provenance_score += 0.5
        if has_document:
            provenance_score += 0.5
        return (
            WEIGHT_SYMPTOM_MATCH * symptom_score
            + WEIGHT_SOURCE_TRUST * trust_score
            + WEIGHT_RETRIEVAL_RANK * rank_score
            + WEIGHT_PROVENANCE * provenance_score
        )

    @staticmethod
    def _excerpt(content: str) -> str:
        text = " ".join((content or "").split())
        if len(text) <= EXCERPT_LENGTH:
            return text
        return text[: EXCERPT_LENGTH - 3] + "..."


__all__ = [
    "DifferentialEngine",
    "NOTE_NO_EVIDENCE",
    "NOTE_NOT_A_DIAGNOSIS",
    "RetrievalDifferentialEngine",
    "TRUST_WEIGHTS",
    "WEIGHT_PROVENANCE",
    "WEIGHT_RETRIEVAL_RANK",
    "WEIGHT_SOURCE_TRUST",
    "WEIGHT_SYMPTOM_MATCH",
]
