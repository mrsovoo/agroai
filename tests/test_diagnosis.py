"""Differential diagnosis: symptom detection, ranking, and honest refusal."""

from __future__ import annotations

import pytest

from agroai.diagnosis import (
    BAND_LOW,
    DifferentialDiagnosis,
    RetrievalDifferentialEngine,
    confidence_band,
    detect_subject,
    detect_symptoms,
    required_observations,
)
from agroai.knowledge.documents import Chunk, KnowledgeDocument, RetrievedChunk
from agroai.knowledge.metadata import TrustLevel


def _retrieved(
    text: str,
    *,
    document_id: str = "d1",
    title: str = "Late blight",
    source: str = "FAO",
    trust: str = TrustLevel.TIER_1.value,
    category: str = "plant_diseases",
    score: float = 1.0,
) -> RetrievedChunk:
    document = KnowledgeDocument(
        id=document_id,
        title=title,
        content=text,
        source=source,
        category=category,
        metadata={"trust_level": trust},
    )
    chunk = Chunk(id=f"{document_id}:0", document_id=document_id, chunk_index=0, content=text)
    return RetrievedChunk(chunk=chunk, score=score, document=document)


class StubRetriever:
    name = "stub"

    def __init__(self, results: list[RetrievedChunk], *, raises: bool = False) -> None:
        self._results = results
        self._raises = raises
        self.calls: list[tuple[str, int]] = []

    def search(self, query: str, top_k: int = 5, categories=None):
        self.calls.append((query, top_k))
        if self._raises:
            raise RuntimeError("index unavailable")
        return self._results[:top_k]

    def health_check(self):
        return None


class LegacyRetriever:
    """A retriever that predates category scoping."""

    name = "legacy"
    calls = 0

    def __init__(self, results: list[RetrievedChunk]) -> None:
        self._results = results

    def search(self, query: str, top_k: int = 5):
        LegacyRetriever.calls += 1
        return self._results[:top_k]

    def health_check(self):
        return None


# ---------------------------------------------------------------------- #
# Symptom / subject detection
# ---------------------------------------------------------------------- #
def test_detect_symptoms_multilingual() -> None:
    assert detect_symptoms("pomidor bargida sariq dog'lar bor") == ["leaf_spots", "yellowing"]
    assert "yellowing" in detect_symptoms("Листья желтеют")
    assert "wilting" in detect_symptoms("the plant is wilting badly")
    assert detect_symptoms("ob-havo qanday?") == []


def test_detect_subject_prefers_first_mention() -> None:
    assert detect_subject("pomidor bargida dog'lar") == "tomato"
    assert detect_subject("sigir kasal") == "cattle"
    assert detect_subject("umumiy savol") == ""


def test_required_observations_adds_veterinary_and_symptom_items() -> None:
    items = required_observations(["leaf_spots"], "tomato")
    assert any("orqa tomoni" in item for item in items)
    vet = required_observations([], "cattle")
    assert any("vazni" in item for item in vet)
    # Generic evidence requests are always present.
    assert len(items) >= 4


def test_confidence_band_labels() -> None:
    assert confidence_band(0.9) == "yuqori / well-supported"
    assert confidence_band(0.6) == "o'rta / moderate"
    assert confidence_band(0.1) == BAND_LOW


# ---------------------------------------------------------------------- #
# Engine
# ---------------------------------------------------------------------- #
def test_engine_refuses_without_evidence() -> None:
    engine = RetrievalDifferentialEngine(StubRetriever([]))
    result = engine.differentiate("pomidor bargida sariq dog'lar")
    assert result.refused is True
    assert result.hypotheses == []
    assert result.symptoms == ["leaf_spots", "yellowing"]
    assert "Bilim bazasida" in result.reason
    assert result.required_observations


def test_engine_ranks_documents_with_evidence() -> None:
    hits = [
        _retrieved("sariq dog'lar va nekroz erta blight belgisi", document_id="d1", title="Early blight"),
        _retrieved("magniy yetishmovchiligida barg sariq bo'ladi", document_id="d2", title="Mg deficiency", source="Blog", trust=TrustLevel.TIER_3.value, score=0.5),
    ]
    engine = RetrievalDifferentialEngine(StubRetriever(hits))
    result = engine.differentiate("pomidor bargida sariq dog'lar va nekroz")

    assert result.refused is False
    assert [item.cause for item in result.hypotheses] == ["Early blight", "Mg deficiency"]
    best = result.best
    assert best is not None
    # A tier-1 source outranks a tier-3 blog on the same evidence.
    assert best.confidence > result.hypotheses[1].confidence
    assert best.trust_level == TrustLevel.TIER_1.value
    assert best.evidence and "FAO" in best.evidence[0]
    assert best.excerpt
    assert best.matched_symptoms


def test_engine_limits_and_dedupes_hypotheses() -> None:
    hits = [
        _retrieved("nekroz bor", document_id="d1", title="Same"),
        _retrieved("nekroz kuchli", document_id="d1", title="Same"),
    ]
    engine = RetrievalDifferentialEngine(StubRetriever(hits), max_hypotheses=3)
    result = engine.differentiate("barg nekroz")
    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].cause == "Same"


def test_engine_survives_a_broken_retriever() -> None:
    engine = RetrievalDifferentialEngine(StubRetriever([], raises=True))
    result = engine.differentiate("pomidor kasalligi")
    assert result.refused is True


def test_engine_works_with_a_retriever_without_category_scoping() -> None:
    engine = RetrievalDifferentialEngine(LegacyRetriever([_retrieved("nekroz")]))
    result = engine.differentiate("nekroz")
    assert not result.refused
    assert LegacyRetriever.calls == 1


# ---------------------------------------------------------------------- #
# Rendering
# ---------------------------------------------------------------------- #
def test_render_states_it_is_not_a_diagnosis() -> None:
    engine = RetrievalDifferentialEngine(StubRetriever([_retrieved("nekroz va dog'lar")]))
    rendered = engine.differentiate("barg nekroz").render()
    assert "tashxis emas" in rendered
    assert "Aniqlikni oshirish uchun" in rendered


def test_engine_reports_empty_documents_are_not_shown() -> None:
    """A chunk without a resolvable document still yields an id-based cause."""
    document = KnowledgeDocument(id="d9", title="", content="nekroz")
    chunk = Chunk(id="d9:0", document_id="d9", chunk_index=0, content="nekroz")
    engine = RetrievalDifferentialEngine(StubRetriever([RetrievedChunk(chunk=chunk, score=1.0, document=document)]))
    result = engine.differentiate("nekroz")
    assert result.hypotheses[0].cause == "d9"
    # The document declares the default source, so the hypothesis keeps it.
    assert result.hypotheses[0].source == "local"


def test_to_dict_is_json_friendly() -> None:
    import json

    engine = RetrievalDifferentialEngine(StubRetriever([_retrieved("nekroz dog'lar")]))
    payload = engine.differentiate("nekroz").to_dict()
    json.dumps(payload)
    assert payload["refused"] is False
    assert payload["confidence_band"]


def test_empty_question_refuses_cleanly() -> None:
    engine = RetrievalDifferentialEngine(StubRetriever([]))
    result = engine.differentiate("")
    assert result.refused is True
    assert result.symptoms == []
    assert isinstance(result, DifferentialDiagnosis)


def test_missing_document_metadata_does_not_break_ranking() -> None:
    chunk = Chunk(id="x:0", document_id="x", chunk_index=0, content="nekroz dog'lar")
    engine = RetrievalDifferentialEngine(
        StubRetriever([RetrievedChunk(chunk=chunk, score=1.0, document=None)])
    )
    result = engine.differentiate("dog'lar")
    assert result.hypotheses[0].trust_level == TrustLevel.UNKNOWN.value
    assert result.hypotheses[0].confidence < 0.75
