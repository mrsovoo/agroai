"""Tests for the safety layer: risk classification and answer validation."""

from __future__ import annotations

from agroai.knowledge.documents import Chunk, KnowledgeDocument, RetrievedChunk
from agroai.safety import (
    CRITICAL,
    HIGH,
    LOW,
    MEDIUM,
    AnswerValidator,
    SafetyCategory,
    SafetyGuard,
    grounding_coverage,
)


def _chunk(text: str, url: str = "", title: str = "Manba") -> RetrievedChunk:
    document = KnowledgeDocument(id="d1", title=title, content="c", source="FAO", source_url=url)
    chunk = Chunk(id="d1:0", document_id="d1", chunk_index=0, content=text)
    return RetrievedChunk(chunk=chunk, score=1.0, document=document)


# --------------------------------------------------------------------- #
# Guardrails
# --------------------------------------------------------------------- #
def test_guard_ignores_harmless_questions() -> None:
    guard = SafetyGuard()
    for question in ("", "   ", "Salom, qanday ishlaysiz?", "Bug'doy qachon ekiladi?"):
        verdict = guard.classify(question)
        assert verdict.risk_level == LOW
        assert verdict.requires_sources is False
        assert guard.prompt_rules(verdict) == ""


def test_guard_marks_plant_protection_questions() -> None:
    verdict = SafetyGuard().classify("Pomidor barglarida sariq dog'lar paydo bo'ldi")
    assert verdict.category is SafetyCategory.PLANT_PROTECTION
    assert verdict.risk_level == MEDIUM
    assert verdict.disclaimers
    assert "probabilistic" in verdict.disclaimers[0]


def test_guard_escalates_dosage_and_veterinary_questions() -> None:
    guard = SafetyGuard()

    dosage = guard.classify("Pestitsid dozasi qancha bo'ladi?")
    assert dosage.category is SafetyCategory.DOSAGE
    assert dosage.risk_level == HIGH
    assert dosage.requires_sources is True
    assert dosage.requires_professional_verification is True
    assert guard.prompt_rules(dosage)

    veterinary = guard.classify("Sigirga qanday antibiotik beriladi?")
    assert veterinary.category is SafetyCategory.VETERINARY_MEDICINE
    assert veterinary.risk_level == HIGH

    food = guard.classify("Sut uchun kutish muddati qancha?")
    assert food.category is SafetyCategory.FOOD_SAFETY
    assert food.risk_level == HIGH


def test_guard_treats_numbers_in_agrochemical_questions_as_high_risk() -> None:
    verdict = SafetyGuard().classify("Bir gektarga 50 kg o'g'it solsam bo'ladimi?")
    assert verdict.category is SafetyCategory.AGROCHEMICAL
    assert verdict.risk_level == HIGH
    assert verdict.numeric_terms
    assert verdict.requires_sources is True


def test_guard_strict_mode_can_relax_agrochemical_risk() -> None:
    question = "O'g'it tanlash haqida umumiy ma'lumot bering"
    assert SafetyGuard(strict=True).classify(question).risk_level == HIGH
    relaxed = SafetyGuard(strict=False).classify(question)
    assert relaxed.risk_level == MEDIUM
    assert relaxed.requires_sources is True


def test_guard_verdict_serializes() -> None:
    data = SafetyGuard().classify("fungitsid dozasi").to_dict()
    assert data["category"] == "dosage"
    assert data["risk_level"] == HIGH
    assert isinstance(data["matched_terms"], list)


def test_guard_disclaimers_mention_sources_when_required() -> None:
    guard = SafetyGuard()
    notes = guard.disclaimers(guard.classify("gerbitsid dozasi qancha?"))
    assert notes
    assert any("manbalar" in note or "sources" in note for note in notes)


# --------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------- #
def test_validation_flags_fabricated_urls() -> None:
    report = AnswerValidator().validate(
        "Batafsil: https://fake.example/guide", [_chunk("matn", url="https://fao.org/real")]
    )
    assert report.ok is False
    assert report.fabricated_urls == ["https://fake.example/guide"]
    assert report.severity == CRITICAL


def test_validation_accepts_urls_that_come_from_the_sources() -> None:
    report = AnswerValidator().validate(
        "Manba: https://fao.org/real", [_chunk("matn", url="https://fao.org/real")]
    )
    assert report.fabricated_urls == []


def test_validation_flags_unsupported_quantities() -> None:
    report = AnswerValidator().validate(
        "Har gektarga 2.5 kg fungitsid sepiladi.",
        [_chunk("Fungitsid haqida umumiy ma'lumot berilgan.")],
    )
    assert "2.5 kg" in report.unsupported_quantities
    assert report.ok is False


def test_validation_accepts_quantities_present_in_the_context() -> None:
    report = AnswerValidator().validate(
        "Har gektarga 2.5 kg fungitsid sepiladi.",
        [_chunk("Me'yor: 2.5 kg fungitsid har gektar uchun.")],
    )
    assert report.unsupported_quantities == []


def test_validation_reports_low_grounding_as_warning() -> None:
    report = AnswerValidator().validate(
        "Zebra yulduzlari haqida she'r yozdim.", [_chunk("Tuproq sho'rlanishi haqida matn")]
    )
    codes = {flag.code for flag in report.flags}
    assert "low_grounding" in codes
    assert report.severity != CRITICAL


def test_validation_requires_sources_for_high_risk_questions() -> None:
    guard = SafetyGuard()
    verdict = guard.classify("pestitsid dozasi qancha?")
    report = AnswerValidator().validate("Biror narsa.", [], verdict)
    codes = {flag.code for flag in report.flags}
    assert "missing_required_sources" in codes
    assert report.ok is False


def test_validation_flags_overconfidence() -> None:
    report = AnswerValidator().validate(
        "Bu 100% aniq tashxis.", [_chunk("Ehtimoliy sabablar ko'rsatilgan matn")],
        SafetyGuard().classify("pomidor kasalligi bormi?"),
    )
    codes = {flag.code for flag in report.flags}
    assert "overconfidence" in codes


def test_validation_report_renders_and_serializes() -> None:
    report = AnswerValidator().validate("javob", [])
    assert isinstance(report.render(), str)
    assert report.to_dict()["severity"] in {"info", "warning", "critical"}
    empty = AnswerValidator().validate("", [])
    assert "muammo topilmadi" in empty.render()


def test_grounding_coverage_is_bounded() -> None:
    assert grounding_coverage("tuproq sho'rlanishi", "tuproq sho'rlanishi") == 1.0
    assert grounding_coverage("butunlay boshqa matn", "") == 0.0
    assert 0.0 < grounding_coverage("tuproq sho'rlanishi", "tuproq haqida matn") < 1.0
