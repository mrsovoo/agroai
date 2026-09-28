"""Differential diagnosis: ranked possibilities, never a verdict.

A single-cause classifier answers "the disease is X". That is the wrong shape for
this domain: the same yellowing tomato leaf can be a nutrient deficiency,
irrigation stress, salinity or an early infection, and the honest answer is a
*ranked list of possibilities with the evidence behind each one*.

This module holds the data contract:

* :func:`detect_symptoms` - what the user actually described, extracted with a
  declared multilingual lexicon. It never invents an observation.
* :class:`Hypothesis` - one possible cause, with the sources that support it.
* :class:`DifferentialDiagnosis` - the full result, including the observations
  that would discriminate between the possibilities.

The confidence numbers are explicitly **ranking aids**, not probabilities. The
engine is in :mod:`agroai.diagnosis.engine`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

#: Confidence bands, in words, so no band is mistaken for a probability.
BAND_HIGH = "yuqori / well-supported"
BAND_MEDIUM = "o'rta / moderate"
BAND_LOW = "past / weak"
BAND_NONE = "dalilsiz / no evidence"

HIGH_THRESHOLD = 0.75
MEDIUM_THRESHOLD = 0.50


# ---------------------------------------------------------------------- #
# Symptom detection
# ---------------------------------------------------------------------- #
#: Declared, multilingual (uz / ru / en) symptom lexicon. Each key is a symptom
#: group; each value lists the words that *mean the user said it*. This is a
#: language aid, not a knowledge claim - a match records the user's words.
SYMPTOM_LEXICON: dict[str, tuple[str, ...]] = {
    "yellowing": (
        "sarg'ay",
        "sargay",
        "sariq",
        "yellow",
        "желт",
        "хлороз",
        "chlorosis",
        "xloroz",
    ),
    "leaf_spots": (
        "dog'",
        "dog'lar",
        "spot",
        "пятн",
        "lesion",
        "yarasi",
        "yaralar",
    ),
    "necrosis": (
        "nekroz",
        "qurib",
        "qurigan",
        "qurish",
        "necrosis",
        "некроз",
        "kuyish",
        "kuygan",
        "dry",
    ),
    "wilting": (
        "so'li",
        "so'lib",
        "soli",
        "wilting",
        "wilted",
        "увяд",
        "вянет",
        "sust",
    ),
    "leaf_curl": ("burma", "buralib", "bural", "curl", "скруч", "deformatsiya"),
    "mold": ("mog'", "mogor", "mold", "mildew", "плесен", "zamburug'", "fungal"),
    "pest_damage": (
        "hasharot",
        "qurt",
        "zararkunanda",
        "pest",
        "insect",
        "насеком",
        "shira",
        "aphid",
        "kanal",
        "tuxum qo'yg",
    ),
    "root_damage": ("ildiz", "root", "корн", "chirigan", "chirish", "rot", "gnil"),
    "stunted_growth": ("o'smay", "osmay", "sekin o's", "stunted", "карлик", "pakana"),
    "fruit_damage": ("meva", "fruit", "плод", "hosil buzil"),
    "discoloration": (
        "rangsiz",
        "rang o'zgar",
        "discolor",
        "побур",
        "jigarrang",
        "brown",
        "qoraytir",
    ),
    "dropping": ("to'kil", "tokil", "drop", "опад", "to'kilish"),
    "soil_salinity": ("sho'r", "shor", "salinity", "засолен", "tuzli", "tuz"),
    "nutrient_deficiency": (
        "yetishmovchilik",
        "yetmaydi",
        "deficiency",
        "недостат",
        "oziqa",
        "ozuqa",
    ),
}

#: Generic evidence requests: these make a differential sharper for any subject.
BASE_OBSERVATIONS: tuple[str, ...] = (
    "Butun o'simlik/hayvonning umumiy ko'rinishi (rasm yoki tavsif).",
    "Muammo qachon boshlangani va qanday tez tarqalayotgani.",
    "Hudud (viloyat/tuman) va o'sish sharoiti.",
)

#: Extra observations per detected symptom group - what would discriminate.
SYMPTOM_OBSERVATIONS: dict[str, tuple[str, ...]] = {
    "yellowing": (
        "Bargning orqa tomoni (zamburug' belgisi bor-yo'qligi uchun).",
        "Tuproq pH va EC, oxirgi o'g'itlash vaqti.",
    ),
    "leaf_spots": (
        "Dog'larning shakli, chegarasi va tarqalish tartibi (quyidan/yuqoridan).",
        "Bargning orqa tomoni.",
    ),
    "necrosis": ("Nekroz chegarasi (tomirlar bilan cheklanganmi?).",),
    "wilting": (
        "Sug'orish jadvali va tuproq namligi.",
        "Ildiz bo'yni holati.",
    ),
    "leaf_curl": ("Shira/zararkunanda bor-yo'qligi (barg ostini tekshirish).",),
    "mold": ("Namlik, harorat va shamollatish sharoiti.",),
    "pest_damage": (
        "Zararkunandaning yaqin rejali rasmi.",
        "Barg ostidagi tuxum yoki lichinkalar.",
    ),
    "root_damage": ("Ildiz tizimining holati (o'simlikni sug'urib tekshirish).",),
    "soil_salinity": (
        "Sug'orish suvi tahlili (EC/tuzlilik).",
        "Drenaj holati.",
    ),
    "nutrient_deficiency": (
        "Barg yoshi: alomatlar eski yoki yangi barglarda?",
        "Oxirgi o'g'itlash tarkibi va dozasi.",
    ),
}

#: Extra observations when the subject is an animal.
VETERINARY_OBSERVATIONS: tuple[str, ...] = (
    "Hayvon turi, yoshi va vazni.",
    "Tana harorati, ishtaha va suv ichishi.",
    "Chorvadagi boshqa hayvonlarda ham shu holat bormi.",
)

#: Subject lexicon (declared data, used only to label the differential).
SUBJECT_KEYWORDS: dict[str, tuple[str, ...]] = {
    "tomato": ("pomidor", "tomato", "помидор"),
    "cucumber": ("bodring", "cucumber", "огурец"),
    "potato": ("kartoshka", "potato", "картоф"),
    "cotton": ("paxta", "cotton", "хлопок"),
    "wheat": ("bug'doy", "bugdoy", "wheat", "пшениц"),
    "maize": ("makkajo'xori", "makka", "corn", "maize", "кукуруз"),
    "onion": ("piyoz", "onion", "лук"),
    "carrot": ("sabzi", "carrot", "морков"),
    "grape": ("uzum", "grape", "виноград"),
    "apple": ("olma", "apple", "яблок"),
    "cattle": ("sigir", "qoramol", "buqa", "cattle", "cow", "корова"),
    "sheep": ("qo'y", "qoy", "sheep", "овц"),
    "goat": ("echki", "goat", "коз"),
    "horse": ("ot ", "otim", "horse", "лошад"),
    "camel": ("tuya", "camel", "верблюд"),
    "poultry": ("tovuq", "jo'ja", "parranda", "poultry", "chicken", "куриц"),
    "rabbit": ("quyon", "rabbit", "кролик"),
}

VETERINARY_SUBJECTS = frozenset(
    {"cattle", "sheep", "goat", "horse", "camel", "poultry", "rabbit"}
)

MAX_HYPOTHESES = 5


def detect_symptoms(text: str) -> list[str]:
    """Return the symptom groups the text actually mentions (sorted)."""
    lowered = (text or "").lower()
    found = [
        name
        for name, keywords in SYMPTOM_LEXICON.items()
        if any(keyword in lowered for keyword in keywords)
    ]
    return sorted(found)


def detect_subject(text: str) -> str:
    """Return the crop/animal the text is about, or an empty string."""
    lowered = " " + (text or "").lower() + " "
    matches: list[tuple[int, str]] = []
    for name, keywords in SUBJECT_KEYWORDS.items():
        for keyword in keywords:
            position = lowered.find(keyword)
            if position >= 0:
                matches.append((position, name))
                break
    if not matches:
        return ""
    matches.sort()
    return matches[0][1]


@dataclass
class Hypothesis:
    """One possible cause, with the evidence that supports it."""

    cause: str
    category: str = ""
    confidence: float = 0.0
    document_id: str = ""
    trust_level: str = ""
    source: str = ""
    matched_symptoms: list[str] = field(default_factory=list)
    evidence: list[str] = field(default_factory=list)
    excerpt: str = ""

    @property
    def band(self) -> str:
        return confidence_band(self.confidence)

    def to_dict(self) -> dict[str, Any]:
        return {
            "cause": self.cause,
            "category": self.category,
            "confidence": round(self.confidence, 3),
            "band": self.band,
            "document_id": self.document_id,
            "trust_level": self.trust_level,
            "source": self.source,
            "matched_symptoms": list(self.matched_symptoms),
            "evidence": list(self.evidence),
            "excerpt": self.excerpt,
        }


@dataclass
class DifferentialDiagnosis:
    """A ranked set of possibilities - explicitly not a diagnosis."""

    question: str
    subject: str = ""
    symptoms: list[str] = field(default_factory=list)
    hypotheses: list[Hypothesis] = field(default_factory=list)
    required_observations: list[str] = field(default_factory=list)
    refused: bool = False
    reason: str = ""
    notes: list[str] = field(default_factory=list)

    @property
    def best(self) -> Hypothesis | None:
        return self.hypotheses[0] if self.hypotheses else None

    @property
    def confidence_band(self) -> str:
        if not self.hypotheses:
            return BAND_NONE
        return self.hypotheses[0].band

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "subject": self.subject,
            "symptoms": list(self.symptoms),
            "hypotheses": [item.to_dict() for item in self.hypotheses],
            "required_observations": list(self.required_observations),
            "refused": self.refused,
            "reason": self.reason,
            "confidence_band": self.confidence_band,
            "notes": list(self.notes),
        }

    def render(self, language: str = "uz") -> str:
        """Render a farmer-readable differential (Uzbek by default)."""
        lines: list[str] = []
        if self.symptoms:
            labels = ", ".join(self.symptoms)
            lines.append(f"Kuzatilgan/kiritilgan belgilar: {labels}")
        if self.subject:
            lines.append(f"Ekin/hayvon: {self.subject}")
        if self.refused:
            lines.append("")
            lines.append(f"Aniq xulosa qilish uchun dalil yetarli emas: {self.reason}")
        else:
            lines.append("")
            lines.append("Ehtimoliy sabablar (taxmin, yakuniy tashxis emas):")
            for index, hypothesis in enumerate(self.hypotheses, start=1):
                lines.append(
                    f"  {index}. {hypothesis.cause} "
                    f"(ishonchlilik: {hypothesis.confidence:.0%}, {hypothesis.band})"
                )
                if hypothesis.source:
                    lines.append(f"     manba: {hypothesis.source}")
            lines.append("")
            lines.append(
                "Bu ro'yxat ehtimolliklar reytingi, tasdiqlangan tashxis emas. "
                "Laboratoriya/hujjatli tasdiq o'rnini bosmaydi."
            )
        if self.required_observations:
            lines.append("")
            lines.append("Aniqlikni oshirish uchun yana nima kerak:")
            for item in self.required_observations:
                lines.append(f"  - {item}")
        for note in self.notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)


def confidence_band(confidence: float) -> str:
    """Map a ranking score to an explicit, non-probabilistic band."""
    if confidence >= HIGH_THRESHOLD:
        return BAND_HIGH
    if confidence >= MEDIUM_THRESHOLD:
        return BAND_MEDIUM
    return BAND_LOW


def required_observations(
    symptoms: Iterable[str], subject: str = ""
) -> list[str]:
    """Build the evidence checklist for a differential (order preserved)."""
    found: list[str] = []
    for item in BASE_OBSERVATIONS:
        found.append(item)
    subject_key = (subject or "").lower()
    if subject_key in VETERINARY_SUBJECTS:
        for item in VETERINARY_OBSERVATIONS:
            found.append(item)
    for symptom in symptoms:
        for item in SYMPTOM_OBSERVATIONS.get(symptom, ()):
            # Local files and regions may repeat; keep the list short and useful.
            found.append(item)
    return found


__all__ = [
    "BAND_HIGH",
    "BAND_LOW",
    "BAND_MEDIUM",
    "BAND_NONE",
    "BASE_OBSERVATIONS",
    "DifferentialDiagnosis",
    "HIGH_THRESHOLD",
    "Hypothesis",
    "MAX_HYPOTHESES",
    "MEDIUM_THRESHOLD",
    "SUBJECT_KEYWORDS",
    "SYMPTOM_LEXICON",
    "SYMPTOM_OBSERVATIONS",
    "VETERINARY_OBSERVATIONS",
    "VETERINARY_SUBJECTS",
    "confidence_band",
    "detect_subject",
    "detect_symptoms",
    "required_observations",
]
