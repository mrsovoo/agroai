"""Guardrails: decide *before* answering how careful the answer must be.

The roadmap's rule is blunt: AgroAI must not invent doses, treatments or
pesticide advice, and must say "no verified information found" when the
knowledge base cannot support an answer. That decision has to be made per
question, deterministically, before any model speaks.

This module classifies a question into a risk category using multilingual
keyword patterns (Uzbek, Russian, English), and returns:

* the risk level (low / medium / high),
* whether the answer **must** be grounded in retrieved sources,
* whether professional verification must be recommended,
* the disclaimers to attach,
* the extra rules to add to the system prompt.

It is intentionally rule-based: it runs offline, is auditable, and cannot be
talked out of a warning by the content of the question.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum

LOW = "low"
MEDIUM = "medium"
HIGH = "high"

RISK_ORDER = {LOW: 0, MEDIUM: 1, HIGH: 2}


class SafetyCategory(str, Enum):
    """What kind of hazard a question touches."""

    GENERAL = "general"
    PLANT_PROTECTION = "plant_protection"
    AGROCHEMICAL = "agrochemical"
    DOSAGE = "dosage"
    VETERINARY_MEDICINE = "veterinary_medicine"
    FOOD_SAFETY = "food_safety"


#: Keyword patterns per category (lowercased matching, uz/ru/en mixed).
CATEGORY_PATTERNS: dict[SafetyCategory, tuple[str, ...]] = {
    # Russian entries are stems on purpose ("доз" covers доза/дозе/дозу/дозы),
    # because a farmer writes the inflected form, not the dictionary form.
    SafetyCategory.DOSAGE: (
        "doza", "dozasi", "dose", "dosage", "mg", "ml/kg", "milligram",
        "gramm", "konsentratsiya", "концентрация", "доз", "дозиров", "normasi",
        "qancha qo'shiladi", "qancha beriladi", "how much should i apply",
        "сколько", "какая доза",
    ),
    SafetyCategory.AGROCHEMICAL: (
        "pestitsid", "pestisid", "fungitsid", "gerbitsid", "insektitsid",
        "pesticide", "fungicide", "herbicide", "insecticide", "acaricide",
        "zahar", "purkash", "spray", "o'g'it", "og'it", "fertilizer",
        "urea", "karbamid", "nitrat", "ammofos", "npk",
        "пестицид", "фунгицид", "гербицид", "инсектицид", "удобрен",
        "ядохимикат", "опрыскив",
    ),
    SafetyCategory.VETERINARY_MEDICINE: (
        "antibiotik", "antibiotic", "vaksina", "vaccine", "emlash", "vaksinatsiya",
        "veterinar", "veterinary", "dori", "preparat", "tabletka", "in'ektsiya",
        "injection", "og'izdan berish", "tratament", "davolash", "лечение",
        "антибиотик", "вакцин", "лекарств", "препарат", "таблетк", "инъекц",
        "укол", "ветеринар", "лечени", "колоть",
    ),
    SafetyCategory.FOOD_SAFETY: (
        "karantin", "withholding", "qoldiq", "residue", "sut", "milk", "go'sht",
        "meat", "tuxum", "egg", "iste'mol", "consumption", "zaharlanish",
        "poisoning", "mrl",
        "остат", "молок", "мясо", "яйц", "отравл",
    ),
    SafetyCategory.PLANT_PROTECTION: (
        "kasallik", "kasalligi", "zararkunanda", "shira", "qurt", "disease",
        "pest", "infection", "zamburug", "fungus", "virus", "nematoda",
        "barg", "leaf", "sariq dog", "so'lishi", "wilt", "chirish", "rot",
        "болезн", "вредител", "насеком", "гриб", "вирус", "нематод", "лист",
        "увядан", "гнил",
    ),
}

#: Categories that always require grounded sources and a professional check.
_ALWAYS_GROUNDED: frozenset[SafetyCategory] = frozenset(
    {
        SafetyCategory.DOSAGE,
        SafetyCategory.AGROCHEMICAL,
        SafetyCategory.VETERINARY_MEDICINE,
        SafetyCategory.FOOD_SAFETY,
    }
)

_NUMERIC_UNIT = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|ml|g|kg|%|ppm|l|gramm|milligram|litr|doza)\b",
    re.IGNORECASE,
)

DISCLAIMERS: dict[SafetyCategory, str] = {
    SafetyCategory.DOSAGE: (
        "Dozalar va konsentratsiyalar faqat tasdiqlangan manbadan olinishi "
        "kerak. Preparat yorlig'i va mahalliy ruxsatnomani tekshiring. "
        "/ Dosages must come from a verified source; check the product label "
        "and local registration."
    ),
    SafetyCategory.AGROCHEMICAL: (
        "Pestitsid/fungitsid/gerbitsid qo'llashda yorliqdagi doza, kutish "
        "muddati, himoya vositalari va suv/tuproq sharoiti ustuvor; agrokimyo "
        "mutaxassisi bilan tekshiring. / For pesticides, the label's rate, "
        "pre-harvest interval, protective equipment and local conditions "
        "outrank any general advice."
    ),
    SafetyCategory.VETERINARY_MEDICINE: (
        "Dori tanlash va davolash veterinariya mutaxassisi ishi; antibiotik va "
        "emlash bo'yicha qaror hayvon holati va mahalliy qoidalarga bog'liq. "
        "/ Drug choice and treatment are a veterinarian's call; antibiotic and "
        "vaccination decisions depend on the animal and local rules."
    ),
    SafetyCategory.FOOD_SAFETY: (
        "Oziq-ovqat xavfsizligi uchun kutish muddati (withholding period) va "
        "qoldiq me'yorlariga rioya qiling. / Observe withholding periods and "
        "residue limits for food-producing animals and crops."
    ),
    SafetyCategory.PLANT_PROTECTION: (
        "Tashxis ehtimoliy: belgilar bir necha sababdan bo'lishi mumkin, "
        "manbalar ko'rsatiladi va mutaxassis tasdig'i tavsiya etiladi. "
        "/ The assessment is probabilistic: symptoms can have several causes; "
        "sources are shown and expert confirmation is recommended."
    ),
}

PROMPT_RULES: dict[SafetyCategory, str] = {
    SafetyCategory.DOSAGE: (
        "7. Bu savolda doza/konsentratsiya bor. Faqat KONTEKSTDAGI aniq "
        "raqamlarni keltiring; kontekstda bo'lmasa, raqam aytmang va "
        "yorliq/mutaxassisga murojaat qilishni ayting."
    ),
    SafetyCategory.AGROCHEMICAL: (
        "7. Bu savolda agrokimyoviy vosita bor. Faqat KONTEKSTDAGI doza va "
        "kutish muddatlarini keltiring; yo'q bo'lsa aytmang. Yorliq ustuvor."
    ),
    SafetyCategory.VETERINARY_MEDICINE: (
        "7. Bu savol veterinariya davolashiga tegishli. Tashxis qo'ymang va "
        "dori/doza o'ylab topmang; ehtimoliy sabablarni manba bilan ko'rsatib, "
        "veterinarga murojaat qilishni tavsiya eting."
    ),
    SafetyCategory.FOOD_SAFETY: (
        "7. Bu savol oziq-ovqat xavfsizligiga tegishli. Kutish muddati va "
        "qoldiq me'yorlarini faqat manbadan keltiring."
    ),
    SafetyCategory.PLANT_PROTECTION: (
        "7. Bu savol o'simlik himoyasiga tegishli. Belgilarni ehtimoliy "
        "sabablar sifatida ajratib ko'rsating va aniqlik darajasini ayting."
    ),
}


@dataclass
class SafetyVerdict:
    """How carefully a particular question must be answered."""

    category: SafetyCategory = SafetyCategory.GENERAL
    risk_level: str = LOW
    requires_sources: bool = False
    requires_professional_verification: bool = False
    matched_terms: list[str] = field(default_factory=list)
    numeric_terms: list[str] = field(default_factory=list)
    disclaimers: list[str] = field(default_factory=list)

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == HIGH

    @property
    def disclaimers_text(self) -> str:
        return "\n".join(self.disclaimers)

    def to_dict(self) -> dict[str, object]:
        return {
            "category": self.category.value,
            "risk_level": self.risk_level,
            "requires_sources": self.requires_sources,
            "requires_professional_verification": self.requires_professional_verification,
            "matched_terms": list(self.matched_terms),
            "numeric_terms": list(self.numeric_terms),
        }


class SafetyGuard:
    """Classify questions and produce the safety envelope for an answer."""

    def __init__(self, *, strict: bool = True) -> None:
        self._strict = strict

    @property
    def strict(self) -> bool:
        return self._strict

    def classify(self, question: str) -> SafetyVerdict:
        """Classify one question. Unknown/empty input is general and low risk."""
        text = (question or "").lower()
        verdict = SafetyVerdict()
        if not text.strip():
            return verdict

        matched: list[str] = []
        categories: list[SafetyCategory] = []
        for category, patterns in CATEGORY_PATTERNS.items():
            hits = [pattern for pattern in patterns if pattern in text]
            if hits:
                categories.append(category)
                matched.extend(hits)

        if not categories:
            verdict.matched_terms = matched
            return verdict

        # The most specific category wins: dosage > veterinary > food safety >
        # agrochemical > plant protection.
        priority = (
            SafetyCategory.DOSAGE,
            SafetyCategory.VETERINARY_MEDICINE,
            SafetyCategory.FOOD_SAFETY,
            SafetyCategory.AGROCHEMICAL,
            SafetyCategory.PLANT_PROTECTION,
        )
        category = next((item for item in priority if item in categories), categories[0])

        numeric = _NUMERIC_UNIT.findall(text)
        grounded = category in _ALWAYS_GROUNDED

        verdict.category = category
        verdict.matched_terms = list(dict.fromkeys(matched))
        verdict.numeric_terms = numeric
        verdict.requires_sources = grounded or bool(numeric)
        verdict.requires_professional_verification = grounded
        verdict.disclaimers = [DISCLAIMERS[category]] if category in DISCLAIMERS else []

        if category in {SafetyCategory.VETERINARY_MEDICINE, SafetyCategory.DOSAGE, SafetyCategory.FOOD_SAFETY}:
            verdict.risk_level = HIGH
        elif category is SafetyCategory.AGROCHEMICAL:
            verdict.risk_level = HIGH if self._strict else MEDIUM
        else:
            verdict.risk_level = MEDIUM

        if numeric and verdict.risk_level == MEDIUM:
            verdict.risk_level = HIGH

        return verdict

    def prompt_rules(self, verdict: SafetyVerdict) -> str:
        """Extra system-prompt rules for this verdict (empty for low risk)."""
        if verdict.risk_level == LOW:
            return ""
        return PROMPT_RULES.get(verdict.category, "")

    def disclaimers(self, verdict: SafetyVerdict) -> list[str]:
        """Disclaimers to show with the answer."""
        notes = list(verdict.disclaimers)
        if verdict.requires_sources:
            notes.append(
                "Ushbu javob faqat bilimlar bazasidagi manbalar bilan "
                "tasdiqlanadi. / This answer is only as good as its cited sources."
            )
        return notes


__all__ = [
    "CATEGORY_PATTERNS",
    "DISCLAIMERS",
    "HIGH",
    "LOW",
    "MEDIUM",
    "PROMPT_RULES",
    "RISK_ORDER",
    "SafetyCategory",
    "SafetyGuard",
    "SafetyVerdict",
]
