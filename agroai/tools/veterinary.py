"""Veterinary tool: structure the case, never diagnose it.

The roadmap's veterinary workflow is *symptoms -> differential diagnosis ->
knowledge retrieval -> possible conditions -> required additional observations ->
risk level -> recommendation*, with safety as the highest priority.

AgroAI can honestly do the parts that do not require a licensed diagnosis:

* turn the reported symptoms into an organised observation set,
* say which **additional observations** would be needed,
* flag urgency signals,
* refuse to supply drug names or dosages from memory.

Listed possible conditions come only from retrieved knowledge-base sources - so
this tool does not produce a diagnosis list at all. It says so, and points to the
knowledge base and to a veterinarian.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from agroai.tools.base import (
    CHOICE,
    TEXT,
    Tool,
    ToolParameter,
    ToolResult,
    ToolSpec,
)

REFERENCE_OBSERVATION = "Structured case-history / observation checklist (clinical practice)"

NO_DIAGNOSIS_NOTE = (
    "Bu vosita tashxis qo'ymaydi. Tashxis va davolash litsenziyali "
    "veterinariya mutaxassisi ishi. / This tool does not diagnose. Diagnosis and "
    "treatment belong to a licensed veterinarian."
)

DOSE_REFUSAL = (
    "Dori nomi va dozasini xotiradan aytish mumkin emas: doza turni, vaznni, "
    "yoshni, holatni va mahalliy ruxsat etilgan preparatni hisobga oladi va "
    "ishonchli manbadan (preparat yorlig'i, veterinariya ma'lumoti) olinishi "
    "shart. Kasallik bo'yicha bilim bazasidan manba topilsa, u ko'rsatiladi. "
    "/ Drug names and dosages cannot be recalled from memory: they must come "
    "from a verified source (product label, veterinary literature) and depend on "
    "species, weight, age and condition."
)

#: Symptom keywords (uz/ru/en) mapped to the body system they suggest.
SYSTEM_KEYWORDS: dict[str, tuple[str, ...]] = {
    "respiratory": (
        "cough", "yotal", "yutal", "nasal", "burun", "discharge", "snot", "suv",
        "pneumonia", "nafas", "дышать", "кашел",
    ),
    "digestive": (
        "diarrhoea", "diarrhea", "ich ket", "ischal", "qorin", "vomit", "qusish",
        "rumen", "appetite", "ishtaha", "ovqat yemaydi", "no appetite",
    ),
    "fever": ("fever", "harorat", "isitma", "temperature", "issiq", "жар"),
    "locomotion": (
        "lame", "cho'loq", "cholok", "yurmaydi", "cannot stand", "turolmaydi",
        "hoof", "tuyoq", "joint",
    ),
    "skin": ("skin", "teri", "rash", "toshma", "wound", "yara", "hair loss", "tuk"),
    "udder_milk": ("udder", "yelin", "milk", "sut", "mastitis"),
    "reproduction": ("abortion", "tashlash", "buzoq", "retained", "placenta", "birth"),
    "nervous": ("convulsion", "tut", "tremor", "qaltira", "paralysis", "falaj"),
    "urinary": ("urine", "siy", "kidney", "buyrak", "bloody urine"),
}

#: Signals that the case needs a veterinarian now, not a checklist.
URGENCY_KEYWORDS: tuple[str, ...] = (
    "blood", "qon", "sudden death", "o'lim", "ola", "cannot stand",
    "turolmaydi", "convulsion", "tutqanoq", "abortion", "tashlash",
    "difficulty breathing", "bo'g'il", "nafas olmaydi",
)

#: Additional observations to request, per detected system.
OBSERVATION_CHECKLIST: dict[str, tuple[str, ...]] = {
    "respiratory": (
        "Nafas soni 1 daqiqada va nafas turi (qiyin/tez).",
        "Yo'tal turi (quruq/ho'l) va qachon paydo bo'lishi.",
        "Burun va ko'z oqishi rangi.",
    ),
    "digestive": (
        "Suv va ozuqa iste'moli (oxirgi 24 soat).",
        "Najas ko'rinishi, rangi, suvli yoki qonli ekani.",
        "Qorin devori bosilganda og'riq bormi (koleyka belgilari).",
    ),
    "fever": (
        "Tana harorati (rektal, raqam bilan).",
        "Harorat qachondan beri ko'tarilgan.",
        "Yaqinda o'tkazilgan emlashlar va yangi hayvonlar.",
    ),
    "locomotion": (
        "Qaysi oyoq, tuyoq harorati va shish bormi.",
        "Yotib qolish, turishga harakat qilishi.",
        "Yaqinda surilish/travma bo'lganmi.",
    ),
    "skin": (
        "Teri o'zgarishi tarqalishi (mahalliy/umumiy) va qichishish.",
        "Zararkunanda (kanalar, pashsha) belgilari.",
        "Yaqinda qo'llanilgan dorilar.",
    ),
    "udder_milk": (
        "Yelin harorati, shish, og'riq (simmetrik/asimmetrik).",
        "Sut ko'rinishi (bo'laklar, rang, qon).",
        "Sog'ish tartibi va mahsuldorlik o'zgarishi.",
    ),
    "reproduction": (
        "Homiladorlik muddati va oxirgi tug'ruq.",
        "Oqindi bo'lganmi.",
        "Yaqin atrofdagi hayvonlarda o'xshash holat.",
    ),
    "nervous": (
        "Xatti-harakat o'zgarishi (agressiya, letargiya).",
        "Harorat va emlash holati.",
        "Zaharlanish manbai (o'simlik, kimyoviy) bo'lganmi.",
    ),
    "urinary": (
        "Siydik chiqarish tezligi va og'riq.",
        "Siydik rangi.",
        "Suv iste'moli.",
    ),
}

ALWAYS_OBSERVE: tuple[str, ...] = (
    "Tana harorati (raqam bilan).",
    "Yosh, jins va vazn (iloji bo'lsa o'lchov).",
    "Oxirgi emlashlar, parvarish va boqish sharoiti.",
    "Kasallik qachon boshlangan va qanday kechayapti (yaxshilanmoqda/yomonlashmoqda).",
)


@dataclass(frozen=True)
class CaseStructure:
    """The organised observation set for one reported case."""

    species: str
    systems: tuple[str, ...]
    observations: tuple[str, ...]
    urgent: bool
    matched_signals: tuple[str, ...]


def structure_case(species: str, symptoms: str) -> CaseStructure:
    """Organise reported symptoms into systems and required observations."""
    lowered = (symptoms or "").lower()
    signals: list[str] = []
    systems: list[str] = []
    for system, keywords in SYSTEM_KEYWORDS.items():
        hits = [keyword for keyword in keywords if keyword in lowered]
        if hits:
            systems.append(system)
            signals.extend(hits)

    observations: list[str] = list(ALWAYS_OBSERVE)
    for system in systems:
        observations.extend(OBSERVATION_CHECKLIST[system])
    if not systems:
        observations.append(
            "Aniq belgilar guruhi aniqlanmadi: belgilarni batafsil yozib bering "
            "(qachon boshlandi, qanday o'zgaryapti)."
        )

    urgent = any(keyword in lowered for keyword in URGENCY_KEYWORDS)
    return CaseStructure(
        species=species,
        systems=tuple(systems),
        observations=tuple(observations),
        urgent=urgent,
        matched_signals=tuple(dict.fromkeys(signals)),
    )


class VeterinaryTool(Tool):
    """Structure a veterinary case; refuse to diagnose or dose from memory."""

    spec = ToolSpec(
        name="veterinary_tool",
        description=(
            "Veterinary case helper: action='observation_checklist' (species, "
            "symptoms) organises the reported signs and lists the additional "
            "observations a veterinarian would need. It does not diagnose. "
            "action='dosage_lookup' is unavailable without a verified source."
        ),
        category="veterinary",
        parameters=(
            ToolParameter(
                "action",
                CHOICE,
                "One of: observation_checklist, dosage_lookup.",
                choices=("observation_checklist", "dosage_lookup"),
            ),
            ToolParameter(
                "species",
                CHOICE,
                "Species: cattle, sheep, goat, horse, camel, poultry, other.",
                required=False,
                choices=("cattle", "sheep", "goat", "horse", "camel", "poultry", "other"),
            ),
            ToolParameter("symptoms", TEXT, "Reported signs, free text.", required=False),
            ToolParameter(
                "question",
                TEXT,
                "The dosage question - for action='dosage_lookup'.",
                required=False,
            ),
        ),
        reference=REFERENCE_OBSERVATION,
        requires_verified_sources=True,
    )

    def run(self, **kwargs: Any) -> ToolResult:
        action = str(kwargs.get("action", "")).strip().lower()
        if action == "observation_checklist":
            return self._observations(kwargs)
        if action == "dosage_lookup":
            return self._dosage(kwargs)
        return self.fail(
            f"unknown action {action!r}; use observation_checklist or dosage_lookup"
        )

    # ------------------------------------------------------------------ #
    def _observations(self, kwargs: dict[str, Any]) -> ToolResult:
        species = str(kwargs.get("species", "")).strip().lower()
        symptoms = str(kwargs.get("symptoms", "")).strip()
        if not species:
            return self.fail("species is required for action='observation_checklist'")
        if not symptoms:
            return self.fail("symptoms is required for action='observation_checklist'")

        case = structure_case(species, symptoms)
        lines = [
            f"Tur/species: {case.species}",
            "Aniqlangan tizimlar/systems: "
            + (", ".join(case.systems) if case.systems else "(aniqlanmadi / none)"),
            "Qo'shimcha kuzatishlar kerak / additional observations required:",
        ]
        lines.extend(f"  - {item}" for item in case.observations)
        if case.urgent:
            lines.append(
                "SHOSHILINCH: veterinarga zudlik bilan murojaat qiling. / URGENT: "
                "contact a veterinarian immediately."
            )

        notes = [NO_DIAGNOSIS_NOTE]
        notes.append(
            "Ehtimoliy sabablar faqat bilimlar bazasidagi manbalar bilan "
            "ko'rsatiladi. / Possible causes are only listed from retrieved "
            "knowledge-base sources."
        )
        if case.urgent:
            notes.append(
                "Belgilangan signallar hayvon hayoti uchun xavfli bo'lishi mumkin. "
                "/ The reported signals may be life-threatening."
            )
        return ToolResult(
            tool=self.name,
            ok=True,
            output="\n".join(lines),
            data={
                "species": case.species,
                "systems": list(case.systems),
                "observations": list(case.observations),
                "urgent": case.urgent,
                "matched_signals": list(case.matched_signals),
            },
            reference=REFERENCE_OBSERVATION,
            notes=notes,
            requires_verified_sources=True,
        )

    def _dosage(self, kwargs: dict[str, Any]) -> ToolResult:
        question = str(kwargs.get("question", "")).strip()
        return self.fail(
            DOSE_REFUSAL,
            data={"action": "dosage_lookup", "question": question, "available": False},
        )


__all__ = [
    "ALWAYS_OBSERVE",
    "DOSE_REFUSAL",
    "NO_DIAGNOSIS_NOTE",
    "OBSERVATION_CHECKLIST",
    "REFERENCE_OBSERVATION",
    "SYSTEM_KEYWORDS",
    "URGENCY_KEYWORDS",
    "CaseStructure",
    "VeterinaryTool",
    "structure_case",
]
