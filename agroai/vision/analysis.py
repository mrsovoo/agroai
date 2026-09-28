"""Media analysis: inspection + differential diagnosis, with the gaps named.

```text
photo/video -> inspect (real bytes) -> vision engine (may be unconfigured)
                                            |
                                     user's description
                                            v
                              differential diagnosis over OUR knowledge
                                            v
                     structured answer + limitations + what would sharpen it
```

The value of this pipeline today is precisely that it does **not** claim to see.
It gives the farmer:

* the objective facts of the file (format, resolution, duration),
* a ranked differential built from what they described plus the knowledge base,
* an explicit statement that visual analysis is not available,
* the exact observations that would make the answer better.

That is a useful, truthful answer - and it is the same shape a real vision model
will fill in later, so no caller needs rewriting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from agroai.diagnosis.base import DifferentialDiagnosis
from agroai.diagnosis.engine import DifferentialEngine, RetrievalDifferentialEngine
from agroai.retrieval.retriever import Retriever
from agroai.vision.base import (
    VISION_ERROR,
    VISION_NOT_CONFIGURED,
    MediaInfo,
    VisionEngine,
    VisionResult,
)
from agroai.vision.engines import get_vision_engine
from agroai.vision.image import is_image
from agroai.vision.video import is_video

#: Declared safety rules printed with every media analysis. These are rules, not
#: findings: they hold whether or not the diagnosis found anything.
SAFETY_RULES: tuple[str, ...] = (
    "Tasdiqlanmagan preparat (pestitsid/dori) nomi yoki dozasini qo'llamang.",
    "Hayvonlarga dori berishdan oldin veterinarga murojaat qiling.",
    "Oziq-ovqat xavfsizligiga tegishli qarorlarni laboratoriya tasdig'isiz qabul qilmang.",
)

LIMITATION_NO_VISION = (
    "Vizual tahlil (vision model) bu buildda mavjud emas: o'simlik/hayvon "
    "rasmidan belgi ajratilmadi. Quyidagi ehtimollar siz yozgan tavsif va "
    "bilim bazasidagi dalilga asoslangan."
)


@dataclass
class MediaAnalysis:
    """Full result of analysing one media file plus a description."""

    path: str
    question: str = ""
    media: MediaInfo | None = None
    vision: VisionResult | None = None
    diagnosis: DifferentialDiagnosis | None = None
    pipeline: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    safety_rules: list[str] = field(default_factory=list)
    latency_target_seconds: float = 30.0

    @property
    def vision_available(self) -> bool:
        """True only when a real vision engine produced observations."""
        return bool(
            self.vision is not None and self.vision.ok and self.vision.observations
        )

    @property
    def status(self) -> str:
        if self.media is None:
            return VISION_ERROR
        if not self.vision_available:
            return VISION_NOT_CONFIGURED
        return "ok"

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "question": self.question,
            "status": self.status,
            "vision_available": self.vision_available,
            "media": self.media.to_dict() if self.media is not None else None,
            "vision": self.vision.to_dict() if self.vision is not None else None,
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis is not None else None,
            "pipeline": list(self.pipeline),
            "limitations": list(self.limitations),
            "safety_rules": list(self.safety_rules),
            "latency_target_seconds": self.latency_target_seconds,
        }

    def render(self) -> str:
        """Render the structured answer format from the specification."""
        lines: list[str] = []
        if self.media is not None:
            lines.append(f"Fayl: {self.media.describe()}")
            for note in self.media.notes:
                lines.append(f"  - {note}")
        lines.append("")

        symptoms = self.diagnosis.symptoms if self.diagnosis is not None else []
        if self.diagnosis is not None and self.diagnosis.subject:
            lines.append(f"Ekin/hayvon: {self.diagnosis.subject}")
        lines.append("Kuzatilgan belgilar (sizning tavsifingizdan):")
        if symptoms:
            for symptom in symptoms:
                lines.append(f"  - {symptom}")
        else:
            lines.append("  - (tavsifda tanish belgi topilmadi)")
        if self.vision_available and self.vision is not None:
            lines.append("Vizual belgilar (vision model):")
            for observation in self.vision.observations:
                lines.append(
                    f"  - {observation.label} ({observation.confidence:.0%})"
                )

        if self.diagnosis is not None:
            lines.append("")
            lines.append(self.diagnosis.render())

        lines.append("")
        lines.append("Nima qilmaslik kerak:")
        for rule in self.safety_rules:
            lines.append(f"  - {rule}")

        if self.limitations:
            lines.append("")
            lines.append("Cheklovlar (ochiq ko'rsatilgan):")
            for item in self.limitations:
                lines.append(f"  - {item}")

        if self.vision is not None and self.vision.detail:
            lines.append("")
            lines.append(f"Izoh: {self.vision.detail}")

        lines.append("")
        lines.append(
            "Manbalar: bilim bazasi hujjatlari (yuqoridagi 'manba' qatorlari). "
            "Tashqi internet manbasi ishlatilmadi."
            if self.diagnosis is not None and self.diagnosis.hypotheses
            else "Manbalar: (topilmadi - javob bilim bazasidagi dalilsiz berilmadi)."
        )
        return "\n".join(lines)


class MediaAnalyzer:
    """Runs the media pipeline; engine and retriever are injected."""

    def __init__(
        self,
        *,
        engine: VisionEngine | None = None,
        differential: DifferentialEngine | None = None,
        retriever: Retriever | None = None,
        latency_target_seconds: float | None = None,
    ) -> None:
        self._engine = engine or get_vision_engine()
        if differential is not None:
            self._differential = differential
        elif retriever is not None:
            self._differential = RetrievalDifferentialEngine(retriever)
        else:
            self._differential = None
        self.latency_target_seconds = (
            latency_target_seconds
            if latency_target_seconds is not None
            else getattr(self._engine, "target_latency_seconds", 30.0)
        )

    @property
    def engine(self) -> VisionEngine:
        return self._engine

    def analyze(
        self,
        path: str,
        *,
        question: str = "",
        top_k: int = 8,
        kind: str = "auto",
    ) -> MediaAnalysis:
        """Analyse one file. Never raises for an ordinary bad input."""
        pipeline: list[str] = []
        limitations: list[str] = []

        selected = (kind or "auto").lower()
        if selected == "auto":
            selected = "video" if is_video(path) else ("image" if is_image(path) else "unknown")

        pipeline.append(f"media_kind={selected}")
        if selected == "video":
            vision = self._engine.analyze_video(path)
        elif selected == "image":
            vision = self._engine.analyze_image(path)
        else:
            vision = VisionResult(
                status=VISION_ERROR,
                engine=self._engine.name,
                observations=[],
                detail=(
                    "File is neither a recognised image nor a recognised video "
                    "container. Nothing was analysed."
                ),
            )
            limitations.append(
                "Fayl turi aniqlanmadi (rasm ham, video ham emas): hech qanday fayl "
                "tekshiruvi bajarilmadi."
            )

        pipeline.append(f"vision_engine={vision.engine}")
        pipeline.append(f"vision_status={vision.status}")

        media = vision.media
        if media is not None:
            pipeline.append(f"inspected({media.format}, {media.resolution})")

        if not self._engine.is_configured:
            limitations.append(LIMITATION_NO_VISION)
        if media is not None and not media.inspected:
            limitations.append(
                "Fayl formati aniqlanmadi; u rasm yoki video sifatida qabul qilinmadi."
            )

        diagnosis: DifferentialDiagnosis | None = None
        if self._differential is not None:
            subject = media.format if media is not None else ""
            prompt = question or path
            diagnosis = self._differential.differentiate(prompt, top_k=top_k)
            pipeline.append(f"differential(retrieval, subject={subject or 'n/a'})")
            if diagnosis.refused:
                limitations.append(diagnosis.reason)
        else:
            limitations.append(
                "Diagnostika dvigateli ulanmagan (retriever berilmagan); "
                "faqat fayl tekshiruvi bajarildi."
            )

        return MediaAnalysis(
            path=str(path),
            question=question,
            media=media,
            vision=vision,
            diagnosis=diagnosis,
            pipeline=pipeline,
            limitations=limitations,
            safety_rules=list(SAFETY_RULES),
            latency_target_seconds=self.latency_target_seconds,
        )


def analyze_media(
    path: str,
    *,
    question: str = "",
    retriever: Retriever | None = None,
    engine: VisionEngine | None = None,
    top_k: int = 8,
    kind: str = "auto",
) -> MediaAnalysis:
    """Convenience wrapper around :class:`MediaAnalyzer`."""
    analyzer = MediaAnalyzer(engine=engine, retriever=retriever)
    return analyzer.analyze(path, question=question, top_k=top_k, kind=kind)


__all__ = [
    "LIMITATION_NO_VISION",
    "MediaAnalysis",
    "MediaAnalyzer",
    "SAFETY_RULES",
    "analyze_media",
]
