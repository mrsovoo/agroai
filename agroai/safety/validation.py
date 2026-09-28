"""Answer validation: check what the model actually produced.

Guardrails decide how careful an answer must be; validation checks whether the
answer we got is acceptable. Every check here is deterministic and offline, so a
reviewer can reproduce it:

* **fabricated links** - any URL in the answer that is not present in the
  retrieved sources (the roadmap's "never invent a source" rule, enforced);
* **unsupported quantities** - dosages/amounts in the answer that appear nowhere
  in the retrieved context (the highest-risk kind of hallucination);
* **overconfidence** - absolute-certainty phrasing about a probabilistic topic;
* **grounding coverage** - how much of the answer's vocabulary actually comes
  from the retrieved context;
* **missing required sources** - a high-risk question answered with no usable
  context at all.

The validator reports; it does not rewrite the answer. The agent decides what to
do with the report, and the CLI shows it.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from agroai.knowledge.documents import RetrievedChunk
from agroai.safety.guardrails import LOW, SafetyVerdict

#: Flag severities.
INFO = "info"
WARNING = "warning"
CRITICAL = "critical"

_URL_PATTERN = re.compile(r"https?://[^\s)\]}>\"']+", re.IGNORECASE)
_QUANTITY_PATTERN = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*(?:mg|ml|g|kg|%|ppm|l|litr|gramm|milligram|doza|"
    r"konsentratsiya)\b",
    re.IGNORECASE,
)
_WORD_PATTERN = re.compile(r"[a-zA-Zа-яА-ЯёЁ\u0400-\u04FF']{3,}")
_OVERCONFIDENCE = (
    "100%",
    "hech shubhasiz",
    "shubhasiz",
    "mutlaqo aniq",
    "aniq tashxis",
    "albatta davolaydi",
    "definitely",
    "certainly",
    "guaranteed",
    "is guaranteed to",
)

#: Words that carry no signal for grounding coverage.
_STOPWORDS = {
    "the", "and", "for", "with", "that", "this", "from", "are", "was", "were",
    "uchun", "bilan", "ham", "yoki", "lekin", "bo'ladi", "bo'lgan", "kerak",
    "siz", "biz", "ular", "bu", "shu", "bir", "hamda", "yana",
}


@dataclass
class ValidationFlag:
    """One problem found in an answer."""

    code: str
    severity: str
    detail: str


@dataclass
class ValidationReport:
    """Everything validation could establish about one answer."""

    grounded: bool = False
    coverage: float = 0.0
    flags: list[ValidationFlag] = field(default_factory=list)
    fabricated_urls: list[str] = field(default_factory=list)
    unsupported_quantities: list[str] = field(default_factory=list)

    @property
    def critical_flags(self) -> list[ValidationFlag]:
        return [flag for flag in self.flags if flag.severity == CRITICAL]

    @property
    def ok(self) -> bool:
        """True when nothing critical was found (warnings are still shown)."""
        return not self.critical_flags

    @property
    def severity(self) -> str:
        if self.critical_flags:
            return CRITICAL
        if any(flag.severity == WARNING for flag in self.flags):
            return WARNING
        return INFO

    def render(self) -> str:
        if not self.flags:
            return "Validatsiya: muammo topilmadi / validation: no issues found."
        lines = ["Validatsiya / validation:"]
        lines.extend(f"  [{flag.severity}] {flag.code}: {flag.detail}" for flag in self.flags)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, object]:
        return {
            "grounded": self.grounded,
            "coverage": round(self.coverage, 4),
            "severity": self.severity,
            "ok": self.ok,
            "fabricated_urls": list(self.fabricated_urls),
            "unsupported_quantities": list(self.unsupported_quantities),
            "flags": [
                {"code": flag.code, "severity": flag.severity, "detail": flag.detail}
                for flag in self.flags
            ],
        }


def _normalize(text: str) -> str:
    return (text or "").lower()


def _words(text: str) -> set[str]:
    return {
        word
        for word in _WORD_PATTERN.findall(_normalize(text))
        if word not in _STOPWORDS
    }


def grounding_coverage(answer: str, context: str) -> float:
    """Fraction of the answer's significant words that occur in the context."""
    answer_words = _words(answer)
    if not answer_words:
        return 0.0
    context_words = _words(context)
    if not context_words:
        return 0.0
    return len(answer_words & context_words) / len(answer_words)


def _normalize_quantity(raw: str) -> str:
    return re.sub(r"\s+", "", _normalize(raw))


class AnswerValidator:
    """Validate a generated answer against the retrieved sources."""

    def __init__(self, *, min_coverage: float = 0.25) -> None:
        self._min_coverage = min_coverage

    @property
    def min_coverage(self) -> float:
        return self._min_coverage

    def validate(
        self,
        answer: str,
        chunks: Iterable[RetrievedChunk] = (),
        verdict: SafetyVerdict | None = None,
    ) -> ValidationReport:
        """Check one answer. ``chunks`` are the sources it was allowed to use."""
        chunks = list(chunks)
        answer = answer or ""
        report = ValidationReport()

        origin_urls = {
            chunk.document.source_url
            for chunk in chunks
            if chunk.document is not None and chunk.document.source_url
        }
        context = "\n".join(
            [chunk.chunk.content for chunk in chunks]
            + [chunk.document.source_url for chunk in chunks if chunk.document is not None]
            + [chunk.document.title for chunk in chunks if chunk.document is not None]
        )

        # 1) Invented links.
        for url in _URL_PATTERN.findall(answer):
            if url.rstrip(".,;") not in origin_urls and url.rstrip(".,;") not in context:
                report.fabricated_urls.append(url)
        if report.fabricated_urls:
            report.flags.append(
                ValidationFlag(
                    "fabricated_url",
                    CRITICAL,
                    "javobda manbalar orasida bo'lmagan havola: "
                    + ", ".join(report.fabricated_urls)
                    + " / URL(s) not present in the retrieved sources",
                )
            )

        # 2) Quantities the context cannot support.
        context_quantities = {
            _normalize_quantity(match) for match in _QUANTITY_PATTERN.findall(context)
        }
        for match in _QUANTITY_PATTERN.findall(answer):
            if _normalize_quantity(match) not in context_quantities:
                report.unsupported_quantities.append(match.strip())
        if report.unsupported_quantities:
            report.flags.append(
                ValidationFlag(
                    "unsupported_quantity",
                    CRITICAL,
                    "manbalarda uchramaydigan miqdor/doza: "
                    + ", ".join(dict.fromkeys(report.unsupported_quantities))
                    + " / quantit(ies) absent from the retrieved sources",
                )
            )

        # 3) Grounding coverage.
        report.coverage = grounding_coverage(answer, context)
        report.grounded = bool(context.strip()) and report.coverage >= self._min_coverage
        if context.strip() and not report.grounded:
            report.flags.append(
                ValidationFlag(
                    "low_grounding",
                    WARNING,
                    f"javob asoslanganlik darajasi past (coverage={report.coverage:.2f} < "
                    f"{self._min_coverage:.2f}) / answer overlaps little with the context",
                )
            )

        # 4) Overconfidence.
        lowered = _normalize(answer)
        found_overconfidence = [phrase for phrase in _OVERCONFIDENCE if phrase in lowered]
        if found_overconfidence:
            severity = WARNING if (verdict and verdict.risk_level != LOW) else INFO
            report.flags.append(
                ValidationFlag(
                    "overconfidence",
                    severity,
                    "mutlaq ishonch ifodasi: " + ", ".join(found_overconfidence)
                    + " / absolute-certainty phrasing on a probabilistic topic",
                )
            )

        # 5) A high-risk question answered without sources at all.
        if verdict is not None and verdict.requires_sources and not context.strip():
            report.flags.append(
                ValidationFlag(
                    "missing_required_sources",
                    CRITICAL,
                    f"'{verdict.category.value}' turidagi savol uchun manba topilmadi "
                    "/ no source was retrieved for a question that requires one",
                )
            )

        return report


__all__ = [
    "CRITICAL",
    "INFO",
    "WARNING",
    "AnswerValidator",
    "ValidationFlag",
    "ValidationReport",
    "grounding_coverage",
]
