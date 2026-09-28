"""Safety layer: guardrails before the answer, validation after it.

* :mod:`agroai.safety.guardrails` decides how careful an answer must be
  (risk category, mandatory grounding, disclaimers, extra prompt rules).
* :mod:`agroai.safety.validation` checks the answer that came back
  (invented links, unsupported dosages, overconfidence, grounding coverage).
"""

from __future__ import annotations

from agroai.safety.guardrails import (
    CATEGORY_PATTERNS,
    DISCLAIMERS,
    HIGH,
    LOW,
    MEDIUM,
    PROMPT_RULES,
    RISK_ORDER,
    SafetyCategory,
    SafetyGuard,
    SafetyVerdict,
)
from agroai.safety.validation import (
    CRITICAL,
    INFO,
    WARNING,
    AnswerValidator,
    ValidationFlag,
    ValidationReport,
    grounding_coverage,
)

__all__ = [
    "CATEGORY_PATTERNS",
    "CRITICAL",
    "DISCLAIMERS",
    "HIGH",
    "INFO",
    "LOW",
    "MEDIUM",
    "PROMPT_RULES",
    "RISK_ORDER",
    "AnswerValidator",
    "SafetyCategory",
    "SafetyGuard",
    "SafetyVerdict",
    "ValidationFlag",
    "ValidationReport",
    "WARNING",
    "grounding_coverage",
]
