"""Routing: which skills does this question belong to?

Deterministic, offline and inspectable: keyword hits per skill, plus a small
weight for the question wording, plus the safety category when it maps onto a
skill. The router never invents a domain and never narrows retrieval when it is
unsure - with no clear winner it returns nothing and the agent searches the whole
knowledge base.

That last rule matters: scoping retrieval to the wrong skill silently loses the
right answer, so the router requires a *clear* winner before it scopes anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from agroai.safety.guardrails import SafetyCategory
from agroai.skills.base import Skill, SkillMatch, SkillRegistry, get_skill_registry

#: Safety categories that imply a skill (used as a corroborating signal).
CATEGORY_SKILL_HINTS: dict[SafetyCategory, str] = {
    SafetyCategory.PLANT_PROTECTION: "plant_disease",
    SafetyCategory.AGROCHEMICAL: "agrochemistry",
    SafetyCategory.DOSAGE: "agrochemistry",
    SafetyCategory.VETERINARY_MEDICINE: "veterinary",
    SafetyCategory.FOOD_SAFETY: "livestock",
}

#: A skill must lead the runner-up by this much before retrieval is scoped.
SCOPE_MARGIN = 1.5


@dataclass
class RoutingResult:
    """The chosen skills plus the retrieval scope they justify."""

    matches: list[SkillMatch]
    scoped_categories: list[str]

    @property
    def names(self) -> list[str]:
        return [match.skill.name for match in self.matches]

    @property
    def is_scoped(self) -> bool:
        return bool(self.scoped_categories)

    def render(self) -> str:
        if not self.matches:
            return "skills: (none matched - searching the whole knowledge base)"
        lines = ["skills: " + ", ".join(
            f"{match.skill.name}({match.score:g})" for match in self.matches
        )]
        lines.append(
            "retrieval scope: " + (", ".join(self.scoped_categories) if self.is_scoped else "(whole knowledge base)")
        )
        return "\n".join(lines)


class SkillRouter:
    """Routes a question to skills and decides whether to scope retrieval."""

    def __init__(
        self,
        registry: SkillRegistry | None = None,
        *,
        max_skills: int = 3,
        scope_margin: float = SCOPE_MARGIN,
    ) -> None:
        self._registry = registry or get_skill_registry()
        self._max_skills = max_skills
        self._scope_margin = scope_margin

    @property
    def registry(self) -> SkillRegistry:
        return self._registry

    def route(
        self,
        question: str,
        *,
        category: SafetyCategory | None = None,
        limit: int | None = None,
    ) -> RoutingResult:
        """Return the skills that match ``question`` (best first)."""
        matches: list[SkillMatch] = []
        for skill in self._registry.all():
            hits = skill.matches(question)
            score = float(len(hits))
            if category is not None and CATEGORY_SKILL_HINTS.get(category) == skill.name:
                score += 1.0
            if score > 0:
                matches.append(SkillMatch(skill=skill, score=score, matched_keywords=hits))
        matches.sort(key=lambda match: (-match.score, match.skill.name))
        matches = matches[: (limit or self._max_skills)]
        return RoutingResult(matches=matches, scoped_categories=self._scope(matches))

    def _scope(self, matches: list[SkillMatch]) -> list[str]:
        """Categories to search, or ``[]`` when the winner is not clear."""
        if not matches:
            return []
        best = matches[0]
        runner_up = matches[1].score if len(matches) > 1 else 0.0
        if best.score < 2 or best.score - runner_up < self._scope_margin:
            # Ambiguous: searching everything is safer than guessing a domain.
            return []
        names = [best.skill.name]
        # A skill of the same strength that shares nothing is a real competitor,
        # but a weaker strongly related skill can contribute its categories too.
        for match in matches[1:]:
            if match.score >= best.score - self._scope_margin and set(match.skill.categories) & set(
                best.skill.categories
            ):
                names.append(match.skill.name)
        return self._registry.categories_of(names)


def skills_context(skills: Iterable[Skill], *, limit: int = 3) -> str:
    """Short, factual prompt block describing the skills in play."""
    skills = list(skills)[:limit]
    if not skills:
        return ""
    lines = ["SOHA / DOMAIN FOCUS:"]
    for skill in skills:
        lines.append(f"- {skill.title}: {skill.guidance or skill.description}")
        if skill.categories:
            lines.append(f"  (knowledge areas: {', '.join(skill.categories)})")
    return "\n".join(lines)


def get_skill_router(registry: SkillRegistry | None = None) -> SkillRouter:
    """Return the default skill router."""
    return SkillRouter(registry or get_skill_registry())


__all__ = [
    "CATEGORY_SKILL_HINTS",
    "SCOPE_MARGIN",
    "RoutingResult",
    "SkillRouter",
    "get_skill_router",
    "skills_context",
]
