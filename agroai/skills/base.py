"""Skills: named domains that scope knowledge, tools and guidance.

A skill is not a model and not a collection of facts. It is the glue the roadmap
describes:

```text
Tomato disease question -> plant_disease skill -> plant_diseases + crops knowledge
                        -> soil skill (context)  -> soil knowledge
                        -> tools: soil_tool, ...
```

Each skill declares:

* ``categories`` - the knowledge categories it draws on (used to scope retrieval),
* ``tools`` - the tools that make sense for it (used by the planner),
* ``keywords`` - multilingual triggers for routing (uz / ru / en),
* ``guidance`` - one short, factual instruction added to the prompt.

Everything is declared data, so routing and scoping stay inspectable and no
skill can smuggle unverified advice into an answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable

from agroai.safety.guardrails import HIGH, LOW, MEDIUM, RISK_ORDER


@dataclass(frozen=True)
class Skill:
    """One agricultural domain AgroAI can focus on."""

    name: str
    title: str
    description: str
    categories: tuple[str, ...] = ()
    tools: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    risk_level: str = LOW
    guidance: str = ""

    def matches(self, text: str) -> list[str]:
        """Keyword hits in ``text`` (lowercased comparison)."""
        lowered = (text or "").lower()
        return [keyword for keyword in self.keywords if keyword in lowered]

    @property
    def is_high_risk(self) -> bool:
        return self.risk_level == HIGH

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "title": self.title,
            "description": self.description,
            "categories": list(self.categories),
            "tools": list(self.tools),
            "keywords": list(self.keywords),
            "risk_level": self.risk_level,
            "guidance": self.guidance,
        }


@dataclass
class SkillMatch:
    """A skill that matched a question, with why it matched."""

    skill: Skill
    score: float
    matched_keywords: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill.name,
            "score": round(self.score, 3),
            "matched_keywords": list(self.matched_keywords),
        }


class SkillRegistry:
    """The set of skills AgroAI knows about."""

    def __init__(self, skills: Iterable[Skill] = ()) -> None:
        self._skills: dict[str, Skill] = {}
        for skill in skills:
            self.register(skill)

    def register(self, skill: Skill) -> None:
        if skill.name in self._skills:
            raise ValueError(f"a skill named {skill.name!r} is already registered")
        self._skills[skill.name] = skill

    def __contains__(self, name: object) -> bool:
        return name in self._skills

    def __len__(self) -> int:
        return len(self._skills)

    def names(self) -> list[str]:
        return sorted(self._skills)

    def get(self, name: str) -> Skill:
        try:
            return self._skills[name]
        except KeyError as exc:
            raise KeyError(
                f"unknown skill {name!r}; available: {', '.join(self.names()) or '(none)'}"
            ) from exc

    def all(self) -> list[Skill]:
        return [self._skills[name] for name in self.names()]

    def categories_of(self, names: Iterable[str]) -> list[str]:
        """Union of the categories of the named skills (order preserved)."""
        found: list[str] = []
        for name in names:
            skill = self._skills.get(name)
            if skill is None:
                continue
            for category in skill.categories:
                if category not in found:
                    found.append(category)
        return found

    def tools_of(self, names: Iterable[str]) -> list[str]:
        found: list[str] = []
        for name in names:
            skill = self._skills.get(name)
            if skill is None:
                continue
            for tool in skill.tools:
                if tool not in found:
                    found.append(tool)
        return found

    def risk_level_of(self, names: Iterable[str]) -> str:
        """Highest risk level of the named skills."""
        level = LOW
        for name in names:
            skill = self._skills.get(name)
            if skill is None:
                continue
            if RISK_ORDER[skill.risk_level] > RISK_ORDER[level]:
                level = skill.risk_level
        return level

    def describe(self) -> str:
        lines: list[str] = []
        for skill in self.all():
            lines.append(f"- {skill.name}: {skill.description}")
            if skill.categories:
                lines.append(f"    knowledge: {', '.join(skill.categories)}")
            if skill.tools:
                lines.append(f"    tools: {', '.join(skill.tools)}")
        return "\n".join(lines)

    def requirements(self) -> dict[str, list[str]]:
        """Every model/tool a skill set needs (used by doctor and the roadmap)."""
        return {
            "tools": sorted({tool for skill in self.all() for tool in skill.tools}),
            "categories": sorted({category for skill in self.all() for category in skill.categories}),
            "skills": self.names(),
        }


def get_skill_registry() -> SkillRegistry:
    """Return the default skill registry.

    Defined here as a lazy import so :mod:`agroai.skills.catalog` can import this
    module without a cycle.
    """
    from agroai.skills.catalog import DEFAULT_SKILLS

    return SkillRegistry(DEFAULT_SKILLS)


__all__ = [
    "HIGH",
    "LOW",
    "MEDIUM",
    "Skill",
    "SkillMatch",
    "SkillRegistry",
    "get_skill_registry",
]
