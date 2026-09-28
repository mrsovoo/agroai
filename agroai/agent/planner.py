"""Planning: turn a question into an explicit, auditable plan.

This is the step that separates an agent from a chatbot. Before anything is
retrieved or generated, the question is turned into an ordered list of steps:

1. assess safety (which rules apply to this question),
2. retrieve knowledge (always - an answer without sources is not acceptable),
3. run tools (only when the planner can name every parameter it needs),
4. generate the answer,
5. validate the answer against the retrieved sources.

The planner is rule-based on purpose: it is fast, offline, deterministic and
inspectable (``agroai plan "<question>"`` prints exactly what it decided). It
never invents tool parameters - if a number is missing, the tool step is simply
not planned, and the plan says so.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from agroai.safety.guardrails import SafetyGuard, SafetyVerdict
from agroai.skills.router import RoutingResult, SkillRouter, get_skill_router
from agroai.tools.base import ToolRegistry, get_default_registry

#: Step actions the executor understands.
ACTION_SAFETY = "assess_safety"
ACTION_RETRIEVE = "retrieve_knowledge"
ACTION_TOOL = "run_tool"
ACTION_GENERATE = "generate_answer"
ACTION_VALIDATE = "validate_answer"

#: Execution order used by the executor and reported by the CLI.
STEP_ORDER = (
    ACTION_SAFETY,
    ACTION_RETRIEVE,
    ACTION_TOOL,
    ACTION_GENERATE,
    ACTION_VALIDATE,
)

NUMBER_PATTERN = re.compile(r"(?<![\w.])(\d+(?:[.,]\d+)?)(?![\w])")

#: Keyword groups that suggest a tool may be relevant (no parameters implied).
TOOL_HINTS: dict[str, tuple[str, ...]] = {
    "soil_tool": (
        "tuproq", "sho'rlan", "shorlan", "sho'r", "ec", "dS/m", "ds/m", "ph",
        "kislotali", "ishqoriy", "saline", "salinity", "soil",
    ),
    "irrigation_tool": (
        "sug'or", "sugor", "suv", "tomchilat", "irrigation", "water", "mm/kun",
        "m3", "norma", "drip",
    ),
    "weather_tool": (
        "ob-havo", "ob havo", "havo", "purkash", "yomg'ir", "shamol", "namlik",
        "weather", "spray", "rain", "wind", "humidity",
    ),
    "veterinary_tool": (
        "qoramol", "sigir", "qo'y", "qoy", "echki", "ot", "tuya", "parranda",
        "tovuq", "hayvon", "mol", "buzoq", "cattle", "cow", "sheep", "goat",
        "poultry", "animal", "veterinar",
    ),
    "calculator": ("hisobla", "hisoblang", "calculate", "qancha", "necha"),
    "unit_converter": ("gektar", "sotka", "ga", "tonna", "sentner", "litr"),
}


@dataclass(frozen=True)
class PlanStep:
    """One planned action, its parameters and why it is there."""

    action: str
    description: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    required: bool = True

    @property
    def tool_name(self) -> str:
        return str(self.params.get("tool", ""))

    def to_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "description": self.description,
            "params": dict(self.params),
            "required": self.required,
        }

    def render(self, index: int) -> str:
        detail = f" {self.params}" if self.params else ""
        suffix = "" if self.required else " (optional)"
        return f"{index}. {self.action}{detail} - {self.description}{suffix}"


@dataclass
class Plan:
    """An ordered plan for one question."""

    question: str
    steps: list[PlanStep] = field(default_factory=list)
    verdict: SafetyVerdict = field(default_factory=SafetyVerdict)
    candidate_tools: list[str] = field(default_factory=list)
    strategy: str = "rule_based"
    notes: list[str] = field(default_factory=list)
    #: Skills the router matched, and the retrieval scope they justify.
    skills: list[str] = field(default_factory=list)
    skill_categories: list[str] = field(default_factory=list)

    def actions(self) -> list[str]:
        return [step.action for step in self.steps]

    def tool_steps(self) -> list[PlanStep]:
        return [step for step in self.steps if step.action == ACTION_TOOL]

    @property
    def risk_level(self) -> str:
        return self.verdict.risk_level

    @property
    def category(self) -> str:
        return self.verdict.category.value

    def render(self) -> str:
        lines = [
            f"Savol/question: {self.question}",
            f"Strategiya/strategy: {self.strategy}",
            f"Xavf/risk: {self.risk_level} ({self.category})",
            "Reja/plan:",
        ]
        lines.extend(step.render(index) for index, step in enumerate(self.steps, start=1))
        if self.skills:
            lines.append("Sohalar/skills: " + ", ".join(self.skills))
            lines.append(
                "Qidiruv doirasi/retrieval scope: "
                + (", ".join(self.skill_categories) if self.skill_categories else "(whole knowledge base)")
            )
        if self.candidate_tools:
            lines.append("Mumkin bo'lgan vositalar/further tools: " + ", ".join(self.candidate_tools))
        for note in self.notes:
            lines.append(f"Eslatma/note: {note}")
        return "\n".join(lines)


class Planner(ABC):
    """Interface for anything that can turn a question into a :class:`Plan`."""

    name: str = "base"

    @abstractmethod
    def plan(self, question: str) -> Plan:
        """Return the plan for ``question``."""


class RuleBasedPlanner(Planner):
    """Deterministic planner: safety, retrieval, tools, generation, validation."""

    name = "rule_based"

    def __init__(
        self,
        guard: SafetyGuard | None = None,
        tools: ToolRegistry | None = None,
        skills: SkillRouter | None = None,
    ) -> None:
        self._guard = guard or SafetyGuard()
        self._tools = tools or get_default_registry()
        self._skills = skills or get_skill_router()

    @property
    def guard(self) -> SafetyGuard:
        return self._guard

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def skills(self) -> SkillRouter:
        return self._skills

    def plan(self, question: str) -> Plan:
        question = (question or "").strip()
        verdict = self._guard.classify(question)
        routing = self._skills.route(question, category=verdict.category)
        plan = Plan(
            question=question,
            verdict=verdict,
            strategy=self.name,
            skills=routing.names,
            skill_categories=routing.scoped_categories,
        )
        if not routing.matches:
            plan.notes.append("no skill matched: searching the whole knowledge base")

        plan.steps.append(
            PlanStep(
                ACTION_SAFETY,
                description=f"safety rules for category '{verdict.category.value}'",
                params={"risk_level": verdict.risk_level, "category": verdict.category.value},
            )
        )
        plan.steps.append(
            PlanStep(
                ACTION_RETRIEVE,
                description="search the local knowledge base for supporting passages",
                params={
                    "min_sources_required": verdict.requires_sources,
                    "categories": list(routing.scoped_categories),
                },
            )
        )

        tool_steps, candidates, notes = self._plan_tools(question, verdict.risk_level)
        plan.candidate_tools = candidates
        plan.notes.extend(notes)

        if verdict.requires_professional_verification:
            plan.notes.append(
                "professional verification required for this category"
            )

        plan.steps.extend(tool_steps)
        plan.steps.append(
            PlanStep(
                ACTION_GENERATE,
                description="compose an answer strictly from the retrieved context",
                params={"grounded_only": True},
            )
        )
        plan.steps.append(
            PlanStep(
                ACTION_VALIDATE,
                description="check citations, quantities and grounding coverage",
                params={"min_coverage": 0.25},
            )
        )
        return plan

    # ------------------------------------------------------------------ #
    def _plan_tools(
        self, question: str, risk_level: str
    ) -> tuple[list[PlanStep], list[str], list[str]]:
        """Build tool steps only where every parameter is unambiguous."""
        steps: list[PlanStep] = []
        notes: list[str] = []
        lowered = question.lower()
        numbers = [float(value.replace(",", ".")) for value in NUMBER_PATTERN.findall(lowered)]

        candidates = sorted(
            {
                tool
                for tool, keywords in TOOL_HINTS.items()
                if tool in self._tools and any(keyword in lowered for keyword in keywords)
            }
        )

        step = self._salinity_step(lowered, numbers, risk_level)
        if step is not None:
            steps.append(step)

        return steps, candidates, notes

    def _salinity_step(
        self, lowered: str, numbers: list[float], risk_level: str
    ) -> PlanStep | None:
        """Plan a salinity classification only when an EC value is explicit."""
        if "soil_tool" not in self._tools:
            return None
        mentions_salinity = any(
            keyword in lowered for keyword in ("sho'rlan", "shorlan", "ec", "dS/m", "saline", "salinity")
        )
        if not mentions_salinity or not numbers:
            return None
        ec = numbers[0]
        if not 0 <= ec <= 100:
            return None
        return PlanStep(
            ACTION_TOOL,
            description="classify soil salinity from the EC value in the question",
            params={"tool": "soil_tool", "action": "salinity", "ec_ds_m": ec},
            required=False,
        )


def build_planner(
    guard: SafetyGuard | None = None,
    tools: ToolRegistry | None = None,
    skills: SkillRouter | None = None,
) -> Planner:
    """Return the planner implementation AgroAI uses."""
    return RuleBasedPlanner(guard=guard, tools=tools, skills=skills)


__all__ = [
    "ACTION_GENERATE",
    "ACTION_RETRIEVE",
    "ACTION_SAFETY",
    "ACTION_TOOL",
    "ACTION_VALIDATE",
    "NUMBER_PATTERN",
    "STEP_ORDER",
    "TOOL_HINTS",
    "Plan",
    "PlanStep",
    "Planner",
    "RoutingResult",
    "RuleBasedPlanner",
    "build_planner",
]
