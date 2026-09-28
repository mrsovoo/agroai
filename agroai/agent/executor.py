"""Execution: run a :class:`~agroai.agent.planner.Plan` step by step.

The executor owns the mechanical part of the pipeline so the agent can stay
about policy and presentation:

* retrieval failures never take the agent down (they become an empty result),
* tool failures are recorded and reported, never hidden,
* generation only happens when a real provider is configured,
* every generated answer is validated against the sources it was allowed to use.

Each step appends a :class:`StepOutcome` so the CLI and tests can see exactly
what ran, what was skipped, and why.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agroai.agent.memory import Memory
from agroai.agent.planner import (
    ACTION_GENERATE,
    ACTION_RETRIEVE,
    ACTION_SAFETY,
    ACTION_TOOL,
    ACTION_VALIDATE,
    Plan,
    PlanStep,
)
from agroai.agent.prompts import SYSTEM_PROMPT, build_user_prompt
from agroai.knowledge.documents import RetrievedChunk
from agroai.models.base import LLMProvider, LLMResponse, Message, ResponseStatus, Role
from agroai.retrieval.retriever import Retriever
from agroai.safety.guardrails import SafetyGuard, SafetyVerdict
from agroai.safety.validation import AnswerValidator, ValidationReport
from agroai.skills.base import Skill
from agroai.skills.router import SkillRouter, get_skill_router, skills_context
from agroai.tools.base import ToolRegistry, ToolResult, get_default_registry

logger = logging.getLogger(__name__)

STATUS_OK = "ok"
STATUS_SKIPPED = "skipped"
STATUS_FAILED = "failed"


@dataclass
class StepOutcome:
    """What happened when one plan step ran."""

    action: str
    status: str
    detail: str = ""

    def render(self) -> str:
        return f"{self.action}: {self.status}" + (f" - {self.detail}" if self.detail else "")


@dataclass
class ExecutionResult:
    """Everything produced by executing a plan."""

    plan: Plan
    chunks: list[RetrievedChunk] = field(default_factory=list)
    tool_results: list[ToolResult] = field(default_factory=list)
    llm_response: LLMResponse | None = None
    validation: ValidationReport | None = None
    outcomes: list[StepOutcome] = field(default_factory=list)

    @property
    def verdict(self) -> SafetyVerdict:
        return self.plan.verdict

    @property
    def generated(self) -> bool:
        return self.llm_response is not None and self.llm_response.status is ResponseStatus.OK

    @property
    def answer(self) -> str:
        return self.llm_response.text if self.llm_response is not None else ""

    @property
    def sources(self) -> list[RetrievedChunk]:
        return self.chunks

    @property
    def has_substantive_context(self) -> bool:
        from agroai.agent.prompts import has_substantive_context

        return has_substantive_context(self.chunks)

    def step_summary(self) -> str:
        return "\n".join(outcome.render() for outcome in self.outcomes)


class PlanExecutor:
    """Executes plans using a retriever, an LLM, tools and the safety layer."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        *,
        tools: ToolRegistry | None = None,
        guard: SafetyGuard | None = None,
        validator: AnswerValidator | None = None,
        skills: SkillRouter | None = None,
        user_context: str = "",
        top_k: int = 5,
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._tools = tools or get_default_registry()
        self._guard = guard or SafetyGuard()
        self._validator = validator or AnswerValidator()
        self._skills = skills or get_skill_router()
        self._user_context = user_context
        self._top_k = top_k

    # Exposed for the CLI, doctor and tests ------------------------------ #
    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def tools(self) -> ToolRegistry:
        return self._tools

    @property
    def guard(self) -> SafetyGuard:
        return self._guard

    @property
    def validator(self) -> AnswerValidator:
        return self._validator

    @property
    def skills(self) -> SkillRouter:
        return self._skills

    @property
    def top_k(self) -> int:
        return self._top_k

    def set_user_context(self, context: str) -> None:
        """Attach remembered farm context to prompts (empty string clears it)."""
        self._user_context = context or ""

    def selected_skills(self, plan: Plan) -> list[Skill]:
        """The :class:`Skill` objects a plan selected (ignoring unknown names)."""
        found: list[Skill] = []
        for name in plan.skills:
            if name in self._skills.registry:
                found.append(self._skills.registry.get(name))
        return found

    # ------------------------------------------------------------------ #
    # Execution
    # ------------------------------------------------------------------ #
    def execute(
        self, plan: Plan, *, memory: Memory | None = None, use_memory: bool = False
    ) -> ExecutionResult:
        """Run every step of ``plan`` and return the collected result."""
        result = ExecutionResult(plan=plan)

        for step in plan.steps:
            if step.action == ACTION_SAFETY:
                self._record_safety(result, step.params)
            elif step.action == ACTION_RETRIEVE:
                self._retrieve(result, plan.question, step)
            elif step.action == ACTION_TOOL:
                self._run_tool(result, step.params)
            elif step.action == ACTION_GENERATE:
                self._generate(result, plan.question, memory, use_memory)
            elif step.action == ACTION_VALIDATE:
                self._validate(result)
            else:
                result.outcomes.append(
                    StepOutcome(step.action, STATUS_SKIPPED, "unknown action")
                )
        return result

    # ------------------------------------------------------------------ #
    def _record_safety(self, result: ExecutionResult, params: dict) -> None:
        verdict = result.verdict
        detail = f"risk={verdict.risk_level}, requires_sources={verdict.requires_sources}"
        result.outcomes.append(StepOutcome(ACTION_SAFETY, STATUS_OK, detail))

    def _retrieve(self, result: ExecutionResult, question: str, step: PlanStep) -> None:
        categories = [str(item) for item in step.params.get("categories", []) if item]
        try:
            try:
                chunks = self._retriever.search(
                    question, top_k=self._top_k, categories=categories or None
                )
            except TypeError:
                # A retriever that predates category scoping: search unscoped
                # rather than failing. It is still a real retriever, just a
                # narrower one.
                chunks = self._retriever.search(question, top_k=self._top_k)
        except Exception as exc:  # noqa: BLE001 - retrieval must never take the agent down
            logger.warning("Retrieval failed: %s", exc)
            result.outcomes.append(StepOutcome(ACTION_RETRIEVE, STATUS_FAILED, str(exc)))
            return
        result.chunks = list(chunks)
        scope = f", scope={','.join(categories)}" if categories else ""
        result.outcomes.append(
            StepOutcome(
                ACTION_RETRIEVE,
                STATUS_OK,
                f"{len(result.chunks)} chunk(s) retrieved{scope}",
            )
        )

    def _run_tool(self, result: ExecutionResult, params: dict) -> None:
        name = str(params.get("tool", ""))
        arguments = {key: value for key, value in params.items() if key != "tool"}
        tool_result = self._tools.run(name, **arguments)
        result.tool_results.append(tool_result)
        status = STATUS_OK if tool_result.ok else STATUS_FAILED
        detail = f"{name}: {tool_result.output or tool_result.error}"
        result.outcomes.append(StepOutcome(ACTION_TOOL, status, detail.strip()))

    def _generate(
        self,
        result: ExecutionResult,
        question: str,
        memory: Memory | None,
        use_memory: bool,
    ) -> None:
        if not self._llm.is_configured:
            result.outcomes.append(
                StepOutcome(ACTION_GENERATE, STATUS_SKIPPED, "no LLM configured")
            )
            return

        prompt = build_user_prompt(
            question,
            result.chunks,
            tool_results=result.tool_results,
            safety_rules=self._guard.prompt_rules(result.verdict),
            skill_context=skills_context(self.selected_skills(result.plan)),
            user_context=self._user_context,
        )
        messages: list[Message] = [
            Message(role=Role.SYSTEM, content=_system_prefix())
        ]
        if use_memory and memory is not None:
            messages.extend(memory.history())
        messages.append(Message(role=Role.USER, content=prompt))

        response = self._llm.chat(messages)
        result.llm_response = response
        if response.status is ResponseStatus.OK:
            result.outcomes.append(
                StepOutcome(ACTION_GENERATE, STATUS_OK, f"{len(response.text)} chars")
            )
            if memory is not None and use_memory:
                memory.add(Message(role=Role.USER, content=question))
                memory.add(Message(role=Role.ASSISTANT, content=response.text))
        else:
            result.outcomes.append(
                StepOutcome(ACTION_GENERATE, STATUS_FAILED, response.status.value)
            )

    def _validate(self, result: ExecutionResult) -> None:
        if not result.generated:
            result.outcomes.append(
                StepOutcome(ACTION_VALIDATE, STATUS_SKIPPED, "nothing to validate")
            )
            return
        report = self._validator.validate(result.answer, result.chunks, result.verdict)
        result.validation = report
        status = STATUS_OK if report.ok else STATUS_FAILED
        result.outcomes.append(
            StepOutcome(
                ACTION_VALIDATE,
                status,
                f"severity={report.severity}, coverage={report.coverage:.2f}",
            )
        )


def _system_prefix() -> str:
    """The system prompt without the context placeholder.

    The per-question context (and tool output) is embedded in the user turn, so
    the system prompt is sent without retrieval data.
    """
    return SYSTEM_PROMPT.split("CONTEXT (retrieved chunks with sources):")[0].strip()


__all__ = [
    "STATUS_FAILED",
    "STATUS_OK",
    "STATUS_SKIPPED",
    "ExecutionResult",
    "PlanExecutor",
    "StepOutcome",
]
