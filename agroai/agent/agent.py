"""The AgroAI agent: planning + RAG orchestration over retriever, tools and LLM.

Pipeline: question -> plan (safety, retrieval, tools, generation, validation) ->
executor runs the plan -> agent presents the answer with sources and honest
caveats.

The agent is honest by design:

* with no configured LLM it returns an explicit not-configured result (it never
  fabricates an answer) while still showing the retrieved knowledge;
* when the knowledge base has nothing relevant it marks the answer as not
  grounded;
* when a high-risk question (dosage, veterinary medicine, agrochemicals) comes
  back ungrounded, the answer is flagged and the safety notice is attached;
* citations are built only from stored documents, never from model output.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from agroai.agent.executor import ExecutionResult, PlanExecutor
from agroai.agent.memory import Memory, ShortTermMemory
from agroai.agent.planner import Plan, Planner, RuleBasedPlanner
from agroai.agent.prompts import build_context, has_substantive_context
from agroai.knowledge.documents import RetrievedChunk
from agroai.knowledge.sources import SourceCitation, citation_for
from agroai.models.base import (
    NOT_CONFIGURED_MESSAGE,
    LLMProvider,
    LLMResponse,
    ResponseStatus,
)
from agroai.retrieval.retriever import Retriever
from agroai.safety.guardrails import SafetyGuard, SafetyVerdict
from agroai.safety.validation import AnswerValidator, ValidationReport
from agroai.skills.router import SkillRouter, get_skill_router
from agroai.tools.base import ToolRegistry, ToolResult

logger = logging.getLogger(__name__)

SAFETY_WARNING = (
    "\n\n---\nXAVFSIZLIK OGOHLANTIRISHI / SAFETY NOTICE: bu savol "
    "guruhida ({category}) javob faqat bilimlar bazasidagi tasdiqlangan "
    "manbalar bilan berilishi kerak, ammo mos manba topilmadi. Yuqoridagi "
    "javobni tasdiqlanmagan deb qabul qiling va mutaxassis bilan tekshiring."
)

VALIDATION_WARNING = (
    "\n\n---\nVALIDATSIYA OGOHLANTIRISHI / VALIDATION WARNING: "
    "javobdagi ayrim ma'lumotlar manbalar bilan tasdiqlanmadi. "
    "Quyidagi tekshiruv natijalariga qarang."
)


@dataclass
class AgentResult:
    """Everything the caller needs to present an answer responsibly."""

    question: str
    answer: str
    status: str  # "ok" | "not_configured" | "error"
    llm_response: LLMResponse | None = None
    sources: list[RetrievedChunk] = field(default_factory=list)
    grounded: bool = False
    # New in the agent-engine layer:
    plan: Plan | None = None
    safety: SafetyVerdict | None = None
    validation: ValidationReport | None = None
    tool_results: list[ToolResult] = field(default_factory=list)
    citations: list[SourceCitation] = field(default_factory=list)
    safety_blocked: bool = False
    execution: ExecutionResult | None = None

    @property
    def has_sources(self) -> bool:
        return bool(self.sources)

    @property
    def disclaimers(self) -> list[str]:
        """Safety notes that should accompany the answer."""
        if self.safety is None:
            return []
        notes = list(self.safety.disclaimers)
        if self.validation is not None and not self.validation.ok:
            notes.extend(flag.detail for flag in self.validation.critical_flags)
        return notes

    def to_dict(self) -> dict[str, object]:
        return {
            "question": self.question,
            "answer": self.answer,
            "status": self.status,
            "grounded": self.grounded,
            "safety_blocked": self.safety_blocked,
            "sources": [source.source_line for source in self.sources],
            "citations": [citation.to_dict() for citation in self.citations],
            "safety": self.safety.to_dict() if self.safety else None,
            "validation": self.validation.to_dict() if self.validation else None,
            "tools": [result.to_dict() for result in self.tool_results],
        }


class Agent:
    """Retrieval-augmented assistant over the local knowledge base."""

    def __init__(
        self,
        retriever: Retriever,
        llm: LLMProvider,
        memory: Memory | None = None,
        top_k: int = 5,
        *,
        planner: Planner | None = None,
        tools: ToolRegistry | None = None,
        guard: SafetyGuard | None = None,
        validator: AnswerValidator | None = None,
        skills: "SkillRouter | None" = None,
        user_context: str = "",
    ) -> None:
        self._retriever = retriever
        self._llm = llm
        self._memory = memory or ShortTermMemory()
        self._top_k = top_k
        self._guard = guard or SafetyGuard()
        self._skills = skills or get_skill_router()
        self._planner = planner or RuleBasedPlanner(
            guard=self._guard, tools=tools, skills=self._skills
        )
        self._executor = PlanExecutor(
            retriever=retriever,
            llm=llm,
            tools=tools,
            guard=self._guard,
            validator=validator,
            skills=self._skills,
            user_context=user_context,
            top_k=top_k,
        )

    # Exposed components (useful for the CLI/doctor and for tests) --------- #
    @property
    def retriever(self) -> Retriever:
        return self._retriever

    @property
    def skills(self) -> SkillRouter:
        """The skill router the planner and executor share."""
        return self._skills

    @property
    def guard(self) -> SafetyGuard:
        return self._guard

    @property
    def planner(self) -> Planner:
        return self._planner

    @property
    def executor(self) -> PlanExecutor:
        return self._executor

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def llm(self) -> LLMProvider:
        return self._llm

    @property
    def memory(self) -> Memory:
        return self._memory

    @property
    def planner(self) -> Planner:
        return self._planner

    @property
    def executor(self) -> PlanExecutor:
        return self._executor

    @property
    def guard(self) -> SafetyGuard:
        return self._guard

    @property
    def tools(self) -> ToolRegistry:
        return self._executor.tools

    @property
    def validator(self) -> AnswerValidator:
        return self._executor.validator

    # ------------------------------------------------------------------ #
    # Pipeline
    # ------------------------------------------------------------------ #
    def plan(self, question: str) -> Plan:
        """Return the plan the agent would execute (used by ``agroai plan``)."""
        return self._planner.plan(question)

    def ask(self, question: str, *, use_memory: bool = False) -> AgentResult:
        """Run the full plan-and-execute RAG pipeline for one question."""
        question = (question or "").strip()
        if not question:
            return AgentResult(
                question="",
                answer="Bo'sh savol. / Empty question.",
                status="error",
            )

        plan = self._planner.plan(question)
        execution = self._executor.execute(
            plan, memory=self._memory, use_memory=use_memory
        )
        citations = [
            citation_for(retrieved.document)
            for retrieved in execution.chunks
            if retrieved.document is not None
        ]

        # 1) No model: retrieval-only result, honestly labelled.
        if not self._llm.is_configured:
            logger.info("LLM not configured; returning retrieval-only result.")
            return self._retrieval_only_result(question, plan, execution, citations)

        # 2) A model answered (or failed): report it as it is.
        response = execution.llm_response
        if response is None:
            return AgentResult(
                question=question,
                answer="Ijro bosqichi javob qaytarmadi. / The execution produced no answer.",
                status="error",
                plan=plan,
                safety=plan.verdict,
                sources=execution.chunks,
                citations=citations,
                execution=execution,
            )

        grounded = (
            response.status is ResponseStatus.OK
            and has_substantive_context(execution.chunks)
        )
        answer = response.text
        safety_blocked = False
        if response.status is ResponseStatus.OK:
            if plan.verdict.requires_sources and not grounded:
                safety_blocked = True
                answer += SAFETY_WARNING.format(category=plan.verdict.category.value)
            if execution.validation is not None and not execution.validation.ok:
                answer += VALIDATION_WARNING

        return AgentResult(
            question=question,
            answer=answer,
            status=(
                "ok"
                if response.status is ResponseStatus.OK
                else ("error" if response.status is ResponseStatus.ERROR else "not_configured")
            ),
            llm_response=response,
            sources=execution.chunks,
            grounded=grounded,
            plan=plan,
            safety=plan.verdict,
            validation=execution.validation,
            tool_results=execution.tool_results,
            citations=citations,
            safety_blocked=safety_blocked,
            execution=execution,
        )

    # ------------------------------------------------------------------ #
    def _retrieval_only_result(
        self,
        question: str,
        plan: Plan,
        execution: ExecutionResult,
        citations: list[SourceCitation],
    ) -> AgentResult:
        """Answer without an LLM: retrieval summary + honest not-configured note."""
        lines: list[str] = []

        for tool_result in execution.tool_results:
            if tool_result.ok:
                lines.append(tool_result.as_text())
                lines.append("")
                lines.append(
                    "Eslatma: yuqoridagi qiymatlar AgroAI vositalari bilan "
                    "siz bergan raqamlardan hisoblandi (bilimlar bazasidan emas). "
                    "/ Note: the values above were computed by AgroAI tools from "
                    "your own numbers, not taken from the knowledge base."
                )
                lines.append("")

        if has_substantive_context(execution.chunks):
            lines.append("Bilimlar bazasidan topilgan bo'limlar / Knowledge base matches:")
            for index, retrieved in enumerate(execution.chunks, start=1):
                snippet = retrieved.chunk.content.strip().replace("\n", " ")
                if len(snippet) > 160:
                    snippet = snippet[:157] + "..."
                lines.append(f"  [{index}] {retrieved.source_line}\n      {snippet}")
            lines.append("")
        else:
            lines.append(
                "Mahalliy bilimlar bazasida bu savol bo'yicha mos ma'lumot topilmadi."
            )
            lines.append("")

        health = self._llm.health_check()
        lines.append(
            "AI model hozircha sozlanmagan / Local LLM not configured: "
            + (health.detail or NOT_CONFIGURED_MESSAGE)
        )
        # Safety disclaimers are exposed through AgentResult.disclaimers so
        # callers (CLI, GUI, API) decide how to present them - not duplicated
        # inside the answer text.

        return AgentResult(
            question=question,
            answer="\n".join(lines),
            status="not_configured",
            sources=execution.chunks,
            grounded=False,
            plan=plan,
            safety=plan.verdict,
            tool_results=execution.tool_results,
            citations=citations,
            execution=execution,
        )


__all__ = ["Agent", "AgentResult", "build_context"]
