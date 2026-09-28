"""Tests for the agent engine: planning, execution and safety integration."""

from __future__ import annotations

import pytest

from agroai.agent import (
    ACTION_GENERATE,
    ACTION_RETRIEVE,
    ACTION_SAFETY,
    ACTION_TOOL,
    ACTION_VALIDATE,
    Agent,
    PlanExecutor,
    RuleBasedPlanner,
)
from agroai.knowledge.documents import Chunk, KnowledgeDocument, RetrievedChunk
from agroai.models.base import HealthStatus, LLMResponse, Message, ResponseStatus
from agroai.safety import HIGH, LOW
from agroai.tools import get_default_registry


# --------------------------------------------------------------------- #
# Stubs (duck-typed: the agent must not require concrete subclasses)
# --------------------------------------------------------------------- #
class StubRetriever:
    name = "stub"

    def __init__(self, chunks) -> None:
        self._chunks = list(chunks)
        self.queries: list[str] = []

    def search(self, query: str, top_k: int = 5):
        self.queries.append(query)
        return self._chunks[:top_k]

    def health_check(self):
        return None


class BrokenRetriever:
    name = "broken"

    def search(self, query: str, top_k: int = 5):
        raise RuntimeError("index unavailable")

    def health_check(self):
        return None


class StubLLM:
    name = "stub"

    def __init__(self, *, configured: bool = True, answer: str = "Javob: fungitsid ishlatish.") -> None:
        self._configured = configured
        self._answer = answer
        self.prompts: list[str] = []

    @property
    def model(self) -> str:
        return "stub-model" if self._configured else ""

    @property
    def is_configured(self) -> bool:
        return self._configured

    def generate(self, prompt: str, **options) -> LLMResponse:
        if not self._configured:
            return LLMResponse(text="not configured", status=ResponseStatus.NOT_CONFIGURED)
        return LLMResponse(text=self._answer, status=ResponseStatus.OK, provider="stub")

    def chat(self, messages: list[Message], **options) -> LLMResponse:
        self.prompts.extend(message.content for message in messages if message.role.value == "user")
        return self.generate("")

    def health_check(self) -> HealthStatus:
        return HealthStatus(
            provider="stub",
            configured=self._configured,
            available=self._configured,
            model=self.model,
            detail="stub",
        )


def _retrieved(text: str, doc_id: str = "d1") -> RetrievedChunk:
    document = KnowledgeDocument(id=doc_id, title="Pomidor", content="c", source="FAO")
    chunk = Chunk(id=f"{doc_id}:0", document_id=doc_id, chunk_index=0, content=text)
    return RetrievedChunk(chunk=chunk, score=1.0, document=document)


SUBSTANTIVE = "Fungitsidlar haqidagi aniq ma'lumot yetarli darajada uzun bo'lak matn."


# --------------------------------------------------------------------- #
# Planner
# --------------------------------------------------------------------- #
def test_plan_has_the_expected_steps_in_order() -> None:
    plan = RuleBasedPlanner().plan("Pomidor barglarida sariq dog'lar paydo bo'ldi")
    assert plan.actions() == [
        ACTION_SAFETY,
        ACTION_RETRIEVE,
        ACTION_GENERATE,
        ACTION_VALIDATE,
    ]
    assert plan.question
    assert "Reja/plan:" in plan.render()


def test_plan_adds_a_tool_step_only_with_explicit_parameters() -> None:
    planner = RuleBasedPlanner()

    with_ec = planner.plan("Sirdaryoda sho'rlangan tuproq, EC 6.5 dS/m. Nima qilay?")
    steps = with_ec.tool_steps()
    assert len(steps) == 1
    assert steps[0].params["tool"] == "soil_tool"
    assert steps[0].params["ec_ds_m"] == 6.5
    assert "soil_tool" in with_ec.candidate_tools

    without_number = planner.plan("Sirdaryoda tuproq sho'rlangan, nima qilay?")
    assert without_number.tool_steps() == []
    assert "soil_tool" in without_number.candidate_tools


def test_plan_does_not_invent_tool_parameters() -> None:
    # An EC value far outside any plausible range is not turned into a call.
    plan = RuleBasedPlanner().plan("EC 500 dS/m sho'rlanish haqida nima deysiz?")
    assert plan.tool_steps() == []


def test_plan_records_high_risk_and_professional_verification() -> None:
    plan = RuleBasedPlanner().plan("Pestitsid dozasi qancha?")
    assert plan.risk_level == HIGH
    assert plan.verdict.requires_sources is True
    assert plan.verdict.requires_professional_verification is True
    assert any("professional verification" in note for note in plan.notes)
    assert "Xavf/risk: high" in plan.render()


def test_plan_step_serializes() -> None:
    data = RuleBasedPlanner().plan("Salom").steps[0].to_dict()
    assert data["action"] == ACTION_SAFETY
    assert isinstance(data["params"], dict)


# --------------------------------------------------------------------- #
# Executor
# --------------------------------------------------------------------- #
def test_executor_runs_every_step_when_a_model_is_configured() -> None:
    llm = StubLLM()
    executor = PlanExecutor(StubRetriever([_retrieved(SUBSTANTIVE)]), llm)
    plan = RuleBasedPlanner(tools=get_default_registry()).plan("Fungitsid qanday qo'llanadi?")

    result = executor.execute(plan)
    actions = [outcome.action for outcome in result.outcomes]
    assert actions == [ACTION_SAFETY, ACTION_RETRIEVE, ACTION_GENERATE, ACTION_VALIDATE]
    assert result.generated is True
    assert result.answer.startswith("Javob:")
    assert result.chunks
    assert result.validation is not None
    assert "fungitsid" in llm.prompts[0].lower()


def test_executor_skips_generation_without_a_model() -> None:
    executor = PlanExecutor(StubRetriever([_retrieved(SUBSTANTIVE)]), StubLLM(configured=False))
    plan = RuleBasedPlanner().plan("Fungitsid qanday qo'llanadi?")
    result = executor.execute(plan)

    statuses = {outcome.action: outcome.status for outcome in result.outcomes}
    assert statuses[ACTION_GENERATE] == "skipped"
    assert statuses[ACTION_VALIDATE] == "skipped"
    assert result.validation is None


def test_executor_survives_retrieval_failure() -> None:
    executor = PlanExecutor(BrokenRetriever(), StubLLM(configured=False))
    result = executor.execute(RuleBasedPlanner().plan("Savol?"))
    statuses = {outcome.action: outcome.status for outcome in result.outcomes}
    assert statuses[ACTION_RETRIEVE] == "failed"
    assert result.chunks == []


def test_executor_runs_tool_steps_and_reports_failures() -> None:
    executor = PlanExecutor(StubRetriever([]), StubLLM(configured=False))
    plan = RuleBasedPlanner().plan("Tuproq sho'rlanishi EC 6.5 dS/m")
    result = executor.execute(plan)

    assert len(result.tool_results) == 1
    assert result.tool_results[0].ok is True
    assert result.tool_results[0].data["class"] == "moderately saline"

    failing = executor.tools.run("soil_tool", action="salinity", ec_ds_m="abc")
    assert failing.ok is False


def test_executor_records_unknown_action_as_skipped() -> None:
    from agroai.agent.planner import Plan, PlanStep

    executor = PlanExecutor(StubRetriever([]), StubLLM())
    plan = Plan(question="x", steps=[PlanStep("nonexistent_action")])
    result = executor.execute(plan)
    assert result.outcomes[0].status == "skipped"


def test_executor_respects_memory_flag() -> None:
    from agroai.agent.memory import ShortTermMemory

    memory = ShortTermMemory()
    executor = PlanExecutor(StubRetriever([_retrieved(SUBSTANTIVE)]), StubLLM())
    plan = RuleBasedPlanner().plan("Birinchi savol")

    executor.execute(plan, memory=memory, use_memory=True)
    assert memory.last_user_question() == "Birinchi savol"

    executor.execute(plan, memory=memory, use_memory=False)
    assert len(memory.history()) == 2


# --------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------- #
def test_agent_exposes_plan_and_execution() -> None:
    agent = Agent(retriever=StubRetriever([_retrieved(SUBSTANTIVE)]), llm=StubLLM())
    result = agent.ask("Fungitsid qanday qo'llanadi?")
    assert result.plan is not None
    assert result.execution is not None
    assert result.citations
    assert result.citations[0].title == "Pomidor"
    assert result.status == "ok"
    assert result.grounded is True
    assert result.safety_blocked is False


def test_agent_blocks_high_risk_answer_without_sources() -> None:
    llm = StubLLM(answer="Har gektarga 3 kg fungitsid sepiladi.")
    agent = Agent(retriever=StubRetriever([]), llm=llm)
    result = agent.ask("Fungitsid dozasi qancha?")

    assert result.safety_blocked is True
    assert "SAFETY NOTICE" in result.answer
    assert result.validation is not None
    # The invented quantity and the missing sources are both reported.
    assert result.validation.ok is False
    assert any("unsupported_quantity" == flag.code for flag in result.validation.flags)
    assert result.disclaimers


def test_agent_does_not_block_low_risk_questions_without_sources() -> None:
    agent = Agent(retriever=StubRetriever([]), llm=StubLLM(answer="Aniq javob."))
    result = agent.ask("Salom, qanday yordam bera olasiz?")
    assert result.safety_blocked is False
    assert "SAFETY NOTICE" not in result.answer
    assert result.safety is not None
    assert result.safety.risk_level == LOW


def test_agent_uses_tools_and_reports_them() -> None:
    agent = Agent(retriever=StubRetriever([]), llm=StubLLM(configured=False))
    result = agent.ask("Sirdaryoda tuproq sho'rlangan, EC 6.5 dS/m. Nima qilay?")

    assert result.status == "not_configured"
    assert len(result.tool_results) == 1
    assert "moderately saline" in result.answer
    assert "bilimlar bazasidan emas" in result.answer


def test_agent_result_to_dict_is_serializable() -> None:
    import json

    agent = Agent(retriever=StubRetriever([_retrieved(SUBSTANTIVE)]), llm=StubLLM())
    data = agent.ask("Fungitsid qanday qo'llanadi?").to_dict()
    assert json.dumps(data)  # no non-serializable values
    assert data["status"] == "ok"
    assert data["safety"]["risk_level"] == HIGH


def test_agent_accepts_an_injected_planner() -> None:
    class FixedPlanner(RuleBasedPlanner):
        def plan(self, question: str):
            return super().plan("pestitsid dozasi")

    agent = Agent(
        retriever=StubRetriever([_retrieved(SUBSTANTIVE)]),
        llm=StubLLM(),
        planner=FixedPlanner(),
    )
    plan = agent.plan("Salom")
    assert plan.risk_level == HIGH
