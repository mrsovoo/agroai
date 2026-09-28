"""Tests for the agent: RAG flow, honest degradation, memory, prompts."""

from __future__ import annotations

from agroai.agent import Agent, build_context, build_user_prompt, has_substantive_context
from agroai.agent.memory import ShortTermMemory
from agroai.knowledge.models import Chunk, KnowledgeDocument, RetrievedChunk
from agroai.models.base import LLMResponse, Message, ResponseStatus, Role


def _chunk(text: str, doc_id: str = "d1", title: str = "Pomidor") -> RetrievedChunk:
    document = KnowledgeDocument(
        id=doc_id, title=title, content="c", source="local_test", source_url=""
    )
    chunk = Chunk(id=f"{doc_id}:0", document_id=doc_id, chunk_index=0, content=text)
    return RetrievedChunk(chunk=chunk, score=1.0, document=document)


def _stub_llm(configured: bool = True, answer: str = "Javob: fungitsid ishlatish.") -> object:
    class StubLLM:
        name = "stub"

        @property
        def model(self) -> str:
            return "stub-model" if configured else ""

        @property
        def is_configured(self) -> bool:
            return configured

        def generate(self, prompt: str, **options) -> LLMResponse:
            if not configured:
                return LLMResponse(text="not configured", status=ResponseStatus.NOT_CONFIGURED)
            return LLMResponse(text=answer, status=ResponseStatus.OK, provider="stub")

        def chat(self, messages: list[Message], **options) -> LLMResponse:
            return self.generate("")

        def health_check(self):
            from agroai.models.base import HealthStatus

            return HealthStatus(
                provider="stub",
                configured=configured,
                available=configured,
                model=self.model,
                detail="stub",
            )

    return StubLLM()


def _stub_retriever(chunks):
    class StubRetriever:
        name = "stub"

        def search(self, query: str, top_k: int = 5):
            return list(chunks)[:top_k]

        def health_check(self):
            return None

    return StubRetriever()


# --------------------------------------------------------------------- #
# Prompt / context building
# --------------------------------------------------------------------- #
def test_build_context_includes_sources() -> None:
    chunks = [_chunk("Kech quri yopish belgilari: jigarrang dog'lar.")]
    context = build_context(chunks)
    assert "Pomidor" in context
    assert "local_test" in context
    assert "jigarrang" in context


def test_build_context_empty_when_no_chunks() -> None:
    assert build_context([]) == ""


def test_has_substantive_context() -> None:
    assert has_substantive_context([_chunk("bu yetarli darajada uzun bir bo'lak matn hisoblanadi")]) is True
    assert has_substantive_context([_chunk("qisqa")]) is False
    assert has_substantive_context([]) is False


def test_build_user_prompt_without_context_warns() -> None:
    prompt = build_user_prompt("Savol?", [])
    assert "Savol?" in prompt
    assert "does not contain enough verified information" in prompt


def test_build_user_prompt_with_context() -> None:
    prompt = build_user_prompt("Savol?", [_chunk("Kontekst matni yetarli uzunlikda bo'lishi kerak.")])
    assert "Savol?" in prompt
    assert "Kontekst matni" in prompt


# --------------------------------------------------------------------- #
# Memory
# --------------------------------------------------------------------- #
def test_short_term_memory_bounded() -> None:
    memory = ShortTermMemory(max_messages=4)
    for index in range(10):
        memory.add(Message(role=Role.USER, content=f"msg {index}"))
    history = memory.history()
    assert len(history) == 4
    assert history[-1].content == "msg 9"
    memory.clear()
    assert memory.history() == []


def test_memory_last_user_question() -> None:
    memory = ShortTermMemory()
    memory.add(Message(role=Role.USER, content="birinchi"))
    memory.add(Message(role=Role.ASSISTANT, content="javob"))
    memory.add(Message(role=Role.USER, content="ikkinchi"))
    assert memory.last_user_question() == "ikkinchi"


# --------------------------------------------------------------------- #
# Agent flow
# --------------------------------------------------------------------- #
def test_agent_with_unconfigured_llm_returns_retrieval_only() -> None:
    chunks = [_chunk("Kech quri yopish kasalligi belgilari juda aniq tasvirlangan bo'lak matn.")]
    agent = Agent(retriever=_stub_retriever(chunks), llm=_stub_llm(configured=False))
    result = agent.ask("Pomidor kasalligi qanday boshlanadi?")
    assert result.status == "not_configured"
    assert result.grounded is False
    assert result.has_sources
    assert "Knowledge base matches" in result.answer


def test_agent_without_retrieval_matches_says_so() -> None:
    agent = Agent(retriever=_stub_retriever([]), llm=_stub_llm(configured=False))
    result = agent.ask("Keng qamrovli savol?")
    assert result.status == "not_configured"
    assert "mos ma'lumot topilmadi" in result.answer


def test_agent_full_rag_flow_grounded() -> None:
    chunks = [_chunk("Fungitsidlar haqidagi aniq ma'lumot yetarli darajada uzun bo'lak matn.")]
    agent = Agent(retriever=_stub_retriever(chunks), llm=_stub_llm(configured=True))
    result = agent.ask("Fungitsid qanday qo'llanadi?")
    assert result.status == "ok"
    assert result.grounded is True
    assert result.answer.startswith("Javob:")
    assert result.has_sources


def test_agent_grounded_false_when_llm_configured_but_no_context() -> None:
    agent = Agent(retriever=_stub_retriever([]), llm=_stub_llm(configured=True))
    result = agent.ask("Noma'lum savol?")
    assert result.status == "ok"
    assert result.grounded is False


def test_agent_empty_question_is_error() -> None:
    agent = Agent(retriever=_stub_retriever([]), llm=_stub_llm(configured=True))
    result = agent.ask("   ")
    assert result.status == "error"


def test_agent_uses_memory_when_enabled() -> None:
    chunks = [_chunk("Kontekst yetarli uzunlikka ega bo'lak matn hisoblanadi.")]
    agent = Agent(retriever=_stub_retriever(chunks), llm=_stub_llm(configured=True))
    agent.ask("Birinchi savol", use_memory=True)
    assert agent.memory.last_user_question() == "Birinchi savol"
    agent.ask("Ikkinchi savol", use_memory=True)
    assert agent.memory.last_user_question() == "Ikkinchi savol"
    assert len(agent.memory.history()) == 4  # 2 exchanges
