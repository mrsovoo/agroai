"""System prompts and context building for the RAG agent.

The prompt encodes AgroAI's contract with the user:
* answers are grounded in retrieved local knowledge;
* the assistant distinguishes documented facts / likely possibilities /
  uncertainty / practical recommendations;
* when the knowledge base is insufficient it says so explicitly;
* sources are only ever cited from retrieved chunks - never invented;
* for veterinary/chemical/dosage topics it recommends professional
  verification.

Tool output is added in its own clearly-labelled block, because a computed value
(unit conversion, ETc arithmetic) is *not* knowledge-base evidence and must not
be presented as if it were a source.
"""

from __future__ import annotations

from typing import Any, Iterable

from agroai.knowledge.documents import RetrievedChunk

MIN_CONTEXT_CHARS = 40

SYSTEM_PROMPT = """\
Siz Qishloq xo'jaligi va veterinariya yordamchisi - AgroAI.

Qoidalar:
1. Javobingiz asosan berilgan KONTEKSTGA tayanadi. Kontekstdagi ma'lumotdan \
tashqari faktlarni o'ylab topmang.
2. Kontekst yetarli bo'lmasa, aniq ayting: "Mahalliy bilimlar bazasida \
bu savol uchun yetarli tasdiqlangan ma'lumot yo'q".
3. Manbalarni faqat kontekstdan keltiring (sarlavha va manba). Hech qachon \
manba yoki havolani o'ylab chiqarmang.
4. Ajratib ko'rsating: tasdiqlangan fakt / ehtimoliy sabab / noaniqlik / \
amaliy tavsiya.
5. Veterinariya, pestitsid, gerbitsid, fungitsid, dorilar va dozalar bo'yicha \
javoblarda ehtiyot bo'ling va mutaxassis tekshiruvini tavsiya eting.
6. Javobni foydalanuvchi tilida bering (o'zbek, rus yoki ingliz).

CONTEXT (retrieved chunks with sources):
{context}
"""

NO_CONTEXT_SUFFIX = """
CONTEXT: (the local knowledge base returned no relevant chunks for this \
question). If you still answer, state clearly that the local knowledge base
does not contain enough verified information, and do not invent sources.
"""

TOOL_RESULTS_HEADER = """
TOOL NATIJALARI / TOOL RESULTS (these values were computed by AgroAI tools from \
the numbers the user supplied - they are arithmetic, NOT knowledge-base sources):
"""

SAFETY_HEADER = """
USHBU SAVOL UCHUN QO'SHIMCHA XAVFSIZLIK QOIDALARI / EXTRA SAFETY RULES FOR THIS \
QUESTION:
"""


def build_context(chunks: list[RetrievedChunk], max_chars: int = 6000) -> str:
    """Format retrieved chunks (with provenance) into a prompt context block."""
    if not chunks:
        return ""
    blocks: list[str] = []
    used = 0
    for index, retrieved in enumerate(chunks, start=1):
        block = (
            f"[{index}] {retrieved.source_line}\n"
            f"{retrieved.chunk.content.strip()}"
        )
        if used + len(block) > max_chars and blocks:
            break
        blocks.append(block)
        used += len(block) + 2
    return "\n\n".join(blocks)


def build_tool_context(tool_results: Iterable[Any] | None) -> str:
    """Format tool output for the prompt (empty when no tool produced output)."""
    results = [result for result in (tool_results or []) if getattr(result, "ok", False)]
    if not results:
        return ""
    blocks = [TOOL_RESULTS_HEADER.strip()]
    for result in results:
        blocks.append(result.as_text())
    return "\n".join(blocks)


def has_substantive_context(chunks: list[RetrievedChunk]) -> bool:
    """True when retrieval returned at least one chunk with real content."""
    return any(len(chunk.chunk.content.strip()) >= MIN_CONTEXT_CHARS for chunk in chunks)


def build_user_prompt(
    question: str,
    chunks: list[RetrievedChunk],
    *,
    tool_results: Iterable[Any] | None = None,
    safety_rules: str = "",
    skill_context: str = "",
    user_context: str = "",
) -> str:
    """Build the user turn: question, domain focus, memory, tools, context."""
    question = (question or "").strip()
    parts: list[str] = [question]

    if skill_context:
        parts.append(skill_context)
    if user_context:
        parts.append(user_context)

    tool_context = build_tool_context(tool_results)
    if tool_context:
        parts.append(tool_context)

    if has_substantive_context(chunks):
        parts.append(build_context(chunks))
    else:
        parts.append(NO_CONTEXT_SUFFIX.strip())

    if safety_rules:
        parts.append(f"{SAFETY_HEADER.strip()}\n{safety_rules}")

    return "\n\n".join(part for part in parts if part)


__all__ = [
    "MIN_CONTEXT_CHARS",
    "NO_CONTEXT_SUFFIX",
    "SAFETY_HEADER",
    "SYSTEM_PROMPT",
    "TOOL_RESULTS_HEADER",
    "build_context",
    "build_tool_context",
    "build_user_prompt",
    "has_substantive_context",
]
