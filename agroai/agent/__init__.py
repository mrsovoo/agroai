"""Agent package: planning, execution, RAG orchestration, prompts and memory.

* :mod:`agroai.agent.planner` - question -> explicit plan (safety, retrieval,
  tools, generation, validation);
* :mod:`agroai.agent.executor` - runs a plan step by step;
* :mod:`agroai.agent.agent` - presents the result with sources and caveats;
* :mod:`agroai.agent.memory` - conversation memory (short-term, in-process).
"""

from __future__ import annotations

from agroai.agent.agent import Agent, AgentResult
from agroai.agent.executor import ExecutionResult, PlanExecutor, StepOutcome
from agroai.agent.memory import Memory, ShortTermMemory
from agroai.agent.planner import (
    ACTION_GENERATE,
    ACTION_RETRIEVE,
    ACTION_SAFETY,
    ACTION_TOOL,
    ACTION_VALIDATE,
    Plan,
    PlanStep,
    Planner,
    RuleBasedPlanner,
    build_planner,
)
from agroai.agent.prompts import (
    SYSTEM_PROMPT,
    build_context,
    build_tool_context,
    build_user_prompt,
    has_substantive_context,
)

__all__ = [
    "ACTION_GENERATE",
    "ACTION_RETRIEVE",
    "ACTION_SAFETY",
    "ACTION_TOOL",
    "ACTION_VALIDATE",
    "Agent",
    "AgentResult",
    "ExecutionResult",
    "Memory",
    "Plan",
    "PlanExecutor",
    "PlanStep",
    "Planner",
    "RuleBasedPlanner",
    "SYSTEM_PROMPT",
    "ShortTermMemory",
    "StepOutcome",
    "build_context",
    "build_planner",
    "build_tool_context",
    "build_user_prompt",
    "has_substantive_context",
]
