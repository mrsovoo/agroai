"""LLM provider abstraction.

The agent only ever talks to this interface, so swapping between the local
placeholder, Ollama, llama.cpp, GGUF models or any future engine is a matter of
providing another :class:`LLMProvider` implementation - no agent rewrite.

Nothing here fabricates AI output. When a provider is not configured its
methods return an explicitly-not-configured response.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

NOT_CONFIGURED_MESSAGE = (
    "Local language model is not configured yet. "
    "AgroAI Core is running in architecture-only mode and does not generate "
    "answers. Configure a local model (for example Ollama or a GGUF file) to "
    "enable generation. See README \"Local LLM architecture\"."
)


class ResponseStatus(str, Enum):
    """Explicit status so callers can never mistake a stub for real output."""

    OK = "ok"
    NOT_CONFIGURED = "not_configured"
    ERROR = "error"


class Role(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class Message:
    """A single chat message."""

    role: Role
    content: str

    def to_dict(self) -> dict[str, str]:
        return {"role": self.role.value, "content": self.content}


@dataclass
class LLMResponse:
    """Uniform response envelope returned by every provider."""

    text: str
    status: ResponseStatus = ResponseStatus.OK
    model: str = ""
    provider: str = ""
    finish_reason: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    raw: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status is ResponseStatus.OK

    @property
    def configured(self) -> bool:
        return self.status is not ResponseStatus.NOT_CONFIGURED


@dataclass
class HealthStatus:
    """Result of a provider health check."""

    provider: str
    configured: bool
    available: bool
    detail: str
    model: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "configured": self.configured,
            "available": self.available,
            "model": self.model,
            "detail": self.detail,
        }


class LLMProvider(ABC):
    """Interface every inference backend must implement."""

    #: Human readable provider identifier, e.g. ``local`` or ``ollama``.
    name: str = "base"

    @property
    @abstractmethod
    def model(self) -> str:
        """Return the active model identifier (may be empty)."""

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """Return True only when a real model is available for inference."""

    @abstractmethod
    def generate(self, prompt: str, **options: Any) -> LLMResponse:
        """Generate a completion for a raw prompt."""

    @abstractmethod
    def chat(self, messages: list[Message], **options: Any) -> LLMResponse:
        """Generate a reply from a chat history."""

    @abstractmethod
    def health_check(self) -> HealthStatus:
        """Report whether the provider is configured and reachable."""

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.name,
            "model": self.model or "(not configured)",
            "configured": self.is_configured,
        }

    # Convenience helpers -------------------------------------------------
    @staticmethod
    def not_configured(reason: str = NOT_CONFIGURED_MESSAGE, provider: str = "base") -> LLMResponse:
        return LLMResponse(
            text=reason,
            status=ResponseStatus.NOT_CONFIGURED,
            provider=provider,
        )
