"""Local inference: the engine contract and the Phase-1 provider.

Two layers are separated here on purpose:

* :class:`InferenceEngine` - the thing that actually runs a model (an Ollama
  HTTP call, a llama.cpp server, an in-process GGUF runtime). None is wired up
  yet, so :func:`get_inference_engine` returns an
  :class:`UnimplementedEngine` that raises a clear error instead of inventing an
  answer.
* :class:`LocalLLMProvider` - the :class:`~agroai.models.base.LLMProvider`
  implementation that the agent talks to. It answers questions about
  configuration, reports health honestly, and only ever produces text through an
  engine that really ran.

This split is what lets a later phase plug in a real engine without touching the
agent, the CLI or the safety layer.
"""

from __future__ import annotations

import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agroai.config.settings import Settings
from agroai.models.base import (
    NOT_CONFIGURED_MESSAGE,
    HealthStatus,
    LLMProvider,
    LLMResponse,
    Message,
    ResponseStatus,
    Role,
)
from agroai.models.loader import ModelNotFoundError, inspect_model_file

#: Phrase used when a model is declared but no runtime can execute it. Kept
#: stable because the CLI, tests and doctor all report it to the user.
NO_ENGINE_MESSAGE = (
    "A local model is configured but no inference engine is wired up in this "
    "phase (no inference engine available). Implement an InferenceEngine "
    "(Ollama, llama.cpp or a GGUF runtime) and register it in "
    "agroai.models.inference.get_inference_engine."
)


@dataclass
class EngineHealth:
    """Availability of an inference engine."""

    engine: str
    available: bool
    detail: str
    model: str = ""


class InferenceEngine(ABC):
    """Interface for anything that can turn messages into text."""

    name: str = "base"

    @abstractmethod
    def is_available(self) -> bool:
        """True only when the engine can actually run right now."""

    @abstractmethod
    def model_id(self) -> str:
        """Identifier of the model the engine would run."""

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **options: Any,
    ) -> str:
        """Generate text. Must raise if it cannot run - never return a guess."""

    def health(self) -> EngineHealth:
        return EngineHealth(
            engine=self.name,
            available=self.is_available(),
            detail="Engine health not detailed.",
            model=self.model_id(),
        )


class UnimplementedEngine(InferenceEngine):
    """The only engine that exists today: it refuses to hallucinate."""

    name = "unimplemented"

    def __init__(self, model: str = "", reason: str = "") -> None:
        self._model = model
        self._reason = reason or (
            "No inference engine is implemented yet. Implement InferenceEngine "
            "and return it from get_inference_engine()."
        )

    def is_available(self) -> bool:
        return False

    def model_id(self) -> str:
        return self._model

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 512,
        temperature: float = 0.2,
        **options: Any,
    ) -> str:
        raise NotImplementedError(self._reason)

    def health(self) -> EngineHealth:
        return EngineHealth(
            engine=self.name,
            available=False,
            detail=self._reason,
            model=self._model,
        )


def get_inference_engine(config: "LocalProviderConfig") -> InferenceEngine:
    """Return the inference engine for ``config``.

    Every local backend (``local``, ``ollama``, ``llamacpp``, ``gguf``) resolves
    to :class:`UnimplementedEngine` today. This function is the single
    registration point a real engine plugs into later.
    """
    declared = config.model or Path(config.model_path).name
    return UnimplementedEngine(model=declared)


@dataclass
class LocalProviderConfig:
    """Configuration handed to a local inference backend."""

    provider: str = "local"
    model: str = ""
    model_path: str = ""
    endpoint: str = "http://127.0.0.1:11434"
    max_tokens: int = 512
    temperature: float = 0.2

    @property
    def model_file_exists(self) -> bool:
        return bool(self.model_path) and Path(self.model_path).expanduser().is_file()


class LocalLLMProvider(LLMProvider):
    """Local model provider: honest about configuration, honest about limits."""

    name = "local"

    def __init__(
        self,
        config: LocalProviderConfig | None = None,
        engine: InferenceEngine | None = None,
    ) -> None:
        self._config = config or LocalProviderConfig()
        self._engine = engine or get_inference_engine(self._config)

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def from_settings(cls, settings: Settings) -> "LocalLLMProvider":
        return cls(
            LocalProviderConfig(
                provider=settings.llm_provider or "local",
                model=settings.llm_model,
                model_path=settings.local_model_path,
                endpoint=settings.llm_endpoint,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
            )
        )

    @property
    def config(self) -> LocalProviderConfig:
        return self._config

    @property
    def engine(self) -> InferenceEngine:
        return self._engine

    @property
    def model(self) -> str:
        if self._config.model:
            return self._config.model
        if self._config.model_path:
            return Path(self._config.model_path).name
        return ""

    @property
    def is_configured(self) -> bool:
        """A local model counts as configured only if a model name/path is set."""
        return bool(self._config.model or self._config.model_path)

    # ------------------------------------------------------------------ #
    # LLMProvider interface
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, **options: Any) -> LLMResponse:
        if not self.is_configured:
            return self.not_configured(NOT_CONFIGURED_MESSAGE, provider=self.name)
        return self._run([Message(role=Role.USER, content=prompt)], **options)

    def chat(self, messages: list[Message], **options: Any) -> LLMResponse:
        if not self.is_configured:
            return self.not_configured(NOT_CONFIGURED_MESSAGE, provider=self.name)
        return self._run(messages, **options)

    def _run(self, messages: list[Message], **options: Any) -> LLMResponse:
        max_tokens = int(options.pop("max_tokens", self._config.max_tokens))
        temperature = float(options.pop("temperature", self._config.temperature))
        try:
            text = self._engine.complete(
                messages, max_tokens=max_tokens, temperature=temperature, **options
            )
        except NotImplementedError as exc:
            return LLMResponse(
                text=NO_ENGINE_MESSAGE,
                status=ResponseStatus.ERROR,
                provider=self.name,
                model=self.model,
                finish_reason="engine_unavailable",
                raw={"engine": self._engine.name, "reason": str(exc)},
            )
        except Exception as exc:  # noqa: BLE001 - a broken engine must not crash the agent
            return LLMResponse(
                text=f"The local inference engine failed: {exc}",
                status=ResponseStatus.ERROR,
                provider=self.name,
                model=self.model,
                finish_reason="engine_error",
            )
        return LLMResponse(
            text=text,
            status=ResponseStatus.OK,
            provider=self.name,
            model=self.model,
            usage={"messages": len(messages)},
        )

    def health_check(self) -> HealthStatus:
        if not self.is_configured:
            return HealthStatus(
                provider=self.name,
                configured=False,
                available=False,
                model="",
                detail=(
                    "No local model configured. Set AGROAI_LLM_MODEL or "
                    "AGROAI_LOCAL_MODEL_PATH in .env (nothing is required for "
                    "the knowledge/retrieval features)."
                ),
            )

        if self._config.model_path and not self._config.model_file_exists:
            return HealthStatus(
                provider=self.name,
                configured=True,
                available=False,
                model=self.model,
                detail=f"Model file not found: {self._config.model_path}",
            )

        return HealthStatus(
            provider=self.name,
            configured=True,
            available=True,
            model=self.model,
            detail=(
                "Model reference present. Inference engine is not implemented in "
                "this phase (see future LLM phase)." + self._loader_detail()
            ),
        )

    def _loader_detail(self) -> str:
        """Describe the model file we really inspected (if there is one)."""
        if not self._config.model_path:
            return ""
        try:
            info = inspect_model_file(self._config.model_path)
        except ModelNotFoundError:
            return ""
        family = info.family or "unknown"
        return (
            f" Loader: format={info.format}, family={family}, "
            f"size={info.size_gb} GB."
        )

    def endpoint_reachable(self, timeout: float = 0.5) -> bool:
        """Best-effort check whether a local inference server is listening."""
        try:
            urllib.request.urlopen(self._config.endpoint, timeout=timeout)  # noqa: S310
            return True
        except urllib.error.URLError:
            return False
        except Exception:
            return False

    def describe(self) -> dict[str, Any]:
        described = super().describe()
        described["engine"] = self._engine.name
        described["engine_available"] = self._engine.is_available()
        return described


__all__ = [
    "NO_ENGINE_MESSAGE",
    "EngineHealth",
    "InferenceEngine",
    "LocalLLMProvider",
    "LocalProviderConfig",
    "UnimplementedEngine",
    "get_inference_engine",
]
