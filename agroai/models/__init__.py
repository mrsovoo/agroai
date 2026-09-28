"""The model layer: provider contract, loader, inference and fine-tuning.

* :mod:`agroai.models.base` - the :class:`LLMProvider` contract the agent uses.
* :mod:`agroai.models.loader` - base-model families and real file inspection.
* :mod:`agroai.models.inference` - the local provider and its engine boundary.
* :mod:`agroai.models.fine_tuning` - dataset schema, LoRA configuration and the
  training-run manager.
* :mod:`agroai.models.registry` - which model version came from which base
  model, dataset, config and evaluation (and whether it may be released).

Only the abstraction is implemented end to end: nothing here generates text
until a real inference engine is wired into
:func:`agroai.models.inference.get_inference_engine`.
"""

from __future__ import annotations

from agroai.config.settings import Settings, get_settings
from agroai.models.base import (
    NOT_CONFIGURED_MESSAGE,
    HealthStatus,
    LLMProvider,
    LLMResponse,
    Message,
    ResponseStatus,
    Role,
)
from agroai.models.inference import (
    EngineHealth,
    InferenceEngine,
    LocalLLMProvider,
    LocalProviderConfig,
    UnimplementedEngine,
    get_inference_engine,
)
from agroai.models.loader import (
    MODEL_FAMILIES,
    ModelFamily,
    ModelFile,
    ModelNotFoundError,
    detect_model_format,
    find_model_files,
    get_family,
    guess_model_family,
    inspect_model_file,
)
from agroai.models.registry import (
    MODEL_KINDS,
    MODEL_KIND_EMBEDDING,
    MODEL_KIND_LLM,
    MODEL_KIND_VISION,
    REGISTRY_FILE,
    TRANSITIONS,
    ModelRecord,
    ModelRegistry,
    ModelStage,
    RegistryError,
)

__all__ = [
    "EngineHealth",
    "HealthStatus",
    "InferenceEngine",
    "LLMProvider",
    "LLMResponse",
    "LocalLLMProvider",
    "LocalProviderConfig",
    "MODEL_FAMILIES",
    "MODEL_KINDS",
    "MODEL_KIND_EMBEDDING",
    "MODEL_KIND_LLM",
    "MODEL_KIND_VISION",
    "Message",
    "ModelFamily",
    "ModelFile",
    "ModelNotFoundError",
    "ModelRecord",
    "ModelRegistry",
    "ModelStage",
    "NOT_CONFIGURED_MESSAGE",
    "REGISTRY_FILE",
    "RegistryError",
    "ResponseStatus",
    "TRANSITIONS",
    "Role",
    "UnimplementedEngine",
    "detect_model_format",
    "find_model_files",
    "get_family",
    "get_inference_engine",
    "get_llm_provider",
    "guess_model_family",
    "inspect_model_file",
]


def get_llm_provider(settings: Settings | None = None) -> LLMProvider:
    """Return the configured LLM provider.

    Only the local provider exists. Requesting another engine (``ollama``,
    ``llamacpp``, ``gguf``) still returns it, because those are the engines the
    local provider will host - and it reports honestly that no engine is wired
    up yet. Unknown provider names are labelled as unsupported rather than
    silently treated as local.
    """
    settings = settings or get_settings()
    name = (settings.llm_provider or "local").lower()

    provider = LocalLLMProvider.from_settings(settings)
    if name in {"local", "ollama", "llamacpp", "llama.cpp", "gguf"}:
        return provider

    provider.name = f"local(unsupported:'{name}')"
    return provider
