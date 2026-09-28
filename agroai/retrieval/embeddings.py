"""Embedding provider abstraction.

Embeddings are an *optional* future capability. AgroAI Core never fabricates
vectors: when embeddings are not configured the provider raises
:class:`EmbeddingsNotConfigured`, and the retriever simply uses lexical search.

Future providers (local sentence-transformers, GGUF embedding models,
ONNX runtimes, ...) implement :class:`EmbeddingProvider`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

from agroai.config.settings import Settings, get_settings


class EmbeddingsNotConfigured(RuntimeError):
    """Raised when an embedding operation is attempted without a provider."""


@dataclass
class EmbeddingHealth:
    provider: str
    configured: bool
    available: bool
    dimensions: int
    detail: str


class EmbeddingProvider(ABC):
    """Interface for turning text into vectors."""

    name: str = "base"

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Vector dimensionality."""

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True only when a real embedding model is loaded."""

    @abstractmethod
    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of documents."""

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        vectors = self.embed([text])
        return vectors[0] if vectors else []

    @abstractmethod
    def health_check(self) -> EmbeddingHealth:
        """Report configuration and availability."""


class NullEmbeddingProvider(EmbeddingProvider):
    """Explicit "no embeddings configured" provider.

    It exists so the rest of the system can depend on a real object and report
    honestly, rather than guarding every call with ``if embeddings is not None``.
    """

    name = "none"

    def __init__(self, dimensions: int = 384) -> None:
        self._dimensions = dimensions

    @property
    def dimensions(self) -> int:
        return self._dimensions

    @property
    def is_configured(self) -> bool:
        return False

    def embed(self, texts: Sequence[str]) -> list[list[float]]:
        raise EmbeddingsNotConfigured(
            "No embedding provider is configured. Semantic/vector retrieval is "
            "available in a later phase; lexical retrieval works now."
        )

    def health_check(self) -> EmbeddingHealth:
        return EmbeddingHealth(
            provider=self.name,
            configured=False,
            available=False,
            dimensions=self._dimensions,
            detail=(
                "Embeddings not configured. Lexical retrieval (SQLite FTS) is "
                "active instead. Configure AGROAI_EMBEDDING_PROVIDER to enable "
                "vector search later."
            ),
        )


def get_embedding_provider(settings: Settings | None = None) -> EmbeddingProvider:
    """Return the configured embedding provider (only ``none`` exists now)."""
    settings = settings or get_settings()
    provider = (settings.embedding_provider or "none").lower()
    # No real provider yet for any value: return the explicit null provider so
    # behavior is honest and predictable.
    return NullEmbeddingProvider(dimensions=settings.embedding_dimensions)


__all__ = [
    "EmbeddingHealth",
    "EmbeddingProvider",
    "EmbeddingsNotConfigured",
    "NullEmbeddingProvider",
    "get_embedding_provider",
]
