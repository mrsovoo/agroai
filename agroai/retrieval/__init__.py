"""Retrieval and embedding abstractions, plus the vector store interface."""

from agroai.retrieval.embeddings import (
    EmbeddingHealth,
    EmbeddingProvider,
    EmbeddingsNotConfigured,
    NullEmbeddingProvider,
    get_embedding_provider,
)
from agroai.retrieval.retriever import (
    RetrievalHealth,
    Retriever,
    SQLiteRetriever,
    VectorRetriever,
    get_retriever,
)
from agroai.retrieval.vector_store import (
    BACKEND_MEMORY,
    BACKEND_NONE,
    InMemoryVectorStore,
    NullVectorStore,
    VectorIndexer,
    VectorMatch,
    VectorRecord,
    VectorStore,
    VectorStoreHealth,
    VectorStoreNotConfigured,
    cosine_similarity,
    get_vector_store,
)

__all__ = [
    "BACKEND_MEMORY",
    "BACKEND_NONE",
    "EmbeddingHealth",
    "EmbeddingProvider",
    "EmbeddingsNotConfigured",
    "InMemoryVectorStore",
    "NullEmbeddingProvider",
    "NullVectorStore",
    "RetrievalHealth",
    "Retriever",
    "SQLiteRetriever",
    "VectorIndexer",
    "VectorMatch",
    "VectorRecord",
    "VectorRetriever",
    "VectorStore",
    "VectorStoreHealth",
    "VectorStoreNotConfigured",
    "cosine_similarity",
    "get_embedding_provider",
    "get_retriever",
    "get_vector_store",
]
