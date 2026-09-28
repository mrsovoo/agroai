"""Knowledge domain: document models, provenance metadata and the manager.

``KnowledgeManager`` is exported lazily (PEP 562) to avoid an import cycle with
:mod:`agroai.database`, which imports the document models at module load.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from agroai.knowledge.documents import (
    Chunk,
    IngestionReport,
    KnowledgeDocument,
    RetrievedChunk,
    chunk_text,
    iter_chunks,
)
from agroai.knowledge.metadata import (
    DocumentMetadata,
    SOURCE_TYPE_TRUST,
    TrustLevel,
    content_hash,
    infer_trust_level,
    normalize_date,
    utcnow_iso,
)
from agroai.knowledge.candidates import (
    CandidateAssessment,
    CandidatePipeline,
    CandidateStatus,
    EvidenceLevel,
    KnowledgeCandidate,
    approve,
    guess_evidence_level,
)
from agroai.knowledge.graph import KnowledgeGraph, build_graph, graph_from_database
from agroai.knowledge.sources import SourceCitation, SourceRecord, citation_for
from agroai.knowledge.versions import (
    DocumentVersion,
    MemoryVersionStore,
    SQLiteVersionStore,
    VersionStore,
    changes_between,
)

if TYPE_CHECKING:  # pragma: no cover
    from agroai.knowledge.manager import KnowledgeManager

__all__ = [
    "CandidateAssessment",
    "CandidatePipeline",
    "CandidateStatus",
    "Chunk",
    "DocumentMetadata",
    "DocumentVersion",
    "EvidenceLevel",
    "IngestionReport",
    "KnowledgeCandidate",
    "KnowledgeDocument",
    "KnowledgeGraph",
    "KnowledgeManager",
    "MemoryVersionStore",
    "RetrievedChunk",
    "SOURCE_TYPE_TRUST",
    "SQLiteVersionStore",
    "SourceCitation",
    "SourceRecord",
    "TrustLevel",
    "VersionStore",
    "approve",
    "build_graph",
    "changes_between",
    "chunk_text",
    "citation_for",
    "content_hash",
    "graph_from_database",
    "guess_evidence_level",
    "infer_trust_level",
    "iter_chunks",
    "normalize_date",
    "utcnow_iso",
]


def __getattr__(name: str) -> Any:  # PEP 562 lazy attribute access
    if name == "KnowledgeManager":  # pragma: no cover - exercised via other modules
        from agroai.knowledge.manager import KnowledgeManager as _KnowledgeManager

        return _KnowledgeManager
    raise AttributeError(f"module 'agroai.knowledge' has no attribute {name!r}")
