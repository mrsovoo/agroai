"""Public model namespace for the knowledge domain.

The models themselves live in focused modules:

* :mod:`agroai.knowledge.metadata` - hashing, timestamps, trust tiers, the
  document metadata schema and front matter parsing;
* :mod:`agroai.knowledge.documents` - :class:`KnowledgeDocument`, chunks,
  chunking and ingestion reports;
* :mod:`agroai.knowledge.sources` - source records and citations;
* :mod:`agroai.knowledge.versions` - versioned knowledge (old + new are kept);
* :mod:`agroai.knowledge.candidates` - unverified material awaiting review;
* :mod:`agroai.knowledge.graph` - relationships declared by the metadata.

This module re-exports all of them so existing code and the tests can keep using
a single stable import path (``from agroai.knowledge.models import
KnowledgeDocument``).
"""

from __future__ import annotations

from agroai.knowledge.candidates import (
    CandidateAssessment,
    CandidatePipeline,
    CandidateStatus,
    EvidenceLevel,
    KnowledgeCandidate,
    approve,
    candidates_from_reports,
    guess_evidence_level,
)
from agroai.knowledge.documents import (
    Chunk,
    IngestionReport,
    KnowledgeDocument,
    RetrievedChunk,
    chunk_text,
    iter_chunks,
)
from agroai.knowledge.graph import (
    GraphEdge,
    GraphNode,
    KnowledgeGraph,
    build_graph,
    graph_from_database,
    related_context,
)
from agroai.knowledge.metadata import (
    DocumentMetadata,
    SOURCE_TYPE_TRUST,
    TrustLevel,
    content_hash,
    infer_trust_level,
    normalize_date,
    parse_front_matter,
    utcnow_iso,
)
from agroai.knowledge.sources import SourceCitation, SourceRecord, citation_for
from agroai.knowledge.versions import (
    DocumentVersion,
    MemoryVersionStore,
    SQLiteVersionStore,
    VersionStore,
    changes_between,
    version_history_summary,
    version_of,
)

__all__ = [
    "CandidateAssessment",
    "CandidatePipeline",
    "CandidateStatus",
    "Chunk",
    "DocumentMetadata",
    "DocumentVersion",
    "EvidenceLevel",
    "GraphEdge",
    "GraphNode",
    "IngestionReport",
    "KnowledgeCandidate",
    "KnowledgeDocument",
    "KnowledgeGraph",
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
    "candidates_from_reports",
    "changes_between",
    "chunk_text",
    "citation_for",
    "content_hash",
    "graph_from_database",
    "guess_evidence_level",
    "infer_trust_level",
    "iter_chunks",
    "normalize_date",
    "parse_front_matter",
    "related_context",
    "utcnow_iso",
    "version_history_summary",
    "version_of",
]
