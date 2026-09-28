"""AgroAI Core - offline-first agricultural AI agent.

The package is organised as the roadmap's architecture, with each layer either
doing real work or refusing honestly:

* ``agroai.agent`` - planning, execution and RAG orchestration
* ``agroai.models`` - local LLM provider contract, model loading, fine-tuning
* ``agroai.knowledge`` - documents, provenance metadata, sources, manager
* ``agroai.ingestion`` - the staged ingestion pipeline (TXT/Markdown today)
* ``agroai.retrieval`` - lexical retrieval, embeddings and vector stores
* ``agroai.tools`` - deterministic agriculture calculators and classifiers
* ``agroai.safety`` - guardrails before an answer, validation after it
* ``agroai.sources`` - curated trusted-source registry and update interface
* ``agroai.database`` - SQLite storage with FTS5 full-text search
* ``agroai.config`` - environment-based configuration (no secrets)

Nothing here fabricates output: no fake LLM answers, no invented sources, no
faked embeddings, no pretend PDF extraction. See docs/ARCHITECTURE.md for the
full map, including which boundaries are intentionally not implemented yet.
"""

from __future__ import annotations

__version__ = "0.1.0"

__all__ = ["__version__"]
