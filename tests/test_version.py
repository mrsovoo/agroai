"""Package-level sanity checks."""

from __future__ import annotations

import agroai


def test_version_is_defined() -> None:
    assert agroai.__version__ == "0.1.0"


def test_all_public_modules_importable() -> None:
    import importlib

    modules = [
        "agroai.config",
        "agroai.database",
        "agroai.knowledge",
        "agroai.ingestion",
        "agroai.models",
        "agroai.models.fine_tuning",
        "agroai.retrieval",
        "agroai.tools",
        "agroai.safety",
        "agroai.agent",
        "agroai.sources",
        "agroai.cli",
    ]
    for name in modules:
        module = importlib.import_module(name)
        assert module is not None


def test_split_knowledge_modules_importable() -> None:
    """The roadmap layout: metadata / documents / sources / manager."""
    import importlib

    for name in (
        "agroai.knowledge.metadata",
        "agroai.knowledge.documents",
        "agroai.knowledge.sources",
        "agroai.knowledge.manager",
        "agroai.ingestion.base",
        "agroai.ingestion.text",
        "agroai.ingestion.pdf",
        "agroai.ingestion.html",
        "agroai.ingestion.pipeline",
        "agroai.retrieval.vector_store",
        "agroai.models.base",
        "agroai.models.loader",
        "agroai.models.inference",
        "agroai.sources.registry",
        "agroai.sources.updater",
        "agroai.agent.planner",
        "agroai.agent.executor",
        "agroai.safety.guardrails",
        "agroai.safety.validation",
        "agroai.tools.base",
    ):
        assert importlib.import_module(name) is not None
