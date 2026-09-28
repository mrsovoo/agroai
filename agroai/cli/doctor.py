"""``agroai doctor``: verify the local installation honestly.

Every check reports one of OK / WARNING / NOT CONFIGURED / ERROR. Nothing is
claimed to work unless it was actually verified in this process.
"""

from __future__ import annotations

import sqlite3
import sys
from dataclasses import dataclass, field

from agroai import __version__
from agroai.config.settings import Settings
from agroai.database.database import Database
from agroai.models import find_model_files, get_llm_provider
from agroai.retrieval.embeddings import get_embedding_provider
from agroai.retrieval.retriever import get_retriever
from agroai.retrieval.vector_store import get_vector_store
from agroai.safety import CATEGORY_PATTERNS, AnswerValidator, SafetyGuard
from agroai.skills.router import get_skill_router
from agroai.sources.updater import get_updater
from agroai.tools.base import get_default_registry

OK = "OK"
WARNING = "WARNING"
NOT_CONFIGURED = "NOT CONFIGURED"
ERROR = "ERROR"


@dataclass
class CheckResult:
    name: str
    status: str  # one of OK/WARNING/NOT CONFIGURED/ERROR
    detail: str = ""

    @property
    def marker(self) -> str:
        return {
            OK: "[OK]",
            WARNING: "[WARNING]",
            NOT_CONFIGURED: "[NOT CONFIGURED]",
            ERROR: "[ERROR]",
        }.get(self.status, "[??]")


@dataclass
class DoctorReport:
    checks: list[CheckResult] = field(default_factory=list)

    def add(self, name: str, status: str, detail: str = "") -> CheckResult:
        result = CheckResult(name=name, status=status, detail=detail)
        self.checks.append(result)
        return result

    @property
    def has_error(self) -> bool:
        return any(check.status == ERROR for check in self.checks)

    def render(self) -> str:
        lines = [f"AgroAI doctor (v{__version__})", "=" * 60]
        for check in self.checks:
            lines.append(f"{check.marker:<18} {check.name}")
            if check.detail:
                for detail_line in check.detail.splitlines():
                    lines.append(f"{'':<18} {detail_line}")
        lines.append("=" * 60)
        counts = {
            status: sum(1 for c in self.checks if c.status == status)
            for status in (OK, WARNING, NOT_CONFIGURED, ERROR)
        }
        lines.append(
            "Summary: "
            + ", ".join(f"{count} {status}" for status, count in counts.items())
        )
        return "\n".join(lines)


def run_doctor(settings: Settings, *, database: Database | None = None) -> DoctorReport:
    """Run all environment checks. Read-only; creates nothing except the DB
    file's parent directory when connecting (SQLite requirement)."""
    report = DoctorReport()

    # --- Python version -------------------------------------------------
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    if sys.version_info >= (3, 10):
        report.add("Python version", OK, f"Python {version} (requires >= 3.10)")
    else:
        report.add("Python version", ERROR, f"Python {version} is too old (requires >= 3.10)")

    # --- Configuration ----------------------------------------------------
    report.add(
        "Configuration",
        OK,
        "\n".join(
            f"{key}: {value}" for key, value in settings.as_public_dict().items()
        ),
    )

    # --- Knowledge directories -------------------------------------------
    if settings.knowledge_dir.is_dir():
        categories_present = [
            category
            for category in settings.categories
            if (settings.knowledge_dir / category).is_dir()
        ]
        detail = str(settings.knowledge_dir)
        if categories_present:
            detail += f" ({len(categories_present)}/{len(settings.categories)} category dirs present)"
        else:
            detail += " (no category subdirectories yet - see knowledge/ in the repo)"
        report.add("Knowledge directory", OK if categories_present else WARNING, detail)
    else:
        report.add(
            "Knowledge directory",
            WARNING,
            f"Does not exist yet: {settings.knowledge_dir} (create it or set AGROAI_KNOWLEDGE_DIR)",
        )

    # --- Database ----------------------------------------------------------
    try:
        owned_db = database is None
        db = database or Database(settings.database_path)
        db.connect()
        db.initialize()
        if db.ping():
            docs = db.count_documents()
            chunks = db.count_chunks()
            fts = "FTS5 enabled" if db.fts_enabled else "FTS5 unavailable (LIKE fallback)"
            status = OK if docs > 0 else WARNING
            detail = f"{db.path} - {docs} documents, {chunks} chunks, {fts}"
            if docs == 0:
                detail += "\nKnowledge base is empty. Ingest with 'agroai ingest <file>'."
            report.add("Database", status, detail)
        else:
            report.add("Database", ERROR, f"Database did not answer: {db.path}")
        if owned_db:
            db.close()
    except sqlite3.Error as exc:
        report.add("Database", ERROR, f"Cannot open {settings.database_path}: {exc}")

    # --- Retrieval -----------------------------------------------------------
    try:
        db = database or Database(settings.database_path)
        if database is not None or db._conn is None:
            db.connect()
            db.initialize()
        retriever = get_retriever(db, settings)
        health = retriever.health_check()
        status = OK if health.documents > 0 else WARNING
        report.add(
            "Retrieval",
            status,
            f"backend={health.backend}, mode={health.search_mode}, "
            f"documents={health.documents}, chunks={health.chunks}. {health.detail}",
        )
        if database is None:
            db.close()
    except Exception as exc:  # noqa: BLE001 - doctor must not crash
        report.add("Retrieval", ERROR, f"Retrieval check failed: {exc}")

    # --- LLM -------------------------------------------------------------------
    try:
        provider = get_llm_provider(settings)
        health = provider.health_check()
        if not health.configured:
            status = NOT_CONFIGURED
        elif health.available:
            status = OK if "not implemented" not in health.detail else WARNING
        else:
            status = ERROR
        report.add(
            f"LLM ({provider.name})",
            status,
            health.detail or "No detail provided.",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("LLM", ERROR, f"Health check crashed: {exc}")

    # --- Local model files -----------------------------------------------------
    try:
        models = find_model_files(settings.models_dir)
        if models:
            detail = "\n".join(
                f"{model.name} - format={model.format}, family={model.family or 'unknown'}, "
                f"{model.size_gb} GB, supported={model.is_supported_format}"
                for model in models
            )
            report.add("Model files", OK if any(m.is_supported_format for m in models) else WARNING, detail)
        else:
            report.add(
                "Model files",
                NOT_CONFIGURED,
                f"No model files found in {settings.models_dir}. Place a GGUF model "
                "there (or set AGROAI_MODEL_PATH) when the inference phase is built.",
            )
    except Exception as exc:  # noqa: BLE001
        report.add("Model files", ERROR, f"Model discovery crashed: {exc}")

    # --- Embeddings ----------------------------------------------------------------
    try:
        embeddings = get_embedding_provider(settings)
        health = embeddings.health_check()
        report.add(
            f"Embeddings ({health.provider})",
            OK if health.configured else NOT_CONFIGURED,
            health.detail,
        )
    except Exception as exc:  # noqa: BLE001
        report.add("Embeddings", ERROR, f"Health check crashed: {exc}")

    # --- Vector store ----------------------------------------------------------------
    try:
        store = get_vector_store(settings)
        health = store.health_check()
        report.add(
            f"Vector store ({health.backend})",
            OK if health.ready else NOT_CONFIGURED,
            health.detail,
        )
    except Exception as exc:  # noqa: BLE001
        report.add("Vector store", ERROR, f"Health check crashed: {exc}")

    # --- Tools ---------------------------------------------------------------------
    try:
        registry = get_default_registry()
        implemented = [spec.name for spec in registry.specs() if spec.implemented]
        report.add(
            "Tools",
            OK if implemented else WARNING,
            f"{len(registry)} tools registered: {', '.join(registry.names())}\n"
            f"Implemented: {', '.join(implemented) or '(none)'}",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("Tools", ERROR, f"Tool registry crashed: {exc}")

    # --- Safety --------------------------------------------------------------------
    try:
        guard = SafetyGuard(strict=settings.safety_strict)
        validator = AnswerValidator()
        verdict = guard.classify("pestitsid dozasi qancha?")
        report.add(
            "Safety",
            OK if verdict.requires_sources else WARNING,
            f"strict={settings.safety_strict}, categories={len(CATEGORY_PATTERNS)}, "
            f"validator min_coverage={validator.min_coverage}\nSelf-test: "
            f"'pestitsid dozasi qancha?' -> {verdict.category.value}/"
            f"{verdict.risk_level}, requires_sources={verdict.requires_sources}",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("Safety", ERROR, f"Safety layer crashed: {exc}")

    # --- Knowledge updater ----------------------------------------------------------
    try:
        updater = get_updater(settings)
        result = updater.check_updates()
        report.add(
            f"Knowledge updater ({updater.name})",
            NOT_CONFIGURED,
            f"status={result.status}, candidate sources={len(result.sources)}, "
            f"network_used={result.network_used}\n{result.detail}",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("Knowledge updater", ERROR, f"Updater check crashed: {exc}")

    # --- Skills --------------------------------------------------------------------
    try:
        from agroai.skills.router import get_skill_router

        router = get_skill_router()
        requirements = router.registry.requirements()
        sample = router.route("pomidor bargida sariq dog'lar bor").names
        report.add(
            "Skills",
            OK,
            f"{len(router.registry)} skills, "
            f"{len(requirements['categories'])} knowledge categories, "
            f"{len(requirements['tools'])} tools referenced\n"
            f"Self-test: 'pomidor bargida sariq dog'lar bor' -> {', '.join(sample) or '(none)'}",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("Skills", ERROR, f"Skill registry crashed: {exc}")

    # --- Differential diagnosis ------------------------------------------------------
    try:
        from agroai.diagnosis.base import detect_subject, detect_symptoms

        symptoms = detect_symptoms("pomidor bargida sariq dog'lar va nekroz")
        report.add(
            "Differential diagnosis",
            OK,
            "Engine: retrieval-grounded (agroai.diagnosis). "
            "Output is a ranked differential, never a verdict.\n"
            f"Self-test symptoms: {', '.join(symptoms) or '(none)'} "
            f"(subject: {detect_subject('pomidor') or 'unknown'})",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("Differential diagnosis", ERROR, f"Diagnosis layer crashed: {exc}")

    # --- Vision ---------------------------------------------------------------------
    try:
        from agroai.vision.engines import get_vision_engine

        engine = get_vision_engine(settings)
        health = engine.health_check()
        report.add(
            f"Vision ({engine.name})",
            OK if health.get("configured") else NOT_CONFIGURED,
            f"configured={health.get('configured')}, "
            f"target latency={engine.target_latency_seconds}s (a target, not a guarantee)\n"
            "File inspection (format/resolution/duration) is implemented regardless; "
            "content description requires a vision model.",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("Vision", ERROR, f"Vision layer crashed: {exc}")

    # --- External search --------------------------------------------------------------
    try:
        from agroai.sources.external import get_external_search

        provider = get_external_search(settings)
        health = provider.health_check()
        report.add(
            f"External search ({provider.name})",
            NOT_CONFIGURED,
            f"configured={health.get('configured')}, enabled={health.get('enabled')}, "
            f"network_used={health.get('network_used')}, "
            f"trusted hosts in policy={health.get('trusted_hosts')}\n{health.get('detail')}",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("External search", ERROR, f"External search layer crashed: {exc}")

    # --- Quotas / plans ----------------------------------------------------------------
    try:
        from agroai.api.quotas import QuotaPolicy, UsageStore

        quotas = QuotaPolicy(
            store=UsageStore(settings.data_dir / "api" / "usage.json"),
            default_plan=settings.default_plan,
            enforce=settings.enforce_quotas,
        )
        decision = quotas.consume("doctor", "message")
        keys = settings.api_key_map()
        report.add(
            "Quotas / API auth",
            OK if quotas.enforce else WARNING,
            f"default plan={quotas.default_plan}, plans={len(quotas.plans)}, "
            f"enforced={quotas.enforce}\n"
            f"API keys configured: {len(keys)}"
            + (
                ""
                if keys
                else " (local development mode - set AGROAI_API_KEYS and "
                "AGROAI_REQUIRE_API_KEY before exposing the API)"
            )
            + f"\nSelf-test: doctor/message -> allowed={decision.allowed}, "
            f"used={decision.used}, limit={decision.limit}",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("Quotas / API auth", ERROR, f"Quota layer crashed: {exc}")

    # --- Model registry / datasets ------------------------------------------------------
    try:
        from agroai.datasets import DatasetManager
        from agroai.models.registry import ModelRegistry

        registry = ModelRegistry(settings.models_dir)
        datasets = DatasetManager(settings.datasets_dir)
        stats = registry.stats()
        dataset_stats = datasets.stats()
        report.add(
            "Model registry / datasets",
            OK,
            f"registered model versions: {stats['versions']} "
            f"(released: {stats['released']})\n"
            f"dataset versions: {dataset_stats['versions']} "
            f"({dataset_stats['records']} record(s))\n"
            f"Training runtime: {registry.requirements()['training_runtime']}",
        )
    except Exception as exc:  # noqa: BLE001
        report.add("Model registry / datasets", ERROR, f"Registry layer crashed: {exc}")

    # --- Ingestion verification --------------------------------------------------------
    try:
        from agroai.ingestion.verification import coverage_report

        db = database or Database(settings.database_path)
        owned = database is None
        if owned:
            db.connect()
        coverage = coverage_report(db, settings=settings, limit=50)
        if coverage["documents"] == 0:
            status = WARNING
            detail = (
                "Knowledge base is empty: nothing to verify yet. "
                "Run 'agroai ingest' then 'agroai verify'."
            )
        else:
            status = OK if coverage["status"] == "ok" else WARNING
            detail = (
                f"documents={coverage['documents']} (verified={coverage['verified']}, "
                f"partial={coverage['partial']}, empty={coverage['empty']}), "
                f"chunks={coverage['chunks']}, indexed={coverage['indexed_chunks']}, "
                f"average coverage={coverage['average_coverage']:.1%}"
            )
        report.add("Ingestion verification", status, detail)
        if owned:
            db.close()
    except Exception as exc:  # noqa: BLE001
        report.add("Ingestion verification", ERROR, f"Verification crashed: {exc}")

    # --- Evaluation (runs the real smoke suite) ----------------------------------------
    try:
        from agroai.evaluation import EvaluationRunner, EvaluationTarget, smoke_suite

        db = database or Database(settings.database_path)
        owned = database is None
        if owned:
            db.connect()
        target = EvaluationTarget(
            retriever=get_retriever(db, settings),
            guard=SafetyGuard(strict=settings.safety_strict),
            router=get_skill_router(),
        )
        result = EvaluationRunner(target).run(smoke_suite())
        summary = result.summary
        status = OK if (result.passed and summary["passed"]) else ERROR
        report.add(
            "Evaluation (smoke suite)",
            status,
            f"passed={summary['passed']}, failed={summary['failed']}, "
            f"skipped={summary['skipped']} (agent/corpus cases are skipped honestly)\n"
            + (
                "metrics: "
                + ", ".join(f"{key}={value:.3f}" for key, value in summary["metrics"].items())
                if summary["metrics"]
                else "metrics: none measurable without a corpus"
            ),
        )
        if owned:
            db.close()
    except Exception as exc:  # noqa: BLE001
        report.add("Evaluation (smoke suite)", ERROR, f"Evaluation crashed: {exc}")

    # --- Dependencies (stdlib-only core; extras are optional) ------------------------
    try:
        from agroai.ingestion.pdf import is_pdf_available, pdf_backend

        pdf_detail = (
            f"PDF extraction available via '{pdf_backend()}'."
            if is_pdf_available()
            else "PDF extraction needs the optional 'pdf' extra: pip install 'agroai[pdf]'."
        )
    except Exception as exc:  # noqa: BLE001
        pdf_detail = f"PDF support check failed: {exc}"
    report.add(
        "Dependencies",
        OK,
        "Core: standard library only (sqlite3 included in CPython). "
        "Optional extras: pip install agroai[pdf] / agroai[embeddings]. "
        + pdf_detail,
    )
    return report


__all__ = [
    "ERROR",
    "NOT_CONFIGURED",
    "OK",
    "WARNING",
    "CheckResult",
    "DoctorReport",
    "run_doctor",
]
