"""``agroai`` command line interface.

Commands:
    agroai                        interactive session
    agroai ask "question"         retrieval-grounded Q&A (plan + RAG)
    agroai plan "question"        show the safety/retrieval/tool plan only
    agroai search "query"         search the local knowledge base
    agroai ingest <file>          ingest one document
    agroai ingest-dir <dir>       ingest a directory tree
    agroai verify                 verify that ingested documents are complete
    agroai structure <file>       show detected chapters/sections
    agroai sources                list ingested sources (--trusted: show registry)
    agroai skills [--for "q"]     list skills / show routing for a question
    agroai candidates             list knowledge candidates (--review to decide)
    agroai tools                  list available tools
    agroai eval [--suite smoke]   run an evaluation suite (+ --baseline gate)
    agroai datasets               list versioned datasets
    agroai models                 model registry, datasets and training runs
    agroai vision <file>          inspect media / analyse it with a description
    agroai plans                  plans, limits and current usage
    agroai api                    route table and service health
    agroai doctor                 environment self-check
    agroai version                print the version

Offline-first: every command works without network access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from agroai import __version__
from agroai.agent.agent import Agent, AgentResult
from agroai.config.settings import Settings, get_settings, reset_settings_cache
from agroai.database.database import Database
from agroai.knowledge.manager import KnowledgeManager
from agroai.models import get_llm_provider
from agroai.retrieval.retriever import get_retriever
from agroai.safety.guardrails import SafetyGuard
from agroai.skills.router import get_skill_router
from agroai.sources.registry import TRUSTED_SOURCES
from agroai.tools.base import get_default_registry


# ---------------------------------------------------------------------- #
# Composition helpers (also used by tests)
# ---------------------------------------------------------------------- #
def build_database(settings: Settings | None = None) -> Database:
    settings = settings or get_settings()
    settings.ensure_directories()
    database = Database(settings.database_path)
    database.connect()
    database.initialize()
    return database


def build_knowledge_manager(
    database: Database | None = None, settings: Settings | None = None
) -> KnowledgeManager:
    settings = settings or get_settings()
    database = database or build_database(settings)
    return KnowledgeManager(database, settings)


def build_agent(
    database: Database | None = None, settings: Settings | None = None
) -> Agent:
    settings = settings or get_settings()
    database = database or build_database(settings)
    retriever = get_retriever(database, settings)
    llm = get_llm_provider(settings)
    return Agent(
        retriever=retriever,
        llm=llm,
        top_k=5,
        guard=SafetyGuard(strict=settings.safety_strict),
        tools=get_default_registry(),
        skills=get_skill_router(),
    )


# ---------------------------------------------------------------------- #
# Display helpers
# ---------------------------------------------------------------------- #
def _print_sources(result: AgentResult) -> None:
    """Print the sources behind an answer, if any."""
    if not result.sources:
        return
    print()
    print("Manbalar / Sources:")
    for index, retrieved in enumerate(result.sources, start=1):
        print(f"  [{index}] {retrieved.source_line}")


def _print_notes(result: AgentResult) -> None:
    """Print safety and validation notes for an answer."""
    if result.safety_blocked:
        print()
        print("DIQQAT / ATTENTION: javob manba bilan tasdiqlanmadi.")
    if result.validation is not None and not result.validation.ok:
        print()
        print(result.validation.render())
    for note in result.disclaimers:
        print(f"  ! {note}")


def cmd_ask(args: argparse.Namespace, settings: Settings) -> int:
    database = build_database(settings)
    try:
        agent = build_agent(database, settings)
        result = agent.ask(args.question)
        agent._last_result = result  # noqa: SLF001 - CLI display helper
        print(result.answer)
        if result.status == "not_configured":
            _print_sources(result)
            _print_notes(result)
            print(
                "\nMaslahat: 'agroai doctor' bilan tekshiring. LLM ni sozlash uchun "
                ".env faylida AGROAI_LLM_MODEL yoki AGROAI_LOCAL_MODEL_PATH belgilang."
            )
        elif result.status == "ok":
            if result.grounded:
                _print_sources(result)
            _print_notes(result)
        else:
            _print_notes(result)
        return 0
    finally:
        database.close()


def cmd_plan(args: argparse.Namespace, settings: Settings) -> int:
    """Show the plan for a question without generating an answer."""
    database = build_database(settings)
    try:
        agent = build_agent(database, settings)
        plan = agent.plan(args.question)
        print(plan.render())
        print()
        print(f"Vositalar/tools available: {len(agent.tools)}")
        return 0
    finally:
        database.close()


def cmd_search(args: argparse.Namespace, settings: Settings) -> int:
    database = build_database(settings)
    try:
        retriever = get_retriever(database, settings)
        results = retriever.search(args.query, top_k=args.top_k)
        if not results:
            print("Hech narsa topilmadi / No matching knowledge found.")
            print(
                f"Bilimlar bazasi: {database.count_documents()} hujjat, "
                f"{database.count_chunks()} bo'lak. Ingest qilish uchun: "
                "'agroai ingest <file>'."
            )
            return 1
        for index, retrieved in enumerate(results, start=1):
            snippet = retrieved.chunk.content.strip().replace("\n", " ")
            if len(snippet) > 220:
                snippet = snippet[:217] + "..."
            print(f"[{index}] score={retrieved.score:.3g}  {retrieved.source_line}")
            print(f"    {snippet}")
            print()
        return 0
    finally:
        database.close()


def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    database = build_database(settings)
    try:
        manager = build_knowledge_manager(database, settings)
        report = manager.ingest_file(Path(args.path), overwrite=args.overwrite)
        _print_ingestion_report(report)
        return 0 if report.documents_ingested or report.documents_skipped else 1
    finally:
        database.close()


def cmd_ingest_dir(args: argparse.Namespace, settings: Settings) -> int:
    database = build_database(settings)
    try:
        manager = build_knowledge_manager(database, settings)
        report = manager.ingest_directory(
            Path(args.directory), recursive=not args.no_recursive, overwrite=args.overwrite
        )
        _print_ingestion_report(report)
        return 0 if report.documents_failed == 0 else 1
    finally:
        database.close()


def _print_ingestion_report(report) -> None:
    print(
        "Ingestion: {seen} ko'rildi, {ingested} qo'shildi, {skipped} takror "
        "(o'tkazib yuborildi), {failed} xato; {chunks} bo'lak yaratildi.".format(
            seen=report.documents_seen,
            ingested=report.documents_ingested,
            skipped=report.documents_skipped,
            failed=report.documents_failed,
            chunks=report.chunks_created,
        )
    )
    for error in report.errors:
        print(f"  XATO / ERROR: {error}", file=sys.stderr)


def cmd_sources(args: argparse.Namespace, settings: Settings) -> int:
    if getattr(args, "trusted", False):
        return _print_trusted_sources()

    database = build_database(settings)
    try:
        rows = database.list_sources()
        if not rows:
            print("Hozircha manbalar yo'q / No sources ingested yet.")
            return 1
        print(f"Manbalar / Sources ({len(rows)}):")
        for row in rows:
            print(f"  - {row.name}  (turi: {row.source_type}, hujjatlar: {row.document_count})")
            if row.url:
                print(f"      {row.url}")
            if row.last_ingested_at:
                print(f"      oxirgi ingest: {row.last_ingested_at}")
        return 0
    finally:
        database.close()


def _print_trusted_sources() -> int:
    print(
        f"Ishonchli manbalar ro'yxati / Trusted source registry "
        f"({len(TRUSTED_SOURCES)} entries, metadata only - nothing is downloaded):"
    )
    for index, source in enumerate(TRUSTED_SOURCES, start=1):
        print(f"  {index}. {source.name}  [{source.source_type}]")
        if source.url:
            print(f"      {source.url}")
        if source.notes:
            print(f"      {source.notes}")
    return 0


def cmd_verify(args: argparse.Namespace, settings: Settings) -> int:
    """Verify that ingested documents are chunked, indexed and fully covered."""
    from agroai.ingestion.verification import verify_ingestion

    database = build_database(settings)
    try:
        if database.count_documents() == 0:
            print("Bilimlar bazasi bo'sh / The knowledge base is empty.")
            print("Avval hujjat qo'shing: 'agroai ingest <file>'.")
            return 1
        batch = verify_ingestion(
            database=database, settings=settings, limit=getattr(args, "limit", None)
        )
        print(batch.render(limit=getattr(args, "show", None)))
        return 0 if batch.ok else 1
    finally:
        database.close()


def cmd_structure(args: argparse.Namespace, settings: Settings) -> int:
    """Show the chapters/sections detected in a file (no ingestion)."""
    from agroai.ingestion.structure import detect_sections

    path = Path(args.path)
    if not path.is_file():
        print(f"Fayl topilmadi / File not found: {path}", file=sys.stderr)
        return 1
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as exc:
        print(f"Faylni o'qib bo'lmadi / Cannot read {path}: {exc}", file=sys.stderr)
        print(
            "Eslatma: PDF uchun avval matnga aylantiring "
            "('agroai ingest' bilan tekshiring)."
        )
        return 1
    structure = detect_sections(raw)
    print(f"Bo'limlar / sections: {len(structure)}")
    if structure.kind_counts:
        print("  " + ", ".join(f"{key}={value}" for key, value in sorted(structure.kind_counts.items())))
    print(structure.outline(limit=args.limit) or "(none detected)")
    return 0


def cmd_skills(args: argparse.Namespace, settings: Settings) -> int:
    """List skills, or show the routing decision for one question."""
    router = get_skill_router()
    registry = router.registry
    question = getattr(args, "for_question", "")
    if question:
        result = router.route(question)
        print(f"Savol / question: {question}")
        print(result.render())
        if result.matches:
            for match in result.matches:
                skill = match.skill
                print(f"\n{skill.name} - {skill.title}")
                print(f"  {skill.description}")
                if skill.categories:
                    print(f"  knowledge: {', '.join(skill.categories)}")
                if skill.tools:
                    print(f"  tools: {', '.join(skill.tools)}")
                if skill.guidance:
                    print(f"  guidance: {skill.guidance}")
        return 0

    print(f"AgroAI skillari / skills ({len(registry)}):")
    print(registry.describe())
    requirements = registry.requirements()
    print()
    print(f"Knowledge categories used: {', '.join(requirements['categories'])}")
    print(f"Tools referenced: {', '.join(requirements['tools'])}")
    return 0


def cmd_candidates(args: argparse.Namespace, settings: Settings) -> int:
    """List knowledge candidates and (optionally) record a review decision."""
    database = build_database(settings)
    try:
        manager = build_knowledge_manager(database, settings)
        review_id = getattr(args, "review", "")
        if review_id:
            candidate = database.get_candidate(review_id)
            if candidate is None:
                print(f"Nomzod topilmadi / Candidate not found: {review_id}", file=sys.stderr)
                return 1
            if not (args.approve or args.reject):
                print("Qaror kerak: --approve yoki --reject bering.")
                return 2
            reviewed, report = manager.review_candidate(
                candidate,
                approved=bool(args.approve),
                reviewer=args.reviewer or "cli",
                notes=args.notes or "",
            )
            status = getattr(getattr(reviewed, "status", None), "value", reviewed.status)
            print(f"{review_id}: {status}")
            if args.approve:
                print(
                    f"  qo'shildi: {report.documents_ingested} hujjat, "
                    f"{report.chunks_created} bo'lak"
                )
            return 0

        status = getattr(args, "status", "") or None
        found = manager.list_candidates(status=status, limit=args.limit)
        if not found:
            print("Nomzodlar yo'q / No knowledge candidates.")
            print(
                "Nomzodlar foydalanuvchi xabarlari va kuzatuvlardan yig'iladi; "
                "hech biri avtomatik bilimga aylanmaydi."
            )
            return 0
        print(f"Nomzodlar / candidates ({len(found)}):")
        for candidate in found:
            evidence = getattr(
                getattr(candidate, "evidence_level", None), "value", candidate.evidence_level
            )
            candidate_status = getattr(
                getattr(candidate, "status", None), "value", candidate.status
            )
            print(
                f"  - {candidate.id} [{candidate_status}] evidence={evidence} "
                f"confidence={candidate.confidence:.2f}"
            )
            print(f"      {candidate.statement[:160]}")
            if getattr(candidate, "conflicts", None):
                print(f"      ! conflicts: {', '.join(candidate.conflicts[:3])}")
        return 0
    finally:
        database.close()


def cmd_eval(args: argparse.Namespace, settings: Settings) -> int:
    """Run an evaluation suite and optionally gate against a baseline report."""
    from agroai.evaluation import (
        EvaluationReport,
        EvaluationTarget,
        EvaluationRunner,
        EvalSuite,
        get_suite,
        regression_gate,
    )

    try:
        suite = get_suite(args.suite) if not args.file else EvalSuite.from_jsonl(Path(args.file))
    except (KeyError, FileNotFoundError, ValueError) as exc:
        print(f"Suite yuklanmadi / Cannot load suite: {exc}", file=sys.stderr)
        return 2

    database = build_database(settings)
    try:
        target = EvaluationTarget(
            retriever=get_retriever(database, settings),
            guard=SafetyGuard(strict=settings.safety_strict),
            router=get_skill_router(),
            agent=build_agent(database, settings) if args.with_agent else None,
        )
        report = EvaluationRunner(target, top_k=args.top_k).run(suite)
        print(report.render())
        if args.save:
            saved = report.save(Path(args.save))
            print(f"\nSaqlandi / saved: {saved}")
        if args.baseline:
            try:
                baseline = EvaluationReport.load(Path(args.baseline))
            except (FileNotFoundError, ValueError) as exc:
                print(f"Baseline o'qilmadi / Cannot read baseline: {exc}", file=sys.stderr)
                return 2
            verdict = regression_gate(baseline, report, tolerance=args.tolerance)
            print()
            print(verdict.render())
            return 0 if verdict.passed else 1
        return 0 if report.passed else 1
    finally:
        database.close()


def cmd_datasets(args: argparse.Namespace, settings: Settings) -> int:
    """List versioned datasets and their splits."""
    from agroai.datasets import DatasetManager

    manager = DatasetManager(settings.datasets_dir)
    if args.create:
        return _create_dataset(manager, args, settings)
    versions = manager.list_versions(args.name or None)
    if not versions:
        print("Datasetlar yo'q / No datasets yet.")
        print(
            "Dataset yaratish: 'agroai datasets --create agroai-sft --version v1' "
            "(bilim bazasidagi hujjatlardan manbaga bog'langan yozuvlar eksport qilinadi)."
        )
        return 1
    for item in versions:
        print(item.render())
        print()
    stats = manager.stats()
    print(
        f"Jami / total: {stats['datasets']} dataset(s), {stats['versions']} version(s), "
        f"{stats['records']} record(s)"
    )
    return 0


def _create_dataset(manager, args: argparse.Namespace, settings: Settings) -> int:
    """Build a dataset version from the knowledge base (extractive only)."""
    from agroai.models.fine_tuning.dataset import grounded_records

    database = build_database(settings)
    try:
        documents = list(database.iter_documents())
        if not documents:
            print("Bilimlar bazasi bo'sh / Knowledge base is empty.", file=sys.stderr)
            return 1
        records = list(grounded_records(documents, min_chars=args.min_chars))
        try:
            dataset = manager.create_version(
                args.create,
                args.version,
                records,
                description=args.description or "Grounded passages exported from the knowledge base.",
                sources=[document.id for document in documents],
                metadata={"documents": len(documents), "generator": "grounded_records"},
                overwrite=args.overwrite,
            )
        except (FileExistsError, ValueError) as exc:
            print(f"Dataset yozilmadi / Dataset not written: {exc}", file=sys.stderr)
            return 1
        print(dataset.render())
        print()
        print(
            "Eslatma: bu faqat manbaga bog'langan (extractive) yozuvlar. "
            "Savol/javob sintezi generator modelni talab qiladi."
        )
        return 0 if dataset.is_trainable else 1
    finally:
        database.close()


def cmd_models(args: argparse.Namespace, settings: Settings) -> int:
    """Show the model registry, dataset stats and training-run status."""
    from agroai.datasets import DatasetManager
    from agroai.models.fine_tuning.manager import FineTuningManager
    from agroai.models.registry import ModelRegistry

    registry = ModelRegistry(settings.models_dir)
    datasets = DatasetManager(settings.datasets_dir)
    training = FineTuningManager(settings.models_dir, dataset_manager=datasets, registry=registry)

    records = registry.list_models(stage=args.stage or None, name=args.name or None)
    if records:
        print(f"Model reyestri / registry ({len(records)}):")
        for record in records:
            print(record.render())
            print()
    else:
        print("Ro'yxatdan o'tgan model yo'q / No models registered yet.")

    training_stats = training.stats()
    print(
        f"Training runs: {training_stats['runs']} "
        f"({', '.join(f'{k}={v}' for k, v in sorted(training_stats['by_status'].items())) or 'none'})"
    )
    requirements = training.requirements()
    print(f"Frameworkdan tayyor: {requirements['implemented']}")
    for item in requirements["needs"]:
        print(f"  - {item}")
    print()
    print(
        "Eslatma: bu build hech qanday modelni tayyorlamaydi (training runtime yo'q); "
        "reyestr faqat haqiqiy faktlarni qayd etadi."
    )
    return 0


def cmd_vision(args: argparse.Namespace, settings: Settings) -> int:
    """Inspect an image/video and (optionally) analyse it with a description."""
    from agroai.vision.analysis import MediaAnalyzer
    from agroai.vision.engines import get_vision_engine

    engine = get_vision_engine(settings)
    if args.info:
        print(f"Vision engine: {engine.name}")
        for key, value in engine.health_check().items():
            print(f"  {key}: {value}")
        return 0
    if not args.path:
        print("Fayl kerak: 'agroai vision <file>' yoki 'agroai vision --info'.", file=sys.stderr)
        return 2

    database = build_database(settings)
    try:
        analyzer = MediaAnalyzer(
            engine=engine, retriever=get_retriever(database, settings)
        )
        analysis = analyzer.analyze(args.path, question=args.question or "", kind=args.kind)
        print(analysis.render())
        return 0 if analysis.media is not None else 1
    finally:
        database.close()


def cmd_plans(args: argparse.Namespace, settings: Settings) -> int:
    """Show plans, limits and this machine's usage counters."""
    from agroai.api.quotas import QuotaPolicy, UsageStore

    quotas = QuotaPolicy(
        store=UsageStore(settings.data_dir / "api" / "usage.json"),
        default_plan=settings.default_plan,
        enforce=settings.enforce_quotas,
    )
    print(quotas.describe())
    user = args.user or "local"
    print()
    usage = quotas.usage(user)
    print(f"Foydalanish / usage for '{user}' ({usage['day']}):")
    for action, item in sorted(usage["actions"].items()):
        limit = "unlimited" if item["remaining"] is None else item["limit"]
        print(f"  {action}: used={item['used']}, limit={limit}")
    if not settings.api_key_map():
        print()
        print(
            "Diqqat: AGROAI_API_KEYS sozlanmagan - bu lokal rejim. "
            "Production uchun kalitlar va require_api_key ni yoqing."
        )
    return 0


def cmd_api(args: argparse.Namespace, settings: Settings) -> int:
    """Print the API route table and a live health summary."""
    from agroai.api import ApiRequest, build_service, describe_routes

    if args.routes:
        print("AgroAI API routes:")
        for route in describe_routes():
            action = route["quota_action"] or "-"
            print(f"  {route['method']:<5} {route['path']:<38} quota={action}")
            print(f"        {route['description']}")
        return 0

    service = build_service(settings)
    try:
        response = service.handle(ApiRequest("GET", "/api/v1/health"))
        print(f"HTTP {response.status}")
        payload = response.body
        print(f"status: {payload.get('status')}")
        for key in ("retrieval", "llm", "vision", "external_search", "quotas"):
            value = payload.get(key)
            if isinstance(value, dict):
                detail = value.get("detail") or value.get("search_mode") or ""
                print(f"  {key}: {detail or value}")
        if payload.get("detail"):
            print(f"  detail: {payload['detail']}")
        return 0
    finally:
        service.close()


def cmd_tools(settings: Settings) -> int:
    registry = get_default_registry()
    print(f"AgroAI vositalari / tools ({len(registry)}):")
    print(registry.describe())
    print()
    print(
        "Eslatma: hisoblash vositalari faqat siz bergan raqamlar bilan ishlaydi; "
        "ob-havo va dori ma'lumotlari manbasiz berilmaydi. / Note: computational "
        "tools only use the numbers you supply; weather and drug data are never "
        "invented."
    )
    return 0


def cmd_doctor(settings: Settings) -> int:
    from agroai.cli.doctor import run_doctor  # local import: fast CLI startup

    report = run_doctor(settings)
    print(report.render())
    return 1 if report.has_error else 0


def cmd_version() -> int:
    print(f"AgroAI {__version__}")
    return 0


# ---------------------------------------------------------------------- #
# Interactive mode
# ---------------------------------------------------------------------- #
HELP_TEXT = """\
Buyruqlar / Commands:
  /help        bu yordam / this help
  /sources     oxirgi javob manbalari / sources behind the last answer
  /plan        oxirgi savol rejasi / plan for the last question
  /tools       mavjud vositalar / list available tools
  /skills      savol uchun skillar / skills for the last question
  /clear       suhbat xotirasini tozalash / clear conversation memory
  /doctor      muhitni tekshirish / run environment self-check
  /exit        chiqish / quit (yoki Ctrl+D / Ctrl+C)

Har qanday boshqa matn - savol sifatida yuboriladi (reja: xavfsizlik + qidiruv +
vositalar + javob + validatsiya).
"""


def interactive_loop(settings: Settings) -> int:
    database = build_database(settings)
    try:
        agent = build_agent(database, settings)
        print(f"AgroAI {__version__} - qishloq xo'jaligi va veterinariya yordamchisi.")
        print("Yozing yoki /help ni bosing. Chiqish: /exit")
        while True:
            try:
                line = input("agroai> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                return 0
            if not line:
                continue
            if line in {"/exit", "/quit", "/q"}:
                return 0
            if line == "/help":
                print(HELP_TEXT)
                continue
            if line == "/clear":
                agent.memory.clear()
                print("Xotira tozalandi / Memory cleared.")
                continue
            if line == "/doctor":
                from agroai.cli.doctor import run_doctor

                print(run_doctor(settings).render())
                continue
            if line == "/tools":
                print(agent.tools.describe())
                continue
            if line == "/skills":
                result = getattr(agent, "_last_result", None)
                question = getattr(result, "question", "") if result is not None else ""
                if not question:
                    print("Hozircha savol yo'q / No question yet.")
                else:
                    print(agent.skills.route(question).render())
                continue
            if line == "/plan":
                result = getattr(agent, "_last_result", None)
                if result is None or result.plan is None:
                    print("Hozircha reja yo'q / No plan yet.")
                else:
                    print(result.plan.render())
                continue
            if line == "/sources":
                result = getattr(agent, "_last_result", None)
                if result is None or not result.sources:
                    print("Hozircha manbalar yo'q / No sources yet.")
                else:
                    for index, retrieved in enumerate(result.sources, start=1):
                        print(f"  [{index}] {retrieved.source_line}")
                continue
            if line.startswith("/"):
                print(f"Noma'lum buyruq / Unknown command: {line} (/help)")
                continue
            result = agent.ask(line, use_memory=True)
            agent._last_result = result  # noqa: SLF001
            print(result.answer)
            if result.status == "not_configured":
                _print_sources(result)
            _print_notes(result)
            print()
    finally:
        database.close()


# ---------------------------------------------------------------------- #
# Argument parsing
# ---------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agroai",
        description=(
            "AgroAI - offline-first agricultural and veterinary AI assistant "
            "(local knowledge retrieval + local LLM)."
        ),
    )
    parser.add_argument("--version", action="version", version=f"AgroAI {__version__}")
    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="ask a question (plan + RAG over local knowledge)")
    ask.add_argument("question", help="your question, in quotes")

    plan = sub.add_parser("plan", help="show the plan for a question (no answer)")
    plan.add_argument("question", help="your question, in quotes")

    search = sub.add_parser("search", help="search the local knowledge base")
    search.add_argument("query", help="search query, in quotes")
    search.add_argument("--top-k", type=int, default=5, help="maximum results (default 5)")

    ingest = sub.add_parser("ingest", help="ingest one document (TXT/Markdown)")
    ingest.add_argument("path", help="path to the file")
    ingest.add_argument("--overwrite", action="store_true", help="re-ingest even if duplicate")

    ingest_dir = sub.add_parser("ingest-dir", help="ingest all supported files in a directory")
    ingest_dir.add_argument("directory", help="directory to ingest")
    ingest_dir.add_argument("--no-recursive", action="store_true", help="do not descend")
    ingest_dir.add_argument("--overwrite", action="store_true", help="re-ingest duplicates")

    sources = sub.add_parser("sources", help="list ingested knowledge sources")
    sources.add_argument(
        "--trusted",
        action="store_true",
        help="list the curated trusted-source registry instead (offline metadata)",
    )

    verify = sub.add_parser("verify", help="verify ingested documents (chunks, index, coverage)")
    verify.add_argument("--limit", type=int, default=None, help="check at most N documents")
    verify.add_argument("--show", type=int, default=25, help="print at most N document lines")

    structure = sub.add_parser("structure", help="show detected chapters/sections of a text file")
    structure.add_argument("path", help="path to a text or markdown file")
    structure.add_argument("--limit", type=int, default=40, help="maximum sections to print")

    skills = sub.add_parser("skills", help="list skills, or show routing for a question")
    skills.add_argument(
        "--for",
        dest="for_question",
        default="",
        metavar="QUESTION",
        help="show which skills and retrieval scope this question would use",
    )

    candidates = sub.add_parser("candidates", help="list knowledge candidates / record a review")
    candidates.add_argument("--status", default="", help="filter: pending / verified / rejected")
    candidates.add_argument("--limit", type=int, default=25, help="maximum candidates to list")
    candidates.add_argument("--review", default="", metavar="ID", help="review a candidate by id")
    candidates.add_argument("--approve", action="store_true", help="approve (adds it to the knowledge base)")
    candidates.add_argument("--reject", action="store_true", help="reject the candidate")
    candidates.add_argument("--reviewer", default="", help="reviewer name for the audit trail")
    candidates.add_argument("--notes", default="", help="review notes")

    evals = sub.add_parser("eval", help="run an evaluation suite")
    evals.add_argument("--suite", default="smoke", help="built-in suite name (default: smoke)")
    evals.add_argument("--file", default="", help="load a suite from a JSONL file instead")
    evals.add_argument("--top-k", type=int, default=5, help="retrieval depth for retrieval cases")
    evals.add_argument("--save", default="", help="save the report JSON to this path")
    evals.add_argument("--baseline", default="", help="compare against a saved report (regression gate)")
    evals.add_argument("--tolerance", type=float, default=0.02, help="allowed metric drop (default 0.02)")
    evals.add_argument("--with-agent", action="store_true", help="include answer cases (needs an LLM)")

    datasets = sub.add_parser("datasets", help="list or create versioned training datasets")
    datasets.add_argument("--name", default="", help="only show versions of this dataset")
    datasets.add_argument("--create", default="", metavar="NAME", help="create a dataset version")
    datasets.add_argument("--version", default="v1", help="version to create (default: v1)")
    datasets.add_argument("--description", default="", help="manifest description")
    datasets.add_argument("--min-chars", type=int, default=200, help="minimum passage length")
    datasets.add_argument("--overwrite", action="store_true", help="replace an existing version")

    models = sub.add_parser("models", help="model registry, training runs and requirements")
    models.add_argument("--name", default="", help="filter by model name")
    models.add_argument("--stage", default="", help="filter by lifecycle stage")

    vision = sub.add_parser("vision", help="inspect/analyse an image or video")
    vision.add_argument("path", nargs="?", default="", help="path to the media file")
    vision.add_argument("--question", default="", help="what the farmer described")
    vision.add_argument("--kind", default="auto", choices=["auto", "image", "video"])
    vision.add_argument("--info", action="store_true", help="show vision engine health instead")

    plans = sub.add_parser("plans", help="show plans, limits and usage counters")
    plans.add_argument(
        "--user",
        default="",
        metavar="USER_ID",
        help="show usage for this user id (default: local)",
    )
    api = sub.add_parser("api", help="API route table and live service health")
    api.add_argument("--routes", action="store_true", help="print the route table instead")

    sub.add_parser("tools", help="list the available tools")
    sub.add_parser("doctor", help="check the local environment")
    sub.add_parser("version", help="print the AgroAI version")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    reset_settings_cache()
    settings = get_settings()

    if args.command is None:
        return interactive_loop(settings)
    if args.command == "ask":
        return cmd_ask(args, settings)
    if args.command == "plan":
        return cmd_plan(args, settings)
    if args.command == "search":
        return cmd_search(args, settings)
    if args.command == "ingest":
        return cmd_ingest(args, settings)
    if args.command == "ingest-dir":
        return cmd_ingest_dir(args, settings)
    if args.command == "sources":
        return cmd_sources(args, settings)
    if args.command == "verify":
        return cmd_verify(args, settings)
    if args.command == "structure":
        return cmd_structure(args, settings)
    if args.command == "skills":
        return cmd_skills(args, settings)
    if args.command == "candidates":
        return cmd_candidates(args, settings)
    if args.command == "eval":
        return cmd_eval(args, settings)
    if args.command == "datasets":
        return cmd_datasets(args, settings)
    if args.command == "models":
        return cmd_models(args, settings)
    if args.command == "vision":
        return cmd_vision(args, settings)
    if args.command == "plans":
        return cmd_plans(args, settings)
    if args.command == "api":
        return cmd_api(args, settings)
    if args.command == "tools":
        return cmd_tools(settings)
    if args.command == "doctor":
        return cmd_doctor(settings)
    if args.command == "version":
        return cmd_version()
    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
