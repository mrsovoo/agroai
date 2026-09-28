"""Tests for ``agroai doctor``: statuses, rendering, exit semantics."""

from __future__ import annotations

from agroai.cli.doctor import ERROR, NOT_CONFIGURED, OK, WARNING, run_doctor


def test_doctor_fresh_project_statuses(settings) -> None:
    report = run_doctor(settings)
    names = {check.name for check in report.checks}
    assert {
        "Python version",
        "Configuration",
        "Knowledge directory",
        "Database",
        "Retrieval",
        "Embeddings (none)",
    } <= names
    # LLM should be explicitly NOT CONFIGURED (honest), never ERROR, in a fresh env.
    llm = next(check for check in report.checks if check.name.startswith("LLM"))
    assert llm.status == NOT_CONFIGURED
    assert not report.has_error


def test_doctor_knowledge_dir_missing_is_warning(tmp_path) -> None:
    from agroai.config.settings import Settings

    settings = Settings.from_env(tmp_path)
    report = run_doctor(settings)
    knowledge = next(check for check in report.checks if check.name == "Knowledge directory")
    assert knowledge.status == WARNING


def test_doctor_with_ingested_data_is_ok(database, settings) -> None:
    from agroai.knowledge.manager import KnowledgeManager
    from pathlib import Path

    directory = settings.knowledge_dir / "soil"
    directory.mkdir(parents=True)
    (directory / "note.txt").write_text("Tuproq tahlili bahorda qilinadi.", encoding="utf-8")
    KnowledgeManager(database, settings).ingest_file(Path(directory / "note.txt"))

    report = run_doctor(settings, database=database)
    db_check = next(check for check in report.checks if check.name == "Database")
    retrieval = next(check for check in report.checks if check.name == "Retrieval")
    assert db_check.status == OK
    assert retrieval.status == OK


def test_doctor_render_includes_summary(settings) -> None:
    report = run_doctor(settings)
    rendered = report.render()
    assert "Summary:" in rendered
    for status in (OK, WARNING, NOT_CONFIGURED, ERROR):
        assert status in rendered
