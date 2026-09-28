"""Tests for the CLI: every command works offline and exits honestly."""

from __future__ import annotations

from pathlib import Path

import pytest

from agroai.cli.main import main


@pytest.fixture()
def cli_env(tmp_path: Path, monkeypatch) -> Path:
    """Point AGROAI_HOME at a temp project root for CLI runs."""
    monkeypatch.setenv("AGROAI_HOME", str(tmp_path))
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "data").mkdir()
    return tmp_path


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_version_exits_zero(cli_env) -> None:
    assert main(["version"]) == 0


def test_help_lists_commands(capsys, cli_env) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--help"])
    assert exc_info.value.code == 0
    output = capsys.readouterr().out
    for command in (
        "ask",
        "plan",
        "search",
        "ingest",
        "ingest-dir",
        "sources",
        "tools",
        "doctor",
        "version",
    ):
        assert command in output


def test_doctor_runs_and_reports(cli_env, capsys) -> None:
    exit_code = main(["doctor"])
    output = capsys.readouterr().out
    assert "AgroAI doctor" in output
    assert "[OK]" in output or "[WARNING]" in output
    assert "[ERROR]" not in output  # nothing should hard-fail in a fresh install
    assert exit_code == 0


def test_search_empty_base_is_friendly(cli_env, capsys) -> None:
    exit_code = main(["search", "pomidor"])
    output = capsys.readouterr().out
    assert "topilmadi" in output
    assert exit_code == 1


def test_ingest_and_search_roundtrip(cli_env, capsys) -> None:
    doc = cli_env / "tomato.txt"
    doc.write_text(
        "Pomidor barglarida sariq dog'lar paydo bo'lsa, bu zamburug'li kasallik bo'lishi mumkin.",
        encoding="utf-8",
    )
    assert main(["ingest", str(doc)]) == 0
    output = capsys.readouterr().out
    assert "qo'shildi" in output

    assert main(["search", "pomidor sariq"]) == 0
    output = capsys.readouterr().out
    assert "pomidor" in output.lower()


def test_ingest_duplicate_reports_skip(cli_env, capsys) -> None:
    doc = _write(cli_env, "same.txt", "Bir xil hujjat matni.")
    assert main(["ingest", str(doc)]) == 0
    assert main(["ingest", str(doc)]) == 0
    output = capsys.readouterr().out
    assert "takror" in output


def test_ingest_unsupported_format_fails_cleanly(cli_env, capsys) -> None:
    doc = _write(cli_env, "report.pdf", "%PDF-1.4")
    assert main(["ingest", str(doc)]) == 1
    error = capsys.readouterr().err
    assert "not implemented" in error


def test_ingest_dir(cli_env, capsys) -> None:
    directory = cli_env / "knowledge" / "pests"
    directory.mkdir()
    (directory / "aphid.txt").write_text("Shira o'simlik shirasini so'radi.", encoding="utf-8")
    (directory / "notes.md").write_text("# Shira\n\nShira haqida ma'lumot.", encoding="utf-8")
    exit_code = main(["ingest-dir", str(cli_env / "knowledge")])
    assert exit_code == 0
    assert main(["search", "shira"]) == 0


def test_sources_lists_ingested(cli_env, capsys) -> None:
    doc = _write(cli_env, "fao_note.txt", "FAO tavsiyalari.")
    main(["ingest", str(doc)])
    capsys.readouterr()
    exit_code = main(["sources"])
    output = capsys.readouterr().out
    assert "fao_note.txt" in output
    assert exit_code == 0


def test_sources_empty_is_not_error_spam(cli_env, capsys) -> None:
    exit_code = main(["sources"])
    output = capsys.readouterr().out
    assert "yo'q" in output
    assert exit_code == 1


def test_plan_command_shows_the_plan(cli_env, capsys) -> None:
    exit_code = main(["plan", "Tuproq sho'rlanishi, EC 6.5 dS/m. Nima qilay?"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "Reja/plan:" in output
    assert "assess_safety" in output
    assert "retrieve_knowledge" in output
    assert "soil_tool" in output  # the EC value made a tool step unambiguous
    assert "Xavf/risk:" in output


def test_tools_command_lists_available_tools(cli_env, capsys) -> None:
    exit_code = main(["tools"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "calculator" in output
    assert "soil_tool" in output
    assert "veterinary_tool" in output


def test_sources_trusted_lists_the_registry(cli_env, capsys) -> None:
    exit_code = main(["sources", "--trusted"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "FAO" in output
    assert "WOAH" in output or "USDA" in output


def test_ask_without_llm_is_honest(cli_env, capsys) -> None:
    doc = _write(cli_env, "tomato.txt", "Pomidor barglaridagi sariq dog'lar zamburug'dan bo'lishi mumkin.")
    main(["ingest", str(doc)])
    capsys.readouterr()
    exit_code = main(["ask", "Pomidor barglarida sariq dog'lar paydo bo'ldi. Nima bo'lishi mumkin?"])
    output = capsys.readouterr().out
    assert exit_code == 0
    assert "not configured" in output or "sozlanmagan" in output
    assert "tomato.txt" in output  # retrieved sources are shown
