"""CLI smoke tests: every fixed command must exit 0 with real output."""

from __future__ import annotations

import pytest

from agroai.cli.main import main


@pytest.fixture()
def cli_env(tmp_path, monkeypatch):
    monkeypatch.setenv("AGROAI_DATABASE_PATH", str(tmp_path / "cli.db"))
    monkeypatch.setenv("AGROAI_DATA_DIR", str(tmp_path / "data"))
    from agroai.config.settings import reset_settings_cache

    reset_settings_cache()
    yield
    reset_settings_cache()


def test_version(capsys, cli_env):
    assert main(["version"]) == 0
    assert "AgroAI" in capsys.readouterr().out


def test_help(capsys, cli_env):
    with pytest.raises(SystemExit) as excinfo:
        main(["--help"])
    assert excinfo.value.code == 0
    assert "agroai" in capsys.readouterr().out


def test_doctor(capsys, cli_env):
    assert main(["doctor"]) in {0, 1}
    output = capsys.readouterr().out
    assert "AgroAI doctor" in output


def test_plans_now_works(capsys, cli_env):
    """Regression: cmd_plans used to crash with AttributeError on args.user."""
    assert main(["plans"]) == 0
    output = capsys.readouterr().out
    assert "AgroAI plans" in output
    assert "free" in output


def test_plans_with_user(capsys, cli_env):
    assert main(["plans", "--user", "farmer"]) == 0
    assert "farmer" in capsys.readouterr().out


def test_models_now_works(capsys, cli_env):
    """Regression: circular import between datasets and fine_tuning."""
    assert main(["models"]) == 0
    output = capsys.readouterr().out
    # With no registered models the command still reports honestly.
    assert "No models registered" in output or "Model reyestri" in output


def test_datasets_now_works(capsys, cli_env):
    """Regression: same circular import."""
    assert main(["datasets"]) == 1  # honest exit code: no datasets yet
    assert "Datasetlar yo'q" in capsys.readouterr().out


def test_search_empty_db(capsys, cli_env):
    assert main(["search", "pomidor"]) == 1
    assert "topilmadi" in capsys.readouterr().out
