"""Shared pytest fixtures for the AgroAI test suite."""

from __future__ import annotations

from pathlib import Path

import pytest

from agroai.config.settings import Settings
from agroai.database.database import Database


@pytest.fixture()
def project_root(tmp_path: Path) -> Path:
    """A fake project root with a knowledge dir, isolated from the repo."""
    (tmp_path / "knowledge").mkdir()
    (tmp_path / "data").mkdir()
    return tmp_path


@pytest.fixture()
def settings(project_root: Path) -> Settings:
    """Settings bound to the temp project root."""
    return Settings.from_env(project_root)


@pytest.fixture()
def database(settings: Settings) -> Database:
    db = Database(settings.database_path)
    db.connect()
    db.initialize()
    yield db
    db.close()
