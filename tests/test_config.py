"""Tests for configuration resolution."""

from __future__ import annotations

from pathlib import Path

from agroai.config.settings import Settings, parse_env_file


def test_defaults_are_offline_safe(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    assert settings.llm_provider == "local"
    assert settings.llm_model == ""
    assert settings.embedding_provider == "none"
    assert settings.retrieval_backend == "sqlite"
    assert settings.database_path == (tmp_path / "data/database/agroai.db").resolve()
    assert settings.knowledge_dir == (tmp_path / "knowledge").resolve()
    assert settings.log_level == "INFO"
    assert settings.default_language == "uz"


def test_env_overrides(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGROAI_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("AGROAI_LLM_MODEL", "llama3.1:8b")
    monkeypatch.setenv("AGROAI_DATABASE_PATH", str(tmp_path / "custom.db"))
    monkeypatch.setenv("AGROAI_CHUNK_SIZE", "512")
    monkeypatch.setenv("AGROAI_DEDUPLICATE", "false")

    settings = Settings.from_env(tmp_path)
    assert settings.llm_provider == "ollama"
    assert settings.llm_model == "llama3.1:8b"
    assert settings.database_path == (tmp_path / "custom.db").resolve()
    assert settings.chunk_size == 512
    assert settings.deduplicate is False


def test_spec_alias_variables_resolve(tmp_path: Path, monkeypatch) -> None:
    """Spec variable names AGROAI_LOCAL_MODEL / AGROAI_MODEL_PATH /
    AGROAI_EMBEDDING_MODEL map onto the internal settings fields."""
    monkeypatch.setenv("AGROAI_LOCAL_MODEL", "llama3.1:8b")
    monkeypatch.setenv("AGROAI_MODEL_PATH", "/models/qwen.gguf")
    monkeypatch.setenv("AGROAI_EMBEDDING_MODEL", "/models/embedder")
    settings = Settings.from_env(tmp_path)
    assert settings.llm_model == "llama3.1:8b"
    assert settings.local_model_path == "/models/qwen.gguf"
    assert settings.embedding_model_path == "/models/embedder"


def test_alias_defaults_stay_empty(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    # Defaults stay empty.
    assert settings.llm_model == ""
    assert settings.local_model_path == ""
    assert settings.embedding_model_path == ""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "# comment\n"
        "AGROAI_LOG_LEVEL=debug\n"
        "export AGROAI_DEFAULT_LANGUAGE=ru\n"
        'AGROAI_LLM_MODEL="local-gguf"\n'
        "AGROAI_CHUNK_OVERLAP=100\n",
        encoding="utf-8",
    )
    settings = Settings.from_env(tmp_path)
    assert settings.log_level == "DEBUG"
    assert settings.default_language == "ru"
    assert settings.llm_model == "local-gguf"
    assert settings.chunk_overlap == 100


def test_parse_env_file_empty_and_missing(tmp_path: Path) -> None:
    assert parse_env_file(tmp_path / "missing.env") == {}
    env_file = tmp_path / "a.env"
    env_file.write_text("", encoding="utf-8")
    assert parse_env_file(env_file) == {}


def test_public_dict_has_no_secrets(settings: Settings) -> None:
    public = settings.as_public_dict()
    assert "llm_provider" in public


def test_model_and_dataset_paths_and_safety_defaults(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    assert settings.models_dir == (tmp_path / "models").resolve()
    assert settings.datasets_dir == (tmp_path / "datasets").resolve()
    assert settings.safety_strict is True
    assert settings.updater_provider == "none"
    assert settings.vector_store_backend == "none"


def test_new_env_overrides_resolve(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AGROAI_SAFETY_STRICT", "false")
    monkeypatch.setenv("AGROAI_MODELS_DIR", "gguf")
    monkeypatch.setenv("AGROAI_DATASETS_DIR", "sets")
    monkeypatch.setenv("AGROAI_UPDATER_PROVIDER", "crawler")
    monkeypatch.setenv("AGROAI_VECTOR_STORE_BACKEND", "memory")

    settings = Settings.from_env(tmp_path)
    assert settings.safety_strict is False
    assert settings.models_dir == (tmp_path / "gguf").resolve()
    assert settings.datasets_dir == (tmp_path / "sets").resolve()
    assert settings.updater_provider == "crawler"
    assert settings.vector_store_backend == "memory"


def test_ensure_directories_creates_the_roadmap_layout(tmp_path: Path) -> None:
    settings = Settings.from_env(tmp_path)
    settings.ensure_directories()
    for path in (
        settings.knowledge_dir,
        settings.data_dir,
        settings.documents_dir,
        settings.indexes_dir,
        settings.models_dir,
        settings.datasets_dir,
        settings.database_path.parent,
    ):
        assert path.is_dir()
