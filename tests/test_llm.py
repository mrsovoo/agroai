"""Tests for the LLM abstraction: honesty when unconfigured, health checks."""

from __future__ import annotations

import pytest

from agroai.config.settings import Settings
from agroai.models import LLMProvider, LocalLLMProvider, Message, ResponseStatus, Role, get_llm_provider
from agroai.models.base import NOT_CONFIGURED_MESSAGE, LLMResponse


def test_unconfigured_local_llm_reports_not_configured(settings) -> None:
    provider = get_llm_provider(settings)
    assert isinstance(provider, LLMProvider)
    assert provider.is_configured is False

    response = provider.generate("salom")
    assert response.status is ResponseStatus.NOT_CONFIGURED
    assert response.ok is False
    assert not response.configured

    response = provider.chat([Message(role=Role.USER, content="salom")])
    assert response.status is ResponseStatus.NOT_CONFIGURED


def test_unconfigured_llm_never_fakes_answers(settings) -> None:
    provider = get_llm_provider(settings)
    text = provider.generate("Pomidor kasalligi haqida gapirib ber").text
    # The message must explain configuration, not pretend to answer.
    assert "not configured" in text or "sozlanmagan" in text.lower() or NOT_CONFIGURED_MESSAGE == text


def test_health_check_unconfigured(settings) -> None:
    provider = LocalLLMProvider.from_settings(settings)
    health = provider.health_check()
    assert health.configured is False
    assert health.available is False
    assert "AGROAI_LLM_MODEL" in health.detail


def test_configured_model_path_but_missing_file_is_error(tmp_path) -> None:
    settings = Settings.from_env()
    settings.local_model_path = str(tmp_path / "missing.gguf")
    provider = LocalLLMProvider.from_settings(settings)
    assert provider.is_configured is True  # a model is declared
    health = provider.health_check()
    assert health.configured is True
    assert health.available is False
    assert "not found" in health.detail


def test_configured_model_without_engine_is_explicit(tmp_path) -> None:
    model_file = tmp_path / "model.gguf"
    model_file.write_bytes(b"fake")
    settings = Settings.from_env()
    settings.local_model_path = str(model_file)
    provider = LocalLLMProvider.from_settings(settings)
    health = provider.health_check()
    assert health.available is True
    assert "not implemented in this phase" in health.detail

    response = provider.generate("salom")
    assert response.status is ResponseStatus.ERROR
    assert "no inference engine" in response.text


def test_unknown_provider_falls_back_with_label(settings) -> None:
    settings.llm_provider = "cloudy-service"
    provider = get_llm_provider(settings)
    assert provider.is_configured is False
    assert "unsupported" in provider.name


def test_message_roundtrip() -> None:
    message = Message(role=Role.ASSISTANT, content="javob")
    assert message.to_dict() == {"role": "assistant", "content": "javob"}


def test_llm_response_ok_property() -> None:
    ok = LLMResponse(text="x", status=ResponseStatus.OK)
    bad = LLMResponse(text="x", status=ResponseStatus.ERROR)
    assert ok.ok and not bad.ok


def test_describe_reports_state(settings) -> None:
    provider = get_llm_provider(settings)
    described = provider.describe()
    assert described["configured"] is False
    assert described["model"] == "(not configured)"
