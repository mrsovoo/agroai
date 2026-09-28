"""Tests for the trusted-source registry helpers and the knowledge updater."""

from __future__ import annotations

from agroai.config.settings import Settings
from agroai.sources import (
    TRUSTED_SOURCES,
    DisabledUpdater,
    KnowledgeUpdater,
    UpdateResult,
    get_source,
    get_updater,
)
from agroai.sources.updater import NOT_IMPLEMENTED


def test_get_source_matches_registry_conservatively() -> None:
    assert get_source("FAO") is not None
    assert get_source("fao report 2024").name.startswith("FAO")  # type: ignore[union-attr]
    assert get_source("USDA") is not None
    assert get_source("") is None
    assert get_source("random_blog.ru") is None
    assert get_source("afo") is None  # an acronym must be a whole word


def test_registry_entries_carry_a_type() -> None:
    types = {source.source_type for source in TRUSTED_SOURCES}
    assert types
    assert all(source.name for source in TRUSTED_SOURCES)


# --------------------------------------------------------------------- #
# Updater
# --------------------------------------------------------------------- #
def test_default_updater_is_disabled_and_honest(settings) -> None:
    updater = get_updater(settings)
    assert isinstance(updater, KnowledgeUpdater)
    assert isinstance(updater, DisabledUpdater)
    assert updater.is_enabled is False
    assert updater.candidate_sources() == list(TRUSTED_SOURCES)


def test_check_updates_reports_not_implemented_without_network(settings) -> None:
    result = get_updater(settings).check_updates()
    assert result.status == NOT_IMPLEMENTED
    assert result.ok is False
    assert result.network_used is False
    assert result.checked == 0
    assert result.sources
    assert "not implemented" in result.detail


def test_apply_updates_never_writes(settings) -> None:
    updater = get_updater(settings)
    dry = updater.apply_updates(dry_run=True)
    assert "dry run" in dry.detail
    assert dry.added == 0 and dry.updated == 0

    real = updater.apply_updates(dry_run=False)
    assert "Nothing was written" in real.detail
    assert real.network_used is False


def test_unknown_updater_provider_is_reported_not_faked() -> None:
    settings = Settings.from_env()
    settings.updater_provider = "crawler"
    updater = get_updater(settings)
    assert isinstance(updater, DisabledUpdater)
    result = updater.check_updates()
    assert "not implemented" in result.detail
    assert updater.is_enabled is False


def test_update_result_serializes() -> None:
    data = UpdateResult(status=NOT_IMPLEMENTED, detail="x", sources=["FAO"]).as_dict()
    assert data["status"] == NOT_IMPLEMENTED
    assert data["sources"] == ["FAO"]
    assert data["network_used"] is False
