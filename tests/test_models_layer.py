"""Tests for the model layer: loader inspection, engines, fine-tuning boundary."""

from __future__ import annotations

from pathlib import Path

import pytest

from agroai.models import (
    MODEL_FAMILIES,
    LocalLLMProvider,
    UnimplementedEngine,
    detect_model_format,
    find_model_files,
    get_family,
    get_inference_engine,
    get_llm_provider,
    guess_model_family,
    inspect_model_file,
)
from agroai.models.base import ResponseStatus
from agroai.models.fine_tuning import (
    DatasetRecord,
    FineTuningDataset,
    LoRAConfig,
    LoRATrainer,
    build_instruction_dataset,
    grounded_records,
    read_jsonl,
    summarize,
    training_requirements,
    validate_records,
    write_jsonl,
)
from agroai.models.inference import LocalProviderConfig
from agroai.models.loader import ModelNotFoundError


def _gguf(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"GGUF" + b"\x03\x00\x00\x00" + b"\x00" * 32)
    return path


# --------------------------------------------------------------------- #
# Loader
# --------------------------------------------------------------------- #
def test_model_families_are_described() -> None:
    keys = {family.key for family in MODEL_FAMILIES}
    assert {"qwen", "llama", "gemma"} <= keys
    assert get_family("Qwen") is not None
    assert get_family("does-not-exist") is None


@pytest.mark.parametrize(
    ("filename", "expected"),
    [
        ("qwen2.5-7b-instruct-q4_k_m.gguf", "qwen"),
        ("Meta-Llama-3.1-8B.gguf", "llama"),
        ("gemma-2-2b-it.gguf", "gemma"),
        ("mixtral-8x7b.gguf", "mistral"),
        ("phi-3-mini.gguf", "phi"),
        ("mystery-model.gguf", None),
    ],
)
def test_guess_model_family(filename: str, expected: str | None) -> None:
    assert guess_model_family(filename) == expected


def test_detect_model_format_reads_bytes(tmp_path: Path) -> None:
    assert detect_model_format(_gguf(tmp_path / "m.gguf")) == "gguf"
    assert detect_model_format(tmp_path / "bytes.bin") == "unknown"
    (tmp_path / "bytes.bin").write_bytes(b"not a model")
    assert detect_model_format(tmp_path / "bytes.bin") == "unknown"

    header = (0).to_bytes(8, "little") + b'{"x": 1}'
    (tmp_path / "m.safetensors").write_bytes(header)
    assert detect_model_format(tmp_path / "m.safetensors") == "safetensors"


def test_inspect_model_file_reports_real_facts(tmp_path: Path) -> None:
    path = _gguf(tmp_path / "models" / "qwen2.5-7b-q4.gguf")
    info = inspect_model_file(path)
    assert info.format == "gguf"
    assert info.family == "qwen"
    assert info.size_bytes == path.stat().st_size
    assert info.is_supported_format is True
    assert info.to_dict()["family"] == "qwen"

    with pytest.raises(ModelNotFoundError):
        inspect_model_file(tmp_path / "missing.gguf")


def test_find_model_files_ignores_non_models(tmp_path: Path) -> None:
    _gguf(tmp_path / "models" / "qwen.gguf")
    (tmp_path / "models" / "notes.txt").write_text("hello", encoding="utf-8")
    found = find_model_files(tmp_path / "models")
    assert [model.name for model in found] == ["qwen.gguf"]
    assert find_model_files(tmp_path / "no-such-dir") == []


# --------------------------------------------------------------------- #
# Engine boundary + provider
# --------------------------------------------------------------------- #
def test_engine_boundary_is_explicit() -> None:
    engine = get_inference_engine(LocalProviderConfig(model="llama3.1:8b"))
    assert isinstance(engine, UnimplementedEngine)
    assert engine.is_available() is False
    assert engine.model_id() == "llama3.1:8b"
    assert "No inference engine" in engine.health().detail


def test_configured_provider_without_engine_never_fakes_text(settings, tmp_path: Path) -> None:
    settings.local_model_path = str(_gguf(tmp_path / "qwen.gguf"))
    provider = LocalLLMProvider.from_settings(settings)
    assert provider.is_configured is True
    assert provider.describe()["engine"] == "unimplemented"

    response = provider.generate("salom")
    assert response.status is ResponseStatus.ERROR
    assert response.raw["engine"] == "unimplemented"

    health = provider.health_check()
    assert "format=gguf" in health.detail
    assert "family=qwen" in health.detail


def test_unknown_provider_is_labelled(settings) -> None:
    settings.llm_provider = "some-cloud"
    assert "unsupported" in get_llm_provider(settings).name


# --------------------------------------------------------------------- #
# Fine-tuning: dataset
# --------------------------------------------------------------------- #
def _record(**overrides) -> DatasetRecord:
    data = {
        "instruction": "Manbaga asoslanib javob bering.",
        "output": "Tuproq sho'rlanishi ECe bilan o'lchanadi.",
        "source": "fao_note.txt",
    }
    data.update(overrides)
    return DatasetRecord(**data)


def test_dataset_record_requires_attribution() -> None:
    assert _record().is_valid() is True
    assert _record(source="").problems() == ["missing source (records must stay attributable)"]
    with pytest.raises(ValueError, match="instruction"):
        DatasetRecord.from_dict({"instruction": "", "output": "x", "source": "s"})
    with pytest.raises(ValueError, match="source"):
        DatasetRecord.from_dict({"instruction": "i", "output": "o"})
    # Unknown keys are ignored, not fatal.
    assert DatasetRecord.from_dict({**_record().to_dict(), "bogus": 1}).instruction


def test_jsonl_roundtrip_and_error_reporting(tmp_path: Path) -> None:
    path = tmp_path / "agro.jsonl"
    assert write_jsonl([_record(), _record(output="boshqa")], path) == 2
    records = read_jsonl(path)
    assert len(records) == 2
    assert records[1].output == "boshqa"

    assert validate_records(records) == []
    summary = summarize(records)
    assert summary["records"] == 2
    assert summary["invalid"] == 0

    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "missing.jsonl")

    broken = tmp_path / "broken.jsonl"
    broken.write_text('{"instruction": "i", "output": "o"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="invalid dataset record"):
        read_jsonl(broken)


def test_grounded_records_are_extractive_only(tmp_path: Path) -> None:
    from agroai.knowledge.documents import KnowledgeDocument

    short = KnowledgeDocument(id="short", title="S", content="qisqa", source="a.txt")
    long = KnowledgeDocument(
        id="long",
        title="Sho'rlanish",
        content="Tuproq sho'rlanishi haqida batafsil ma'lumot. " * 10,
        source="FAO",
        category="soil",
    )
    records = list(grounded_records([short, long]))
    assert len(records) == 1
    assert records[0].source == "FAO"
    assert records[0].output == long.content.strip()
    assert records[0].trust_level == "tier_1"
    assert records[0].metadata["document_id"] == "long"

    dataset = FineTuningDataset.from_documents([long])
    assert len(dataset) == 1
    assert dataset.problems == []

    dataset_path = tmp_path / "out.jsonl"
    assert dataset.to_jsonl(dataset_path) == 1
    assert len(FineTuningDataset.from_jsonl(dataset_path)) == 1


def test_instruction_dataset_synthesis_is_not_faked() -> None:
    with pytest.raises(NotImplementedError, match="requires a generator model"):
        build_instruction_dataset([])


# --------------------------------------------------------------------- #
# Fine-tuning: LoRA
# --------------------------------------------------------------------- #
def test_lora_config_validation() -> None:
    valid = LoRAConfig(base_model="models/qwen.gguf")
    assert valid.is_valid is True
    assert valid.to_dict()["target_modules"] == ["q_proj", "v_proj"]

    broken = LoRAConfig(base_model="", rank=0, dropout=2.0, epochs=0, max_seq_length=10)
    problems = broken.problems()
    assert any("base_model" in problem for problem in problems)
    assert any("rank" in problem for problem in problems)
    assert any("dropout" in problem for problem in problems)
    assert any("epochs" in problem for problem in problems)
    assert any("max_seq_length" in problem for problem in problems)


def test_lora_trainer_refuses_honestly(tmp_path: Path) -> None:
    config = LoRAConfig(base_model="models/qwen.gguf", dataset_path=str(tmp_path / "missing.jsonl"))
    trainer = LoRATrainer(config)
    outcome = trainer.train()
    assert outcome.ok is False
    assert outcome.status == "not_implemented"
    assert "not implemented" in outcome.detail
    assert any("does not exist" in problem for problem in trainer.preflight())
    assert trainer.dataset_summary()["records"] == 0

    assert training_requirements()["implemented"] is False


def test_lora_trainer_reports_dataset_preflight(tmp_path: Path) -> None:
    # An unattributed record cannot even be loaded from disk...
    dataset_path = tmp_path / "data.jsonl"
    write_jsonl([_record(source="")], dataset_path)
    with pytest.raises(ValueError, match="missing required field"):
        read_jsonl(dataset_path)

    # ...and in memory it is reported as a blocking problem.
    in_memory = LoRATrainer(
        LoRAConfig(base_model="m.gguf"), FineTuningDataset([_record(source="")])
    )
    assert in_memory.dataset_summary()["records"] == 1
    assert any("missing source" in problem for problem in in_memory.preflight())

    good_path = tmp_path / "good.jsonl"
    write_jsonl([_record()], good_path)
    clean = LoRATrainer(LoRAConfig(base_model="m.gguf", dataset_path=str(good_path)))
    assert clean.dataset_summary()["records"] == 1
    assert clean.preflight() == []
