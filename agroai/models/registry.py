"""Model registry: which AgroAI model came from what, and whether it is safe.

```text
AgroAI-base
AgroAI-v1
AgroAI-v1.1
AgroAI-v2
AgroAI-Vision-v1
```

For every one of those the roadmap requires the same four answers: which **base
model** it came from, which **dataset**, which **training configuration**, and
which **evaluation** result. This registry records exactly that - one JSON file,
no server, standard library only.

Two rules are enforced, because they are the whole point of a registry:

* a model may not become ``RELEASED`` without a recorded evaluation. The
  promotion path refuses, rather than letting "released" mean "we hope so";
* nothing here trains or invents numbers. A :class:`ModelRecord` stores facts a
  caller declares; the registry's job is to keep them consistent and traceable.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Iterable

from agroai.knowledge.metadata import utcnow_iso

REGISTRY_FILE = "registry.json"

MODEL_KIND_LLM = "llm"
MODEL_KIND_VISION = "vision"
MODEL_KIND_EMBEDDING = "embedding"
MODEL_KINDS: tuple[str, ...] = (MODEL_KIND_LLM, MODEL_KIND_VISION, MODEL_KIND_EMBEDDING)


class ModelStage(str, Enum):
    """Lifecycle stage of a model version."""

    PLANNED = "planned"
    TRAINING = "training"
    TRAINED = "trained"
    EVALUATED = "evaluated"
    RELEASED = "released"
    REJECTED = "rejected"
    FAILED = "failed"

    @property
    def is_usable(self) -> bool:
        return self in {ModelStage.EVALUATED, ModelStage.RELEASED}


#: Allowed transitions. Anything else raises, so the lifecycle cannot drift.
TRANSITIONS: dict[ModelStage, frozenset[ModelStage]] = {
    ModelStage.PLANNED: frozenset({ModelStage.TRAINING, ModelStage.REJECTED, ModelStage.FAILED}),
    ModelStage.TRAINING: frozenset({ModelStage.TRAINED, ModelStage.FAILED, ModelStage.REJECTED}),
    ModelStage.TRAINED: frozenset({ModelStage.EVALUATED, ModelStage.FAILED, ModelStage.REJECTED}),
    ModelStage.EVALUATED: frozenset({ModelStage.RELEASED, ModelStage.TRAINING, ModelStage.REJECTED}),
    ModelStage.RELEASED: frozenset({ModelStage.REJECTED}),
    ModelStage.REJECTED: frozenset({ModelStage.PLANNED}),
    ModelStage.FAILED: frozenset({ModelStage.PLANNED}),
}


class RegistryError(RuntimeError):
    """Raised for an invalid registry operation (bad transition, duplicate, ...)."""


@dataclass
class ModelRecord:
    """One version of one model, with everything needed to reproduce it."""

    name: str
    version: str
    kind: str = MODEL_KIND_LLM
    base_model: str = ""
    #: Dataset reference as ``name@version`` (see :mod:`agroai.datasets`).
    dataset: str = ""
    #: Training configuration actually used (LoRA ranks, epochs, ...).
    config: dict[str, Any] = field(default_factory=dict)
    stage: str = ModelStage.PLANNED.value
    path: str = ""
    #: Evaluation summary: ``{"suite": ..., "metrics": {...}, "report": path}``.
    evaluation: dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)
    notes: list[str] = field(default_factory=list)

    @property
    def key(self) -> str:
        return f"{self.name}@{self.version}"

    @property
    def stage_enum(self) -> ModelStage:
        return ModelStage(self.stage)

    @property
    def is_released(self) -> bool:
        return self.stage == ModelStage.RELEASED.value

    @property
    def has_evaluation(self) -> bool:
        return bool(self.evaluation)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ModelRecord":
        known = set(cls.__dataclass_fields__)
        clean = {key: value for key, value in data.items() if key in known}
        if not clean.get("name") or not clean.get("version"):
            raise ValueError("a model record needs both 'name' and 'version'")
        return cls(**clean)  # type: ignore[arg-type]

    def render(self) -> str:
        lines = [f"{self.key}  [{self.kind}, {self.stage}]"]
        if self.base_model:
            lines.append(f"  base model: {self.base_model}")
        if self.dataset:
            lines.append(f"  dataset: {self.dataset}")
        if self.config:
            lines.append(
                "  config: "
                + ", ".join(f"{key}={value}" for key, value in sorted(self.config.items()))
            )
        if self.path:
            lines.append(f"  path: {self.path}")
        if self.evaluation:
            metrics = self.evaluation.get("metrics", {})
            shown = ", ".join(f"{key}={value}" for key, value in sorted(metrics.items())[:6])
            lines.append(
                f"  evaluation: suite={self.evaluation.get('suite', '?')}"
                + (f" ({shown})" if shown else "")
            )
        else:
            lines.append("  evaluation: (none recorded)")
        for note in self.notes:
            lines.append(f"  ! {note}")
        return "\n".join(lines)


class ModelRegistry:
    """JSON-backed registry of model versions, with enforced lifecycle rules."""

    def __init__(self, models_dir: Path | str) -> None:
        self.root = Path(models_dir)
        self.path = self.root / REGISTRY_FILE
        self._records: dict[str, ModelRecord] = {}
        self._load()

    # ------------------------------------------------------------------ #
    @property
    def records(self) -> list[ModelRecord]:
        return sorted(self._records.values(), key=lambda item: (item.name, item.version))

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, key: object) -> bool:
        return key in self._records

    # ------------------------------------------------------------------ #
    def register(self, record: ModelRecord, *, overwrite: bool = False) -> ModelRecord:
        """Record a model version. Duplicates raise unless ``overwrite``."""
        if record.kind not in MODEL_KINDS:
            raise RegistryError(
                f"unknown model kind {record.kind!r}; expected one of {MODEL_KINDS}"
            )
        ModelStage(record.stage)  # validates the stage string
        existing = self._records.get(record.key)
        if existing is not None and not overwrite:
            raise RegistryError(
                f"{record.key} is already registered (stage={existing.stage}); "
                "bump the version or pass overwrite=True"
            )
        if existing is not None:
            record.created_at = existing.created_at
        record.updated_at = utcnow_iso()
        self._records[record.key] = record
        self._save()
        return record

    def get(self, name: str, version: str) -> ModelRecord:
        key = f"{name}@{version}"
        try:
            return self._records[key]
        except KeyError as exc:
            raise KeyError(f"{key} is not registered ({len(self)} record(s) known)") from exc

    def list_models(
        self,
        *,
        name: str | None = None,
        kind: str | None = None,
        stage: str | None = None,
        usable_only: bool = False,
    ) -> list[ModelRecord]:
        found = self.records
        if name is not None:
            found = [item for item in found if item.name == name]
        if kind is not None:
            found = [item for item in found if item.kind == kind]
        if stage is not None:
            found = [item for item in found if item.stage == stage]
        if usable_only:
            found = [item for item in found if item.stage_enum.is_usable]
        return found

    def latest(self, name: str, *, usable_only: bool = False) -> ModelRecord | None:
        found = self.list_models(name=name, usable_only=usable_only)
        if not found:
            return None
        return sorted(found, key=lambda item: (item.created_at, item.version))[-1]

    # ------------------------------------------------------------------ #
    def promote(
        self,
        name: str,
        version: str,
        stage: ModelStage | str,
        *,
        evaluation: dict[str, Any] | None = None,
        path: str | None = None,
        config: dict[str, Any] | None = None,
        note: str = "",
    ) -> ModelRecord:
        """Move a model version to a new lifecycle stage.

        Enforces the transition table and refuses ``RELEASED`` without a recorded
        evaluation - the check that keeps "released" meaningful.
        """
        target = ModelStage(stage) if not isinstance(stage, ModelStage) else stage
        record = self.get(name, version)
        current = record.stage_enum
        if target is not current and target not in TRANSITIONS[current]:
            raise RegistryError(
                f"{record.key}: cannot move {current.value} -> {target.value}; "
                f"allowed: {', '.join(sorted(item.value for item in TRANSITIONS[current])) or '(none)'}"
            )
        if target is ModelStage.RELEASED and not (evaluation or record.evaluation):
            raise RegistryError(
                f"{record.key}: a model cannot be RELEASED without a recorded "
                "evaluation. Run the evaluation suite and attach its result."
            )
        if target is ModelStage.EVALUATED and not (evaluation or record.evaluation):
            raise RegistryError(
                f"{record.key}: EVALUATED requires the evaluation summary that "
                "justifies it."
            )

        if evaluation:
            record.evaluation = dict(evaluation)
        if path:
            record.path = path
        if config:
            record.config = {**record.config, **config}
        record.stage = target.value
        record.updated_at = utcnow_iso()
        if note:
            record.notes.append(note)
        self._save()
        return record

    # ------------------------------------------------------------------ #
    def history(self, name: str) -> list[ModelRecord]:
        """Every registered version of a model, oldest first."""
        return sorted(self.list_models(name=name), key=lambda item: item.created_at)

    def lineage(self, name: str, version: str) -> dict[str, Any]:
        """The full provenance chain of one version."""
        record = self.get(name, version)
        return {
            "model": record.key,
            "kind": record.kind,
            "base_model": record.base_model,
            "dataset": record.dataset,
            "config": dict(record.config),
            "stage": record.stage,
            "evaluation": dict(record.evaluation),
            "path": record.path,
            "created_at": record.created_at,
            "updated_at": record.updated_at,
        }

    def stats(self) -> dict[str, Any]:
        by_stage: dict[str, int] = {}
        by_kind: dict[str, int] = {}
        for record in self._records.values():
            by_stage[record.stage] = by_stage.get(record.stage, 0) + 1
            by_kind[record.kind] = by_kind.get(record.kind, 0) + 1
        return {
            "path": str(self.path),
            "models": len({record.name for record in self._records.values()}),
            "versions": len(self._records),
            "by_stage": by_stage,
            "by_kind": by_kind,
            "released": sum(1 for record in self._records.values() if record.is_released),
        }

    def requirements(self) -> dict[str, Any]:
        """What the roadmap still needs before a real model can exist."""
        return {
            "training_runtime": "not implemented (requires PyTorch/transformers + GPU)",
            "dataset_ready": "datasets/ managed by agroai.datasets.DatasetManager",
            "evaluation_ready": "agroai.evaluation (run 'agroai eval')",
            "registered_versions": len(self._records),
            "released_versions": sorted(
                record.key for record in self._records.values() if record.is_released
            ),
        }

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            # A corrupt registry is not silently replaced; it is reported loudly.
            raise RegistryError(
                f"registry file {self.path} is not valid JSON; refusing to overwrite it"
            ) from None
        for item in data.get("models", []):
            try:
                record = ModelRecord.from_dict(item)
            except (ValueError, TypeError):
                continue
            self._records[record.key] = record

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": utcnow_iso(),
            "models": [record.to_dict() for record in self.records],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # Convenience ------------------------------------------------------- #
    def bulk_register(self, records: Iterable[ModelRecord]) -> list[ModelRecord]:
        return [self.register(record) for record in records]


__all__ = [
    "MODEL_KINDS",
    "MODEL_KIND_EMBEDDING",
    "MODEL_KIND_LLM",
    "MODEL_KIND_VISION",
    "ModelRecord",
    "ModelRegistry",
    "ModelStage",
    "REGISTRY_FILE",
    "RegistryError",
    "TRANSITIONS",
]
