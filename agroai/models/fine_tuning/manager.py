"""Fine-tuning run manager: plan, trace and record training runs honestly.

```text
AgroAI v1 -> new verified knowledge -> better dataset -> training -> AgroAI v2
```

The manager exists so that when a real training run happens, its provenance is
already recorded in the right shape. It links the three artefacts the roadmap
demands stay connected:

```text
TrainingRun -> dataset (name@version) -> ModelRecord (base model, config, eval)
```

Its boundary is explicit:

* :meth:`FineTuningManager.plan_run` - **real**: validates the dataset, the
  configuration and the references, then records a ``PLANNED`` run.
* :meth:`FineTuningManager.record_completion` - **real**: records a run that
  actually happened elsewhere (a GPU box, a Colab notebook) and moves the model
  to ``TRAINED``.
* :meth:`FineTuningManager.start_run` - **raises** ``TrainingNotAvailable``.
  AgroAI Core ships no training runtime, and a run that did not happen must never
  appear as one that did.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from agroai.knowledge.metadata import utcnow_iso
from agroai.models.fine_tuning.lora import LoRAConfig, training_requirements
from agroai.models.registry import (
    TRANSITIONS,
    ModelRecord,
    ModelRegistry,
    ModelStage,
)

if TYPE_CHECKING:  # pragma: no cover - import-time cycle guard
    from agroai.datasets.manager import DatasetManager
else:
    DatasetManager = None  # resolved lazily via _dataset_manager()


def _dataset_manager_class() -> type:
    """Import DatasetManager lazily.

    ``agroai.datasets.manager`` imports the dataset record schema from
    ``agroai.models.fine_tuning.dataset``; importing that package's ``__init__``
    here at module level closes a cycle (datasets -> fine_tuning -> datasets)
    that broke ``agroai models``/``agroai datasets``.
    """
    from agroai.datasets.manager import DatasetManager as _DM

    return _DM


RUNS_FILE = "training_runs.json"


class RunStatus(str, Enum):
    """Lifecycle of a training run."""

    PLANNED = "planned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class TrainingNotAvailable(RuntimeError):
    """Raised when a caller asks this build to actually train a model."""


class FineTuningError(RuntimeError):
    """Raised for an invalid run plan (bad reference, invalid config, ...)."""


@dataclass
class TrainingRun:
    """One planned or completed fine-tuning run, with its full provenance."""

    run_id: str
    model_name: str
    model_version: str
    base_model: str = ""
    dataset: str = ""
    config: dict[str, Any] = field(default_factory=dict)
    status: str = RunStatus.PLANNED.value
    output_path: str = ""
    evaluation: dict[str, Any] = field(default_factory=dict)
    problems: list[str] = field(default_factory=list)
    log: list[str] = field(default_factory=list)
    created_at: str = field(default_factory=utcnow_iso)
    updated_at: str = field(default_factory=utcnow_iso)

    @property
    def status_enum(self) -> RunStatus:
        return RunStatus(self.status)

    @property
    def model_key(self) -> str:
        return f"{self.model_name}@{self.model_version}"

    @property
    def is_trainable(self) -> bool:
        return not self.problems and self.status_enum is RunStatus.PLANNED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TrainingRun":
        known = set(cls.__dataclass_fields__)
        clean = {key: value for key, value in data.items() if key in known}
        if not clean.get("run_id"):
            raise ValueError("a training run needs a 'run_id'")
        return cls(**clean)  # type: ignore[arg-type]

    def render(self) -> str:
        lines = [f"run {self.run_id}: {self.model_key} [{self.status}]"]
        if self.base_model:
            lines.append(f"  base model: {self.base_model}")
        if self.dataset:
            lines.append(f"  dataset: {self.dataset}")
        if self.config:
            lines.append(
                "  config: "
                + ", ".join(f"{key}={value}" for key, value in sorted(self.config.items()))
            )
        if self.output_path:
            lines.append(f"  output: {self.output_path}")
        if self.evaluation:
            lines.append(f"  evaluation: {self.evaluation.get('suite', '?')}")
        for problem in self.problems:
            lines.append(f"  ! {problem}")
        return "\n".join(lines)


def promotion_path(current: ModelStage, target: ModelStage) -> list[ModelStage]:
    """Shortest allowed path from ``current`` to ``target`` (empty when equal).

    The registry's transition table is the authority; this only walks it, so a
    caller can move a model through ``PLANNED -> TRAINING -> TRAINED`` without
    hand-coding each step.
    """
    if current is target:
        return []
    queue: list[list[ModelStage]] = [[current]]
    visited = {current}
    while queue:
        path = queue.pop(0)
        for step in sorted(TRANSITIONS[path[-1]], key=lambda item: item.value):
            if step is target:
                return [*path[1:], step]
            if step in visited:
                continue
            visited.add(step)
            queue.append([*path, step])
    raise FineTuningError(
        f"no allowed lifecycle path from {current.value} to {target.value}"
    )


class FineTuningManager:
    """Plans and records fine-tuning runs; never fakes one."""

    def __init__(
        self,
        models_dir: Path | str,
        *,
        dataset_manager: "DatasetManager | None" = None,
        registry: ModelRegistry | None = None,
    ) -> None:
        self.root = Path(models_dir)
        self.path = self.root / RUNS_FILE
        self.datasets = dataset_manager or _dataset_manager_class()(
            self.root.parent / "datasets"
        )
        self.registry = registry or ModelRegistry(self.root)
        self._runs: dict[str, TrainingRun] = {}
        self._load()

    # ------------------------------------------------------------------ #
    @property
    def runs(self) -> list[TrainingRun]:
        return sorted(self._runs.values(), key=lambda item: (item.created_at, item.run_id))

    def __len__(self) -> int:
        return len(self._runs)

    def get_run(self, run_id: str) -> TrainingRun:
        try:
            return self._runs[run_id]
        except KeyError as exc:
            raise KeyError(f"no training run {run_id!r}") from exc

    # ------------------------------------------------------------------ #
    def plan_run(
        self,
        *,
        model_name: str,
        model_version: str,
        base_model: str,
        dataset: str,
        config: LoRAConfig | dict[str, Any] | None = None,
        run_id: str | None = None,
    ) -> TrainingRun:
        """Validate and record a training plan (no training is performed).

        Validation is real: the dataset must exist and be trainable, the config
        must be valid, and every problem is recorded on the run as a reason it
        cannot start.
        """
        problems: list[str] = []

        dataset_version = None
        if not dataset or "@" not in dataset:
            problems.append(
                "dataset must reference name@version (see 'agroai datasets' to create one)"
            )
        else:
            name, _, version = dataset.partition("@")
            try:
                dataset_version = self.datasets.get(name, version)
            except FileNotFoundError:
                problems.append(f"dataset {dataset} does not exist on disk")
            else:
                if not dataset_version.is_trainable:
                    problems.append(
                        f"dataset {dataset} is not trainable "
                        f"({len(dataset_version.problems)} problem(s) recorded)"
                    )
                    problems.extend(dataset_version.problems)

        resolved_config: dict[str, Any] = {}
        if isinstance(config, LoRAConfig):
            resolved_config = config.to_dict()
            problems.extend(config.problems())
        elif isinstance(config, dict):
            resolved_config = dict(config)
        else:
            resolved_config = {}

        if not base_model.strip():
            problems.append("base_model is required (a model may not be improvised)")

        run = TrainingRun(
            run_id=run_id or f"{model_name}-{model_version}",
            model_name=model_name,
            model_version=model_version,
            base_model=base_model,
            dataset=dataset,
            config=resolved_config,
            status=RunStatus.PLANNED.value,
            problems=problems,
            log=[
                "plan created",
                f"dataset records: {dataset_version.total if dataset_version else 'unknown'}",
            ],
        )
        if run.run_id in self._runs:
            raise FineTuningError(f"run {run.run_id!r} already exists")
        self._runs[run.run_id] = run
        self._save()

        # Register the model version so provenance exists before training does.
        if run.model_key not in self.registry:
            self.registry.register(
                ModelRecord(
                    name=model_name,
                    version=model_version,
                    base_model=base_model,
                    dataset=dataset,
                    config=resolved_config,
                    stage=ModelStage.PLANNED.value,
                    notes=["registered by FineTuningManager.plan_run"],
                )
            )
        return run

    def start_run(self, run_id: str) -> TrainingRun:
        """Refuse honestly: this build cannot train.

        Raises :class:`TrainingNotAvailable` with the concrete missing pieces
        (also recorded on the run) instead of returning a fake outcome.
        """
        run = self.get_run(run_id)
        requirements = training_requirements()
        detail = (
            "LoRA fine-tuning is not implemented in this build: AgroAI Core has no "
            "training runtime. " + "; ".join(str(item) for item in requirements["needs"])
        )
        run.log.append("start refused: " + detail)
        if run.problems:
            detail += " Recorded problems: " + "; ".join(run.problems) + "."
        run.updated_at = utcnow_iso()
        self._save()
        raise TrainingNotAvailable(detail)

    def record_completion(
        self,
        run_id: str,
        *,
        output_path: str,
        stages: int = 0,
        evaluation: dict[str, Any] | None = None,
        note: str = "",
    ) -> TrainingRun:
        """Record a training run that happened outside this process.

        The caller must have actually produced ``output_path``; this method
        records the fact and moves the model version to ``TRAINED`` (or
        ``EVALUATED`` when an evaluation summary is supplied).
        """
        run = self.get_run(run_id)
        if run.problems:
            raise FineTuningError(
                "refusing to record completion for a run with unresolved problems: "
                + "; ".join(run.problems)
            )
        if not output_path:
            raise FineTuningError("output_path is required: an adapter that was not produced cannot be recorded")
        if not Path(output_path).exists():
            raise FineTuningError(
                f"output_path does not exist: {output_path} (record only what was actually produced)"
            )

        run.status = RunStatus.COMPLETED.value
        run.output_path = output_path
        run.updated_at = utcnow_iso()
        run.log.append(
            f"completed: {stages} step(s) -> {output_path}" + (f" ({note})" if note else "")
        )
        if evaluation:
            run.evaluation = dict(evaluation)
        self._save()

        target = ModelStage.EVALUATED if evaluation else ModelStage.TRAINED
        record = self.registry.get(run.model_name, run.model_version)
        current = ModelStage(record.stage)
        for step in promotion_path(current, target):
            self.registry.promote(
                run.model_name,
                run.model_version,
                step,
                evaluation=evaluation if step is target else None,
                path=output_path,
                note=f"recorded by run {run.run_id}",
            )
        return run

    def record_failure(self, run_id: str, reason: str) -> TrainingRun:
        """Record that a run failed, with the reason."""
        if not reason.strip():
            raise FineTuningError("a failure must state a reason")
        run = self.get_run(run_id)
        run.status = RunStatus.FAILED.value
        run.updated_at = utcnow_iso()
        run.log.append(f"failed: {reason}")
        self._save()
        try:
            self.registry.promote(run.model_name, run.model_version, ModelStage.FAILED, note=reason)
        except Exception:  # noqa: BLE001 - registry state must not hide the failure
            pass
        return run

    # ------------------------------------------------------------------ #
    def requirements(self) -> dict[str, Any]:
        return {
            **training_requirements(),
            "runs": len(self._runs),
            "planned": sum(
                1 for run in self._runs.values() if run.status_enum is RunStatus.PLANNED
            ),
            "completed": sum(
                1 for run in self._runs.values() if run.status_enum is RunStatus.COMPLETED
            ),
        }

    def stats(self) -> dict[str, Any]:
        by_status: dict[str, int] = {}
        for run in self._runs.values():
            by_status[run.status] = by_status.get(run.status, 0) + 1
        return {
            "path": str(self.path),
            "runs": len(self._runs),
            "by_status": by_status,
            "trainable": sum(1 for run in self._runs.values() if run.is_trainable),
        }

    # ------------------------------------------------------------------ #
    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return
        for item in data.get("runs", []):
            try:
                run = TrainingRun.from_dict(item)
            except (ValueError, TypeError):
                continue
            self._runs[run.run_id] = run

    def _save(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "updated_at": utcnow_iso(),
            "runs": [run.to_dict() for run in self.runs],
        }
        self.path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )


__all__ = [
    "FineTuningError",
    "FineTuningManager",
    "RUNS_FILE",
    "RunStatus",
    "TrainingNotAvailable",
    "TrainingRun",
    "promotion_path",
]
