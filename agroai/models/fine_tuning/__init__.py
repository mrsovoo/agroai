"""Fine-tuning interfaces: dataset schema/IO, LoRA configuration, run manager.

Neither training nor instruction-data synthesis is implemented; both boundaries
raise or report honestly. What *is* real:

* the dataset schema and JSONL IO (:mod:`~agroai.models.fine_tuning.dataset`),
* validated LoRA configuration and preflight checks
  (:mod:`~agroai.models.fine_tuning.lora`),
* planning and recording training runs, with their dataset/model provenance
  (:mod:`~agroai.models.fine_tuning.manager`).
"""

from __future__ import annotations

from agroai.models.fine_tuning.dataset import (
    KIND_GROUNDED_PASSAGE,
    KIND_INSTRUCTION,
    KIND_REASONING,
    KIND_TERMINOLOGY,
    DatasetRecord,
    FineTuningDataset,
    build_instruction_dataset,
    grounded_records,
    read_jsonl,
    summarize,
    validate_records,
    write_jsonl,
)
from agroai.models.fine_tuning.lora import (
    LoRAConfig,
    LoRATrainer,
    TrainingOutcome,
    training_requirements,
)
from agroai.models.fine_tuning.manager import (
    RUNS_FILE,
    FineTuningError,
    FineTuningManager,
    RunStatus,
    TrainingNotAvailable,
    TrainingRun,
    promotion_path,
)

__all__ = [
    "KIND_GROUNDED_PASSAGE",
    "KIND_INSTRUCTION",
    "KIND_REASONING",
    "KIND_TERMINOLOGY",
    "DatasetRecord",
    "FineTuningDataset",
    "FineTuningError",
    "FineTuningManager",
    "LoRAConfig",
    "LoRATrainer",
    "RUNS_FILE",
    "RunStatus",
    "TrainingNotAvailable",
    "TrainingOutcome",
    "TrainingRun",
    "build_instruction_dataset",
    "grounded_records",
    "promotion_path",
    "read_jsonl",
    "summarize",
    "training_requirements",
    "validate_records",
    "write_jsonl",
]
