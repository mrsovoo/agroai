"""LoRA fine-tuning: configuration schema and an honest trainer boundary.

The roadmap's model phase is ``base model + AgroAI dataset -> LoRA/fine-tuning ->
AgroAI model``. The *configuration* is real and validated here, because it is
what a future training run and its provenance record need. The *training* is
not implemented: it needs GPU-class dependencies that AgroAI Core deliberately
does not require, and no run may be reported that did not happen.

Note the architectural rule from the roadmap: the knowledge base is **not**
stuffed into the model. Fine-tuning teaches terminology, style and reasoning
patterns; the facts keep living in the retrievable knowledge base.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

from agroai.models.fine_tuning.dataset import FineTuningDataset


@dataclass
class LoRAConfig:
    """Validated configuration for a LoRA/adapter training run."""

    base_model: str
    dataset_path: str = ""
    output_dir: str = "models/agroai-lora"
    rank: int = 8
    alpha: int = 16
    dropout: float = 0.05
    target_modules: tuple[str, ...] = ("q_proj", "v_proj")
    learning_rate: float = 2e-4
    epochs: int = 3
    batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_seq_length: int = 1024
    seed: int = 42
    notes: str = field(default="")

    def problems(self) -> list[str]:
        """Reasons this configuration cannot be trained as written."""
        found: list[str] = []
        if not self.base_model.strip():
            found.append("base_model is required")
        if self.rank < 1:
            found.append("rank must be >= 1")
        if self.alpha < 1:
            found.append("alpha must be >= 1")
        if not 0.0 <= self.dropout < 1.0:
            found.append("dropout must be in [0, 1)")
        if not 0.0 < self.learning_rate <= 1.0:
            found.append("learning_rate must be in (0, 1]")
        if self.epochs < 1:
            found.append("epochs must be >= 1")
        if self.batch_size < 1:
            found.append("batch_size must be >= 1")
        if self.max_seq_length < 64:
            found.append("max_seq_length must be >= 64")
        if not self.target_modules:
            found.append("target_modules must not be empty")
        return found

    @property
    def is_valid(self) -> bool:
        return not self.problems()

    def to_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["target_modules"] = list(self.target_modules)
        return data


@dataclass
class TrainingOutcome:
    """What a training attempt actually produced."""

    status: str
    detail: str
    adapter_path: str = ""
    steps: int = 0

    @property
    def ok(self) -> bool:
        return self.status == "ok"


class LoRATrainer:
    """Interface for a future LoRA training run."""

    def __init__(self, config: LoRAConfig, dataset: FineTuningDataset | None = None) -> None:
        self.config = config
        self.dataset = dataset

    def dataset_summary(self) -> dict[str, object]:
        """Summarise the data this run would use (works offline today)."""
        if self.dataset is None:
            if self.config.dataset_path and Path(self.config.dataset_path).is_file():
                self.dataset = FineTuningDataset.from_jsonl(self.config.dataset_path)
            else:
                return {"records": 0, "source": self.config.dataset_path or "(none)"}
        summary = self.dataset.summary()
        summary["source"] = self.config.dataset_path or "(in-memory)"
        return summary

    def preflight(self) -> list[str]:
        """Everything that must be fixed before training could start."""
        problems = list(self.config.problems())
        if not self.config.dataset_path and self.dataset is None:
            problems.append("no dataset configured (set dataset_path or pass a dataset)")
        elif (
            self.config.dataset_path
            and self.dataset is None
            and not Path(self.config.dataset_path).is_file()
        ):
            problems.append(f"dataset_path does not exist: {self.config.dataset_path}")
        if self.dataset is not None:
            problems.extend(self.dataset.problems)
        return problems

    def train(self) -> TrainingOutcome:
        """Refuse honestly: AgroAI Core ships no training runtime."""
        problems = self.preflight()
        detail = (
            "LoRA fine-tuning is not implemented in this phase. AgroAI Core has "
            "no training runtime (and deliberately no GPU dependencies); a "
            "training run must not be reported as done when it never happened."
        )
        if problems:
            detail += " Configuration problems: " + "; ".join(problems) + "."
        return TrainingOutcome(status="not_implemented", detail=detail)


def training_requirements() -> dict[str, object]:
    """What the fine-tuning phase will need, stated plainly and offline."""
    return {
        "implemented": False,
        "needs": [
            "a base model in GGUF or safetensors format",
            "a validated JSONL dataset (see agroai.models.fine_tuning.dataset)",
            "a training runtime installed by the user (e.g. PEFT/LoRA tooling)",
            "compute: GPU is strongly recommended",
        ],
        "produces": "an adapter directory, never a changed knowledge base",
    }


__all__ = ["LoRAConfig", "LoRATrainer", "TrainingOutcome", "training_requirements"]
