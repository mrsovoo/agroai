"""Dataset versions and manifests.

```text
raw data -> clean -> verified -> labeled -> train -> validation -> test
                                              -> AgroAI dataset v1
```

The roadmap requires the dataset itself to be versioned, with each version
recording what went into it. Everything here is **declared metadata plus real
record counts**: a manifest never claims a quality it did not measure.

A :class:`DatasetVersion` owns three splits (train/valid/test) and a manifest
file. Splits are assigned deterministically (:mod:`agroai.datasets.splits`), so
the same records always produce the same split - a prerequisite for comparing
two training runs at all.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from agroai.knowledge.metadata import utcnow_iso

SPLIT_TRAIN = "train"
SPLIT_VALID = "valid"
SPLIT_TEST = "test"

SPLITS: tuple[str, ...] = (SPLIT_TRAIN, SPLIT_VALID, SPLIT_TEST)

#: Default split ratios (must sum to 1.0).
DEFAULT_SPLIT_RATIOS: dict[str, float] = {
    SPLIT_TRAIN: 0.8,
    SPLIT_VALID: 0.1,
    SPLIT_TEST: 0.1,
}

MANIFEST_FILE = "manifest.json"


@dataclass
class DatasetVersion:
    """One versioned dataset: a name, a version, three splits and a manifest."""

    name: str
    version: str
    path: str = ""
    description: str = ""
    created_at: str = field(default_factory=utcnow_iso)
    splits: dict[str, int] = field(default_factory=dict)
    #: Where the records came from (knowledge documents, exported files, ...).
    sources: list[str] = field(default_factory=list)
    #: Free-form provenance: knowledge base counts, ingestion dates, filters.
    metadata: dict[str, Any] = field(default_factory=dict)
    #: Problems found while building the version (non-empty = do not train on it).
    problems: list[str] = field(default_factory=list)

    @property
    def total(self) -> int:
        return sum(self.splits.values())

    @property
    def is_trainable(self) -> bool:
        """A version is trainable only if it has data and no recorded problems."""
        return self.total > 0 and not self.problems

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "DatasetVersion":
        known = set(cls.__dataclass_fields__)
        clean = {key: value for key, value in data.items() if key in known}
        if not clean.get("name") or not clean.get("version"):
            raise ValueError("a dataset version needs both 'name' and 'version'")
        return cls(**clean)  # type: ignore[arg-type]

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "records": self.total,
            "splits": dict(self.splits),
            "trainable": self.is_trainable,
            "problems": list(self.problems),
            "created_at": self.created_at,
        }

    def render(self) -> str:
        lines = [
            f"Dataset {self.name} v{self.version} - {self.total} records",
            f"  splits: " + ", ".join(f"{key}={value}" for key, value in self.splits.items()),
            f"  trainable: {self.is_trainable}",
        ]
        if self.description:
            lines.append(f"  {self.description}")
        if self.sources:
            lines.append(f"  sources: {', '.join(self.sources[:8])}")
        for problem in self.problems:
            lines.append(f"  ! {problem}")
        return "\n".join(lines)


__all__ = [
    "DEFAULT_SPLIT_RATIOS",
    "DatasetVersion",
    "MANIFEST_FILE",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VALID",
    "SPLITS",
]
