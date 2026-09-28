"""Dataset manager: build, version and inspect training datasets on disk.

```text
datasets/
  agroai-sft/
    v1/
      manifest.json
      train.jsonl
      valid.jsonl
      test.jsonl
    v2/
      ...
```

Layout mirrors ``models/`` so a training run can name exactly which dataset it
consumed (``agroai-sft@v2``). The manager:

* splits records deterministically (see :mod:`agroai.datasets.splits`),
* refuses to write a dataset containing invalid records - an invalid dataset
  silently trained on is a corrupted model,
* records provenance (where records came from) in the manifest,
* checks for split leakage and records it as a problem.

It does not train anything and does not fabricate records: it organises data that
already exists.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from agroai.datasets.base import (
    DEFAULT_SPLIT_RATIOS,
    MANIFEST_FILE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALID,
    SPLITS,
    DatasetVersion,
)
from agroai.datasets.splits import find_leakage, split_records
from agroai.models.fine_tuning.dataset import DatasetRecord, read_jsonl, write_jsonl


class DatasetManager:
    """Filesystem-backed store of versioned datasets."""

    def __init__(self, datasets_dir: Path | str) -> None:
        self.root = Path(datasets_dir)

    # ------------------------------------------------------------------ #
    # Paths
    # ------------------------------------------------------------------ #
    def version_dir(self, name: str, version: str) -> Path:
        return self.root / _safe(name) / _safe(version)

    def manifest_path(self, name: str, version: str) -> Path:
        return self.version_dir(name, version) / MANIFEST_FILE

    def split_path(self, name: str, version: str, split: str) -> Path:
        if split not in SPLITS:
            raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
        return self.version_dir(name, version) / f"{split}.jsonl"

    # ------------------------------------------------------------------ #
    # Writing
    # ------------------------------------------------------------------ #
    def create_version(
        self,
        name: str,
        version: str,
        records: Iterable[DatasetRecord],
        *,
        description: str = "",
        sources: Iterable[str] = (),
        metadata: dict[str, Any] | None = None,
        ratios: dict[str, float] | None = None,
        overwrite: bool = False,
        allow_invalid: bool = False,
    ) -> DatasetVersion:
        """Split ``records`` and write a versioned dataset with its manifest."""
        directory = self.version_dir(name, version)
        manifest = self.manifest_path(name, version)
        if manifest.exists() and not overwrite:
            raise FileExistsError(
                f"dataset {name}@{version} already exists at {directory}; "
                "bump the version or pass overwrite=True"
            )

        materialized = list(records)
        problems = _validate(materialized)
        if problems and not allow_invalid:
            raise ValueError(
                "refusing to write an invalid dataset (pass allow_invalid=True to "
                "record it anyway):\n  - " + "\n  - ".join(problems[:20])
            )

        splits = split_records(materialized, ratios or DEFAULT_SPLIT_RATIOS)
        directory.mkdir(parents=True, exist_ok=True)
        counts: dict[str, int] = {}
        for split in SPLITS:
            path = self.split_path(name, version, split)
            counts[split] = write_jsonl(splits.get(split, []), path)

        leakage = find_leakage(splits)
        problems = [*problems, *leakage]
        empty_splits = [split for split, count in counts.items() if count == 0]
        if empty_splits and len(materialized) >= len(SPLITS):
            problems.append(
                "empty split(s): " + ", ".join(empty_splits)
                + " (adjust the ratios or add more records)"
            )

        dataset = DatasetVersion(
            name=name,
            version=version,
            path=str(directory),
            description=description,
            splits=counts,
            sources=[str(item) for item in sources],
            metadata={
                **(metadata or {}),
                "ratios": {key: round(value, 4) for key, value in (ratios or DEFAULT_SPLIT_RATIOS).items()},
            },
            problems=problems,
        )
        manifest.write_text(
            json.dumps(dataset.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        return dataset

    # ------------------------------------------------------------------ #
    # Reading
    # ------------------------------------------------------------------ #
    def get(self, name: str, version: str) -> DatasetVersion:
        manifest = self.manifest_path(name, version)
        if not manifest.is_file():
            raise FileNotFoundError(f"dataset {name}@{version} not found at {manifest}")
        return DatasetVersion.from_dict(json.loads(manifest.read_text(encoding="utf-8")))

    def list_versions(self, name: str | None = None) -> list[DatasetVersion]:
        """Every dataset version on disk, newest name/version last."""
        found: list[DatasetVersion] = []
        if not self.root.is_dir():
            return found
        for manifest in sorted(self.root.glob(f"*/*/{MANIFEST_FILE}")):
            try:
                found.append(
                    DatasetVersion.from_dict(json.loads(manifest.read_text(encoding="utf-8")))
                )
            except (json.JSONDecodeError, ValueError):
                # A corrupt manifest is skipped, never guessed at.
                continue
        if name is not None:
            found = [item for item in found if item.name == name]
        return found

    def latest(self, name: str) -> DatasetVersion | None:
        versions = self.list_versions(name)
        if not versions:
            return None
        return sorted(versions, key=lambda item: item.created_at)[-1]

    def load_records(
        self, name: str, version: str, split: str = SPLIT_TRAIN
    ) -> list[DatasetRecord]:
        path = self.split_path(name, version, split)
        if not path.is_file():
            raise FileNotFoundError(f"no {split} split for {name}@{version}: {path}")
        return read_jsonl(path)

    def stats(self) -> dict[str, Any]:
        """Summary of every dataset on disk (names, versions, records)."""
        versions = self.list_versions()
        names: dict[str, dict[str, Any]] = {}
        for item in versions:
            entry = names.setdefault(item.name, {"versions": 0, "records": 0, "problems": 0})
            entry["versions"] += 1
            entry["records"] += item.total
            entry["problems"] += len(item.problems)
        return {
            "root": str(self.root),
            "datasets": len(names),
            "versions": len(versions),
            "records": sum(item.total for item in versions),
            "by_name": names,
        }


def _safe(name: str) -> str:
    """Sanitise a dataset/version name for use as a directory name."""
    cleaned = "".join(
        character if character.isalnum() or character in "-_." else "_"
        for character in (name or "").strip().lower()
    )
    if not cleaned or cleaned in {".", ".."}:
        raise ValueError(f"invalid dataset name/version: {name!r}")
    return cleaned


def _validate(records: list[DatasetRecord]) -> list[str]:
    """Return a problem per invalid record (empty list = valid)."""
    problems: list[str] = []
    for index, record in enumerate(records):
        for problem in record.problems():
            problems.append(f"record {index}: {problem}")
    return problems


__all__ = [
    "DatasetManager",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VALID",
]
