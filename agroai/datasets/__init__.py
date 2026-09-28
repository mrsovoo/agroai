"""Dataset management: versioned, split, provenance-carrying training data.

```text
raw data -> clean -> verified -> labeled -> train -> validation -> test
                                              -> AgroAI dataset v1
```

The roadmap asks for dataset management with its own versioning. What that means
concretely here:

* :class:`~agroai.datasets.base.DatasetVersion` - a name+version with its splits,
  sources and recorded problems;
* :mod:`agroai.datasets.splits` - **deterministic** train/valid/test assignment
  (hash buckets, duplicates kept together, leakage detected);
* :class:`~agroai.datasets.manager.DatasetManager` - writes
  ``train.jsonl``/``valid.jsonl``/``test.jsonl`` + ``manifest.json`` under
  ``datasets/<name>/<version>/`` and refuses to write invalid records.

The record schema itself lives in
:mod:`agroai.models.fine_tuning.dataset`; this package organises records, it does
not invent them. Which dataset a model was trained on is then recorded in
:class:`agroai.models.registry.ModelRecord`.
"""

from __future__ import annotations

from agroai.datasets.base import (
    DEFAULT_SPLIT_RATIOS,
    MANIFEST_FILE,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALID,
    SPLITS,
    DatasetVersion,
)
from agroai.datasets.manager import DatasetManager
from agroai.datasets.splits import (
    assign_split,
    bucket_of,
    find_leakage,
    record_key,
    split_keys,
    split_records,
)

__all__ = [
    "DEFAULT_SPLIT_RATIOS",
    "DatasetManager",
    "DatasetVersion",
    "MANIFEST_FILE",
    "SPLITS",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VALID",
    "assign_split",
    "bucket_of",
    "find_leakage",
    "record_key",
    "split_keys",
    "split_records",
]
