"""Deterministic, reproducible dataset splitting.

Two properties matter more than cleverness here:

1. **Determinism** - the same records always land in the same split, so two
   training runs are comparable at all. Assignment is a hash bucket, not a
   shuffle, so it does not depend on iteration order or a global RNG state.
2. **No leakage** - an identical (instruction, output) pair must not appear in
   two splits, or the test score measures memorisation. Identical records are
   grouped into the *same* split instead of being scattered.
"""

from __future__ import annotations

import hashlib
from typing import Iterable, Sequence

from agroai.datasets.base import (
    DEFAULT_SPLIT_RATIOS,
    SPLIT_TEST,
    SPLIT_TRAIN,
    SPLIT_VALID,
    SPLITS,
)


def record_key(instruction: str, output: str) -> str:
    """Stable key used for bucketing and duplicate detection."""
    digest = hashlib.sha256()
    digest.update((instruction or "").strip().encode("utf-8"))
    digest.update(b"\x1f")
    digest.update((output or "").strip().encode("utf-8"))
    return digest.hexdigest()


def bucket_of(key: str) -> float:
    """Map a key into ``[0, 1)`` deterministically."""
    return int(key[:12], 16) / float(16**12)


def _normalize(ratios: dict[str, float]) -> dict[str, float]:
    total = sum(ratios.values())
    if total <= 0:
        raise ValueError("split ratios must sum to a positive number")
    return {name: value / total for name, value in ratios.items()}


def assign_split(key: str, ratios: dict[str, float] | None = None) -> str:
    """Return the split name for ``key`` using cumulative ratio boundaries."""
    normalized = _normalize(ratios or DEFAULT_SPLIT_RATIOS)
    position = bucket_of(key)
    cumulative = 0.0
    for name in SPLITS:
        weight = normalized.get(name, 0.0)
        if weight <= 0:
            continue
        cumulative += weight
        if position < cumulative:
            return name
    # Floating point can leave position just below 1.0; the last non-empty split
    # takes the remainder rather than dropping the record.
    for name in reversed(SPLITS):
        if normalized.get(name, 0.0) > 0:
            return name
    raise ValueError("no split has a positive ratio")


def split_keys(
    keys: Iterable[str], ratios: dict[str, float] | None = None
) -> dict[str, list[str]]:
    """Group keys into splits, keeping duplicate keys together."""
    groups: dict[str, list[str]] = {name: [] for name in SPLITS}
    assigned: dict[str, str] = {}
    for key in keys:
        # Duplicate keys share the first assignment: no leakage across splits.
        if key not in assigned:
            assigned[key] = assign_split(key, ratios)
        groups[assigned[key]].append(key)
    return groups


def split_records(
    records: Iterable, ratios: dict[str, float] | None = None
) -> dict[str, list]:
    """Split :class:`~agroai.models.fine_tuning.dataset.DatasetRecord` objects.

    Records are matched by their (instruction, output) key, so two records with
    identical content stay in one split. Object identity is preserved: the same
    objects come out that went in.
    """
    materialized = list(records)
    keyed: dict[str, list] = {}
    order: list[str] = []
    for record in materialized:
        key = record_key(
            str(getattr(record, "instruction", "")), str(getattr(record, "output", ""))
        )
        if key not in keyed:
            keyed[key] = []
            order.append(key)
        keyed[key].append(record)

    assignment = split_keys(order, ratios)
    result: dict[str, list] = {name: [] for name in SPLITS}
    for name, keys in assignment.items():
        for key in keys:
            result[name].extend(keyed[key])
    return result


def find_leakage(splits: dict[str, Sequence]) -> list[str]:
    """Return one message per content key that appears in more than one split."""
    seen: dict[str, set[str]] = {}
    for name, records in splits.items():
        for record in records:
            key = record_key(
                str(getattr(record, "instruction", "")),
                str(getattr(record, "output", "")),
            )
            seen.setdefault(key, set()).add(name)
    return [
        f"content {key[:12]} appears in splits: {', '.join(sorted(names))}"
        for key, names in sorted(seen.items())
        if len(names) > 1
    ]


__all__ = [
    "assign_split",
    "bucket_of",
    "find_leakage",
    "record_key",
    "split_keys",
    "split_records",
    "SPLIT_TEST",
    "SPLIT_TRAIN",
    "SPLIT_VALID",
]
