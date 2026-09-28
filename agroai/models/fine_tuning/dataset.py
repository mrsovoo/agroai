"""Fine-tuning dataset: schema, IO and the honest generation boundary.

The roadmap's dataset phase has four kinds of material: instruction data, domain
terminology, reasoning examples and source-grounded QA. AgroAI can produce the
**grounded** kind itself, offline, because it is extractive: a passage plus the
citation it came from. Everything that would require *writing new text*
(questions, explanations, reasoning traces) needs a generator model and belongs
to a later phase - :func:`build_instruction_dataset` says exactly that instead of
emitting machine-written filler.

Records are stored as JSONL so they can be inspected, diffed and versioned like
any other source artifact.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable, Iterator

from agroai.knowledge.documents import KnowledgeDocument
from agroai.knowledge.metadata import TrustLevel, utcnow_iso

#: Record kinds the schema understands.
KIND_GROUNDED_PASSAGE = "grounded_passage"
KIND_INSTRUCTION = "instruction"
KIND_REASONING = "reasoning"
KIND_TERMINOLOGY = "terminology"

REQUIRED_FIELDS = ("instruction", "output", "source")


@dataclass
class DatasetRecord:
    """One training record, always carrying the source it came from."""

    instruction: str
    output: str
    source: str
    input: str = ""
    kind: str = KIND_GROUNDED_PASSAGE
    category: str = ""
    language: str = "uz"
    trust_level: str = TrustLevel.TIER_3.value
    created_at: str = field(default_factory=utcnow_iso)
    metadata: dict[str, object] = field(default_factory=dict)

    def problems(self) -> list[str]:
        found: list[str] = []
        if not self.instruction.strip():
            found.append("empty instruction")
        if not self.output.strip():
            found.append("empty output")
        if not self.source.strip():
            found.append("missing source (records must stay attributable)")
        return found

    def is_valid(self) -> bool:
        return not self.problems()

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "DatasetRecord":
        missing = [name for name in REQUIRED_FIELDS if not str(data.get(name, "")).strip()]
        if missing:
            raise ValueError(f"dataset record is missing required field(s): {', '.join(missing)}")
        known = {name for name in cls.__dataclass_fields__}
        clean = {key: value for key, value in data.items() if key in known}
        return cls(**clean)  # type: ignore[arg-type]


# ---------------------------------------------------------------------- #
# IO
# ---------------------------------------------------------------------- #
def write_jsonl(records: Iterable[DatasetRecord], path: Path | str) -> int:
    """Write records to a JSONL file. Returns the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")
            count += 1
    return count


def read_jsonl(path: Path | str) -> list[DatasetRecord]:
    """Read records back from a JSONL file (skipping blank lines).

    A malformed line raises: silently dropping training data would hide a
    corrupt dataset.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset not found: {path}")
    records: list[DatasetRecord] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(DatasetRecord.from_dict(json.loads(line)))
            except (json.JSONDecodeError, ValueError) as exc:
                raise ValueError(f"{path}:{number}: invalid dataset record: {exc}") from exc
    return records


def validate_records(records: Iterable[DatasetRecord]) -> list[str]:
    """Return a problem description per invalid record (empty list = all good)."""
    problems: list[str] = []
    for index, record in enumerate(records):
        for problem in record.problems():
            problems.append(f"record {index}: {problem}")
    return problems


def summarize(records: Iterable[DatasetRecord]) -> dict[str, object]:
    """Summarise a dataset: size, kinds, languages, categories, trust levels."""
    records = list(records)
    kinds: dict[str, int] = {}
    languages: dict[str, int] = {}
    categories: dict[str, int] = {}
    trust: dict[str, int] = {}
    for record in records:
        kinds[record.kind] = kinds.get(record.kind, 0) + 1
        languages[record.language] = languages.get(record.language, 0) + 1
        categories[record.category or "unknown"] = categories.get(record.category or "unknown", 0) + 1
        trust[record.trust_level] = trust.get(record.trust_level, 0) + 1
    return {
        "records": len(records),
        "kinds": kinds,
        "languages": languages,
        "categories": categories,
        "trust_levels": trust,
        "invalid": len(validate_records(records)),
    }


# ---------------------------------------------------------------------- #
# Building
# ---------------------------------------------------------------------- #
INSTRUCTION_TEMPLATE = (
    "Quyidagi manbaga asoslanib javob bering. Faqat manbadagi ma'lumotdan "
    "foydalaning va manbani ko'rsating."
)


def grounded_records(
    documents: Iterable[KnowledgeDocument],
    *,
    min_chars: int = 200,
    max_records: int | None = None,
) -> Iterator[DatasetRecord]:
    """Yield extractive, source-grounded records from stored documents.

    This is real data: each record pairs an instruction with a passage that
    genuinely exists in the knowledge base, plus the citation it came from. No
    text is synthesized, so nothing here can be a hallucination. Question and
    reasoning synthesis is a later phase (see :func:`build_instruction_dataset`).
    """
    produced = 0
    for document in documents:
        content = (document.content or "").strip()
        if len(content) < min_chars:
            continue
        if max_records is not None and produced >= max_records:
            return
        yield DatasetRecord(
            instruction=INSTRUCTION_TEMPLATE,
            input="",
            output=content,
            source=document.source or document.id,
            kind=KIND_GROUNDED_PASSAGE,
            category=document.category,
            language=document.language,
            trust_level=document.trust_level.value,
            metadata={
                "document_id": document.id,
                "title": document.title,
                "source_url": document.source_url,
                "region": document.region(),
                "country": document.country(),
            },
        )
        produced += 1


def build_instruction_dataset(*args: object, **kwargs: object) -> None:
    """Not implemented: requires a generator model (later dataset phase).

    Raises :class:`NotImplementedError` rather than emitting templated filler
    that would teach the model nothing true.
    """
    raise NotImplementedError(
        "Instruction/reasoning dataset synthesis is not implemented in this "
        "phase: it requires a generator model plus human review. "
        "Use grounded_records() for extractive, source-grounded records."
    )


class FineTuningDataset:
    """An in-memory collection of :class:`DatasetRecord` objects."""

    def __init__(self, records: Iterable[DatasetRecord] | None = None) -> None:
        self.records: list[DatasetRecord] = list(records or [])

    def __len__(self) -> int:
        return len(self.records)

    @classmethod
    def from_documents(cls, documents: Iterable[KnowledgeDocument], **options: object) -> "FineTuningDataset":
        return cls(grounded_records(documents, **options))  # type: ignore[arg-type]

    @classmethod
    def from_jsonl(cls, path: Path | str) -> "FineTuningDataset":
        return cls(read_jsonl(path))

    @property
    def problems(self) -> list[str]:
        return validate_records(self.records)

    def summary(self) -> dict[str, object]:
        return summarize(self.records)

    def to_jsonl(self, path: Path | str) -> int:
        return write_jsonl(self.records, path)


__all__ = [
    "KIND_GROUNDED_PASSAGE",
    "KIND_INSTRUCTION",
    "KIND_REASONING",
    "KIND_TERMINOLOGY",
    "DatasetRecord",
    "FineTuningDataset",
    "build_instruction_dataset",
    "grounded_records",
    "read_jsonl",
    "summarize",
    "validate_records",
    "write_jsonl",
]
