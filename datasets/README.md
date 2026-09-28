# `datasets/` — training data for the AgroAI model

Datasets are **JSONL**: one JSON object per line, one record per line of
training data. Every record keeps the source it came from — an unattributed
record cannot even be loaded, because a model that learns facts without
provenance cannot be audited later.

```json
{"instruction": "Quyidagi manbaga asoslanib javob bering...", "input": "", "output": "Tuproq sho'rlanishi ECe bilan o'lchanadi...", "source": "FAO", "kind": "grounded_passage", "category": "soil", "language": "uz", "trust_level": "tier_1", "created_at": "2026-01-01T00:00:00+00:00", "metadata": {"document_id": "doc_x", "title": "..."}}
```

## What works today

Grounded, **extractive** records can be built from the knowledge base with no
model involved: the passage genuinely exists in the knowledge base and the
citation is real, so nothing can be hallucinated.

```bash
python scripts/export_dataset.py                        # -> datasets/agroai_grounded.jsonl
python scripts/export_dataset.py --out datasets/subset.jsonl --limit 500
```

The library equivalent:

```python
from agroai.models.fine_tuning import FineTuningDataset, summarize

dataset = FineTuningDataset.from_documents(documents)
print(summarize(dataset.records))
```

## What does not work yet

* **Question and reasoning synthesis** (`build_instruction_dataset`) requires a
  generator model plus human review. It raises `NotImplementedError` instead of
  emitting templated filler.
* **Training.** `LoRATrainer.train()` refuses; see `models/README.md`.

## Record kinds

| `kind` | Meaning | Available |
| --- | --- | --- |
| `grounded_passage` | extracted passage + its citation | yes |
| `instruction` | question/answer pairs written by (or reviewed by) a human | later phase |
| `reasoning` | diagnosis/recommendation traces with sources | later phase |
| `terminology` | domain glossary entries | later phase |
