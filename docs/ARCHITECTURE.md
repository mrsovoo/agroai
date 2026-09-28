# AgroAI architecture

How the roadmap fits together, module by module — and, just as important, what
is **real today** versus what is an honest, non-functional boundary.

```text
                            USER QUESTION
                                 │
                                 ▼
  ┌──────────────────────  AGROAI AGENT  ──────────────────────┐
  │                                                            │
  │  planner.py            explicit plan (auditable, offline)  │
  │    1. assess_safety    ──► safety/guardrails.py            │
  │    2. retrieve         ──► retrieval/retriever.py          │
  │    3. run_tool         ──► tools/                          │
  │    4. generate         ──► models/inference.py             │
  │    5. validate         ──► safety/validation.py            │
  │                                                            │
  │  executor.py           runs those steps, records what ran  │
  │  agent.py              presents answer + sources + caveats │
  └────────────────────────────────────────────────────────────┘
                                 │
                                 ▼
              knowledge base (SQLite FTS5)  +  tools  +  safety
                                 │
                                 ▼
                        ANSWER + SOURCES + VALIDATION
```

## Package layout (roadmap → code)

| Roadmap area | Module | State |
| --- | --- | --- |
| Agent engine | `agroai/agent/agent.py` | **real** — orchestrates the whole pipeline |
| Planning | `agroai/agent/planner.py` | **real** — rule-based, deterministic, inspectable |
| Execution | `agroai/agent/executor.py` | **real** — step outcomes, graceful degradation |
| Memory | `agroai/agent/memory.py` | **real** — bounded, in-process (persistent memory: later) |
| Prompts / context | `agroai/agent/prompts.py` | **real** — grounding + tool + safety blocks |
| Base model / LLM layer | `agroai/models/base.py` | **real** — the `LLMProvider` contract |
| Model loading | `agroai/models/loader.py` | **real** — model families, GGUF magic, discovery |
| Inference | `agroai/models/inference.py` | **interface only** — no engine installed |
| Fine-tuning | `agroai/models/fine_tuning/` | **dataset real**, training refuses |
| Knowledge models | `agroai/knowledge/documents.py` | **real** — documents, chunks, chunking |
| Provenance metadata | `agroai/knowledge/metadata.py` | **real** — hashing, trust tiers, front matter |
| Sources & citations | `agroai/knowledge/sources.py` | **real** — source records, citations |
| Knowledge manager | `agroai/knowledge/manager.py` | **real** — facade over the pipeline + DB |
| Ingestion | `agroai/ingestion/` | **TXT/Markdown real**; PDF/HTML/JSON/CSV honest stubs |
| Ingestion pipeline | `agroai/ingestion/pipeline.py` | **real** — 8 explicit stages |
| Embeddings | `agroai/retrieval/embeddings.py` | **interface only** — no provider |
| Vector store | `agroai/retrieval/vector_store.py` | **in-memory store real**, backends later |
| Retrieval | `agroai/retrieval/retriever.py` | **real** — FTS5/BM25 + a real vector path |
| Tools | `agroai/tools/` | **real** — calculators and classifiers; network tools refuse |
| Safety | `agroai/safety/` | **real** — guardrails before, validation after |
| Trusted sources | `agroai/sources/registry.py` | **real** — curated metadata, offline matching |
| Knowledge updates | `agroai/sources/updater.py` | **interface only** — no network, ever |
| Database | `agroai/database/database.py` | **real** — SQLite, FTS5, streaming reads |
| Configuration | `agroai/config/settings.py` | **real** — env-based, no secrets |

## The ingestion pipeline

Eight stages, in order. Each one either does real work or refuses — nothing
guesses.

```text
load ──► extract ──► clean ──► detect_language ──► metadata ──► deduplicate ──► chunk ──► store
 │         │           │              │                │             │            │        │
extension  reader   normalize     uz/ru/en        DocumentMetadata  SHA-256   provenance  SQLite
  →        (TXT/MD   whitespace   script+lexicon   + trust level    content    preserved   (+FTS)
reader     only)                   heuristic       from registry     hash
```

* **Deduplication** is by content hash, so the same FAO document served from ten
  places is stored once.
* **Trust level** (tier 1/2/3) is inferred from the curated registry, or declared
  in a document's front matter — never invented.
* **Language** is detected heuristically and falls back to the configured
  default when the evidence is weak, because a wrong label is worse than none.

### Declaring provenance in a document

```markdown
---
title: Tuproq sho'rlanishi
organization: FAO
source_type: government
source_url: https://www.fao.org/...
publication_date: 2024/3/5
region: Sirdaryo
country: UZ
license: CC-BY
---

Body text...
```

The block is optional, must be the first thing in the file, and only known keys
are accepted (so a stray colon in prose cannot become provenance).

## Safety model

Two checkpoints, both deterministic and offline:

1. **Before** — `SafetyGuard.classify(question)` returns a risk level (low /
   medium / high), whether sources are mandatory, whether professional
   verification must be recommended, the disclaimers, and extra system-prompt
   rules for that category:
   `dosage`, `veterinary_medicine`, `food_safety`, `agrochemical`,
   `plant_protection`.
2. **After** — `AnswerValidator.validate(answer, chunks, verdict)` reports
   invented URLs, quantities that appear in no source, overconfidence, grounding
   coverage, and high-risk questions answered without sources.

If a high-risk answer came back ungrounded, the agent keeps the answer but sets
`safety_blocked` and attaches a safety notice; a critical validation finding is
appended as a warning. Nothing is silently rewritten.

## Tool policy

| Tool | Kind | Behaviour |
| --- | --- | --- |
| `calculator` | arithmetic | safe AST evaluation, no `eval` |
| `unit_converter` | arithmetic | area/mass/volume/length/ratios |
| `soil_tool` | classification | salinity, pH, FAO leaching requirement |
| `irrigation_tool` | arithmetic | ETc, volumes, drip runtime — inputs are yours |
| `weather_tool` | threshold check | spray windows from *your* readings; live weather refuses |
| `veterinary_tool` | structuring | observation checklist; no diagnosis, no dosage |

Design rules: a tool never invents a parameter (missing numbers mean the tool
step is not planned), never claims a source it cannot support (references are
method names, not URLs), and puts "verify locally / the label wins" in its notes
wherever a real decision depends on it.

## Honest boundaries

These are *architecturally present* but **not implemented**, and every one of
them says so rather than faking a result:

| Boundary | Reported as |
| --- | --- |
| Local inference engine | `ERROR: no inference engine available` |
| Embedding provider | `EmbeddingsNotConfigured`, `NOT CONFIGURED` in doctor |
| Vector DB backends (faiss/qdrant/chroma/sqlite-vec) | `not implemented yet`, `NOT CONFIGURED` |
| PDF/HTML/JSON/CSV ingestion | `NotImplementedError` with the planned phase |
| Automatic knowledge updates | `status=not_implemented, network_used=False` |
| Instruction/reasoning dataset synthesis | `NotImplementedError` |
| LoRA training | `status=not_implemented` |

## Offline-first

Startup, configuration, ingestion, retrieval, the tools, the safety layer and
the whole test suite run with **no network**. The core has **zero third-party
dependencies**; every future network capability (weather, source updates) is
opt-in and isolated behind an interface that currently refuses.
