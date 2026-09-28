# AgroAI

**AgroAI** — qishloq xo'jaligi va veterinariya uchun offline-first AI agent.
An offline-first agricultural and veterinary AI agent that runs entirely on your
machine: local knowledge base, local retrieval, local tools, local LLM.

> **Phase 1 — Foundation (architecture).** Every layer of the roadmap exists as
> real, tested code: the agent engine (planning + execution), the ingestion
> pipeline with provenance and trust levels, lexical retrieval, an in-memory
> vector store, the tool set, the safety layer, the model abstraction and the
> trusted-source registry. The capabilities that are **not** implemented — local
> inference, embeddings, PDF/HTML extraction, automatic source updates, model
> training — say so explicitly and refuse. Nothing is faked, in code or in this
> file. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full map.

## What is AgroAI?

AgroAI answers agricultural and veterinary questions in **Uzbek** (plus Russian
and English) by *planning* how to answer, retrieving relevant passages from a
**local knowledge base**, running deterministic **tools** where arithmetic or
classification is needed, asking a **local LLM** to compose an answer grounded in
those passages, and then **validating** that answer against its sources.

Domain focus: crops, plant diseases, pests, plant protection, livestock,
veterinary medicine, animal diseases, poultry, animal nutrition, fertilizers,
pesticides, fungicides, herbicides, soil, irrigation, greenhouse production,
agricultural machinery, agricultural technology and Uzbekistan agriculture.

### Design principles

* **Never fabricate.** No fake AI answers, no invented sources or citations, no
  fake embeddings, no fake PDF extraction, no reported training that never ran.
  Every capability is either real or explicitly reported as not configured.
* **Grounded answers.** The system prioritizes retrieved knowledge and says so
  explicitly when the knowledge base lacks verified information.
* **Provenance by construction.** Documents carry source, URL, author, date,
  license, region and a **trust level** (tier 1/2/3). Citations can only be built
  from stored documents, so an answer cannot cite something that does not exist.
* **Agricultural safety.** Guardrails classify each question before answering;
  dosages, veterinary medicine and agrochemicals require sources and professional
  verification. After generation, the answer is validated for invented links,
  unsupported quantities, overconfidence and grounding coverage.
* **Offline-first.** Startup, search, ingestion, tools, tests and doctor all work
  with no network. The core has zero third-party dependencies.

## Architecture

```text
Question
   │
   ▼
Agent (agroai/agent)
   ├─ planner      explicit plan: safety → retrieve → tools → generate → validate
   ├─ executor     runs the steps, records what ran and what was skipped
   └─ memory       bounded, in-process conversation context
   │
   ├──► Safety guardrails (agroai/safety/guardrails)   risk category + rules
   ├──► Retriever (agroai/retrieval)                   SQLite FTS5 (+ vector path)
   ├──► Tools (agroai/tools)                           calculators, classifiers
   ├──► Local LLM (agroai/models/inference)            provider contract (no engine yet)
   └──► Answer validation (agroai/safety/validation)   citations, doses, coverage
   │
   ▼
Answer + sources + disclaimers
```

Knowledge layer: `agroai/knowledge` (documents, chunks, provenance metadata,
citations, manager) over `agroai/database` (SQLite + FTS5).
Ingestion: `agroai/ingestion` (8-stage pipeline; TXT/Markdown implemented).
Sources: `agroai/sources` (curated trusted registry; update interface that
refuses to touch the network).

### Package layout

```text
agroai/
├── agent/         agent.py, planner.py, executor.py, memory.py, prompts.py
├── models/        base.py, loader.py, inference.py, fine_tuning/
├── knowledge/     documents.py, metadata.py, sources.py, manager.py
├── ingestion/     base.py, text.py, pdf.py, html.py, pipeline.py
├── retrieval/     retriever.py, embeddings.py, vector_store.py
├── tools/         base.py, calculator.py, soil.py, irrigation.py, weather.py, veterinary.py
├── safety/        guardrails.py, validation.py
├── sources/       registry.py, updater.py
├── database/      database.py
├── config/        settings.py
└── cli/           main.py, doctor.py

knowledge/    the knowledge base (by category)
models/       local model files (never committed; see models/README.md)
datasets/     exported training data (see datasets/README.md)
data/         database, indexes, collected documents (git-ignored)
docs/         architecture documentation
scripts/      reset_database.sh, export_dataset.py
tests/        290 offline tests
```

The LLM is **replaceable**: the agent only talks to the `LLMProvider` interface
(`generate()`, `chat()`, `health_check()`), and `LocalLLMProvider` only produces
text through an `InferenceEngine` — the single place a real runtime (Ollama,
llama.cpp, GGUF) plugs in later.

## Installation

Requires Python **3.10+**.

```bash
git clone <this-repo> agroai
cd agroai
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
agroai --help
```

`pip install -e .` installs the `agroai` console command. Nothing else is
required — the core uses only the Python standard library.

## Configuration

Configuration is environment-based (no secrets in source, no API keys).
Copy `.env.example` to `.env` and edit, or export `AGROAI_*` variables:

```bash
cp .env.example .env
```

Key variables (all optional — defaults are safe for local development):

| Variable | Default | Purpose |
| --- | --- | --- |
| `AGROAI_LLM_PROVIDER` | `local` | Provider: `local` (Ollama/llama.cpp/GGUF later) |
| `AGROAI_LOCAL_MODEL` | *(empty)* | Model name/alias for a local engine |
| `AGROAI_MODEL_PATH` | *(empty)* | Path to a local GGUF model file |
| `AGROAI_MODELS_DIR` | `models` | Where model files are discovered |
| `AGROAI_DATASETS_DIR` | `datasets` | Where datasets are exported |
| `AGROAI_EMBEDDING_MODEL` | *(empty)* | Future embedding model |
| `AGROAI_DATABASE_PATH` | `data/database/agroai.db` | SQLite database file |
| `AGROAI_VECTOR_DB_PATH` | `data/indexes` | Future vector index directory |
| `AGROAI_KNOWLEDGE_DIR` | `knowledge` | Knowledge base root |
| `AGROAI_RETRIEVAL_BACKEND` | `sqlite` | `sqlite` now; `vector` needs embeddings |
| `AGROAI_VECTOR_STORE_BACKEND` | `none` | `none` or `memory` (real in-memory store) |
| `AGROAI_SAFETY_STRICT` | `true` | Raise agrochemical questions to high risk |
| `AGROAI_UPDATER_PROVIDER` | `none` | Only `none` exists — no network |
| `AGROAI_LOG_LEVEL` | `INFO` | Logging level |

Never create a real `.env` with secrets; AgroAI needs none.

## CLI commands

```bash
agroai                          # interactive session
agroai ask "savol?"             # plan + retrieval-grounded Q&A
agroai plan "savol?"            # show the plan (safety/retrieval/tools) only
agroai search "pomidor"         # search the local knowledge base
agroai ingest ./doc.md          # ingest one document
agroai ingest-dir ./knowledge   # ingest a directory tree
agroai sources                  # list ingested sources
agroai sources --trusted        # list the curated trusted-source registry
agroai tools                    # list available tools
agroai doctor                   # environment self-check
agroai version                  # print version
agroai --help                   # help
```

Interactive mode supports `/help`, `/sources`, `/plan`, `/tools`, `/clear`,
`/doctor`, `/exit`.

### First run

```bash
agroai ingest-dir ./knowledge   # load the seed knowledge
agroai search "pomidor kasalliklari"
agroai plan "Sirdaryoda tuproq sho'rlangan, EC 6.5 dS/m. Nima qilay?"
agroai ask "Pomidor barglarida sariq dog'lar paydo bo'ldi. Nima bo'lishi mumkin?"
agroai doctor
```

Without a configured LLM, `ask` still runs the plan and the tools, shows the
retrieved knowledge-base matches and clearly reports that generation is not
configured — it never invents an answer.

## Knowledge base

The `knowledge/` directory is organized by category:

```text
knowledge/
├── crops/            # ekinlar
├── plant_diseases/   # o'simlik kasalliklari
├── pests/            # zararkunandalar
├── livestock/        # chorva mollar
├── veterinary/       # veterinariya
├── poultry/          # parrandachilik
├── medicines/        # dorilar
├── fertilizers/      # o'g'itlar
├── soil/             # tuproq
├── irrigation/       # sug'orish
├── greenhouse/       # issiqxona
├── machinery/        # qishloq xo'jaligi texnikasi
└── uzbekistan/       # O'zbekiston qishloq xo'jaligi
```

The category is assigned automatically from the parent directory at ingest
time — or declared explicitly in a document's front matter.

## Document ingestion

Implemented today: **TXT** and **Markdown**. PDF, HTML, JSON and CSV are
registered with honest interfaces that report "not implemented" — no fake
extraction.

The pipeline runs eight stages, each either doing real work or refusing:

```text
load → extract → clean → detect language → metadata → deduplicate → chunk → store
```

* **Cleaning** normalizes newlines, trailing whitespace and control characters.
* **Language detection** is a script/lexicon heuristic (uz / ru / en) that falls
  back to the configured default when the evidence is weak.
* **Metadata** builds the provenance record, including the trust level.
* **Deduplication** is by SHA-256 content hash, so re-ingesting an unchanged file
  is skipped (`--overwrite` forces it).

### Declaring provenance

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

The block is optional, must be the first thing in the file, and only recognized
keys are kept.

### Trust levels

| Tier | Sources | Treatment |
| --- | --- | --- |
| **1** | FAO, WOAH, USDA, government bodies, universities, peer-reviewed research | authoritative, quotable |
| **2** | professional/extension organizations, research centres | reliable, usually secondary |
| **3** | everything else, including blogs, forums, social posts | never treated as fact automatically |

The level is inferred from the curated registry (or a declared `source_type`) —
never invented by a model.

## Retrieval

`Retriever.search(query, top_k)` is independent of any LLM. Two real backends:

* **`SQLiteRetriever`** (default) — SQLite FTS5 lexical search (BM25 ranking),
  with an automatic `LIKE` fallback when FTS5 is unavailable. This is honest
  keyword search, not semantic retrieval.
* **`VectorRetriever`** — semantic search over a vector store. Its machinery is
  real and tested (`InMemoryVectorStore` does exact cosine search with the
  standard library, `VectorIndexer` embeds and indexes chunks), but it stays
  **unready** until an embedding provider is configured, and reports exactly
  what is missing instead of returning invented matches.

Configured third-party vector backends (faiss, qdrant, chroma, sqlite-vec) are
reported as not implemented rather than silently substituted.

## RAG and the agent engine

`agroai ask` runs: question → plan → run the plan → answer + sources.

The planner (`agroai/agent/planner.py`) is rule-based, deterministic and
inspectable — `agroai plan "<question>"` prints exactly what it decided:

1. **assess safety** — risk category, whether sources are mandatory;
2. **retrieve knowledge** — an answer without sources is not acceptable;
3. **run tools** — only when every parameter can be named (a missing number
   means the step is not planned, never guessed);
4. **generate** — grounded-only prompt with context, tool output and safety rules;
5. **validate** — the answer is checked against the sources it was allowed to use.

The executor records a `status` for every step, so a failed retrieval, a refused
tool or a skipped generation is visible rather than hidden.

The prompt instructs the model to ground answers in the retrieved context, to
state plainly when the knowledge base lacks verified information, to cite only
retrieved sources, and to recommend professional verification for veterinary or
chemical topics. Sources are never invented: they come from stored documents.

## Safety

`agroai/safety` implements both checkpoints:

* **Guardrails (before).** Classifies the question into `dosage`,
  `veterinary_medicine`, `food_safety`, `agrochemical`, `plant_protection` or
  `general`, with risk level low/medium/high, mandatory grounding, disclaimers
  and extra prompt rules.
* **Validation (after).** Reports invented URLs, quantities/dosages that appear
  in no source, overconfidence phrasing, grounding coverage, and high-risk
  questions answered without sources.

If a high-risk answer came back ungrounded, the agent keeps the answer, sets
`safety_blocked` and attaches a safety notice; critical validation findings are
appended as warnings. Nothing is silently rewritten and nothing is invented.

## Tools

| Tool | What it does |
| --- | --- |
| `calculator` | safe arithmetic (AST-based, no `eval`) |
| `unit_converter` | area, mass, volume, length and concentration units |
| `soil_tool` | salinity (ECe) and pH classes, FAO leaching requirement |
| `irrigation_tool` | ETc = ET0 × Kc, irrigation volumes, drip runtime |
| `weather_tool` | spray-window check from *your* readings; live weather refuses |
| `veterinary_tool` | structures a case into an observation checklist; no diagnosis, no dosage |

Tools never invent an input, reference a method (not a fake URL), and add
"verify locally / the label wins" notes wherever a real decision depends on it.
List them with `agroai tools`.

## Local LLM

`agroai/models/base.py` defines the provider contract; `inference.py` implements
`LocalLLMProvider`, which reports **NOT CONFIGURED** until a model is declared
and **`no inference engine available`** when a model file exists but no runtime
is wired up. It never generates placeholder text.

`agroai/models/loader.py` really inspects model files (GGUF magic bytes, size,
base-model family from the file name), and `agroai doctor` reports what it found.

To integrate a real backend later, implement `InferenceEngine` and return it
from `agroai.models.inference.get_inference_engine`.

## Fine-tuning and datasets

`agroai/models/fine_tuning` provides the real parts of the model phase:

* `DatasetRecord` / `FineTuningDataset` — a JSONL schema where every record keeps
  its source (an unattributed record cannot even be loaded);
* `grounded_records()` — extractive, source-grounded records built from the
  knowledge base (no synthesis, so no hallucinated training data):
  `python scripts/export_dataset.py`;
* `LoRAConfig` — a validated configuration schema;
* `LoRATrainer.train()` and `build_instruction_dataset()` — **refuse**, with an
  explanation. AgroAI Core ships no training runtime and will not report a run
  that never happened.

The architectural rule from the roadmap holds: the knowledge base is **not**
stuffed into the model. Fine-tuning will teach terminology, style and reasoning
patterns; the facts stay in the retrievable knowledge base.

## Trusted sources and updates

`agroai/sources/registry.py` maintains a metadata registry of trusted origins
(FAO, USDA, WOAH, Uzbekistan agricultural and veterinary authorities, agrarian
universities, peer-reviewed research). `agroai sources --trusted` lists it.

`agroai/sources/updater.py` defines how the future update phase must behave
(opt-in, incremental, deduplicated by hash, provenance-preserving, network only
with an explicitly configured provider). Today it returns
`status=not_implemented, network_used=False`. **Nothing is scraped or
downloaded.**

## Doctor

`agroai doctor` verifies: Python version, configuration, knowledge directories,
database, retrieval, LLM configuration, model files, embeddings, vector store,
tools, safety and the knowledge updater — reporting each as `OK`, `WARNING`,
`NOT CONFIGURED` or `ERROR` based on what was actually verified in that run.

## Testing

```bash
python -m pytest          # or: pytest
```

290 offline tests cover configuration, provenance metadata, front matter, trust
levels, knowledge models, hashing, chunking, the ingestion pipeline (cleaning,
language detection, deduplication, storage), the knowledge manager, the SQLite
layer, the vector store and indexer, lexical and vector retrieval, embeddings
honesty, the model loader, the LLM/engine honesty, fine-tuning boundaries, the
tool set, safety guardrails, answer validation, planning, execution, the agent
flow, memory, sources, the updater, doctor and the CLI. No network and no paid
APIs are required.

## Development

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e .[dev]
pytest
bash scripts/reset_database.sh          # reset DB + re-ingest seed knowledge
python scripts/export_dataset.py        # export grounded dataset records
```

Code quality: Python 3.10+, type hints, dataclasses, `pathlib`, clean module
boundaries, meaningful exceptions, logging, minimal dependencies (stdlib-only
core).

## Storage & scalability

Everything is designed to scale from thousands to millions of chunks: the
SQLite database stores documents, chunks and the FTS index separately from the
original files; reads are lazy cursors, writes are batched, and nothing loads
the whole knowledge base into RAM. Original documents stay separable from
normalized/index data (`data/documents`, `data/indexes`).

## Offline-first

All of this works without internet: start AgroAI, inspect configuration, search
local knowledge, ingest local documents, run the tools, run tests, run doctor.
Internet-dependent functionality (weather, trusted-source updates) is optional,
isolated behind an interface, and refused by default.

## Roadmap

* **Phase 1 (this release):** architecture in full — config, CLI, ingestion
  pipeline, knowledge base, retrieval, tools, safety, agent engine (planning +
  execution), model abstraction, sources, doctor, tests.
* **Phase 2:** large trusted agricultural knowledge collection.
* **Phase 3:** embeddings + vector database indexing (the vector path is ready).
* **Phase 4:** local LLM engine (Ollama / llama.cpp / GGUF) behind
  `InferenceEngine`.
* **Phase 5:** RAG optimization and hybrid lexical/semantic retrieval.
* **Phase 6:** automatic trusted-source updates (opt-in, verified, incremental).
* **Phase 7:** advanced agricultural and veterinary reasoning.
* **Phase 8:** optional GUI / mobile / web interfaces.

## Project audit (2026-09-28)

To'liq loyiha auditi natijasi (read-only tekshiruv asosida):

### Umumiy holat

* **Faza 1 poydevori deyarli to'liq qurilgan:** 88 paket fayli (~19 000 qator) + 25 test fayli.
* **Testlar: 290/290 o'tadi** (0.71 s, oflayn). `python -m compileall` — xatosiz.
* Core umumiy kutubxonalarsiz (faqat stdlib), tarmoqqa ulanmaydi.
* Muhit: Python 3.12.3, `.venv` ichida editable install, `pytest 9.1.1`.
* **Git repository mavjud emas** (`git status` → "not a git repository") — tarix saqlanmagan.
* Lokal LLM, embeddinglar va vision model **sozlanmagan** (va bu halol tarzda "NOT CONFIGURED" deb xisobot qilinadi — hech narsa soxta emas).

### Ishlayotgan qismlar (sinf: COMPLETE)

| Qism | Dalil |
| --- | --- |
| Konfiguratsiya (`config/settings.py`) | `agroai doctor` to'liq konfigni ko'rsatadi |
| Ingestion pipeline (TXT/MD, 8 bosqich) | dedup sha-256, til aniqlash (uz/ru/en), ishonch darajalari (tier 1/2/3) |
| SQLite + FTS5 baza | 12 jadval, `data/database/agroai.db` |
| Leksik qidiruv | `agroai search "pomidor"` → reytingli natijalar |
| Agent (planner/executor/memory) | `agroai plan` — 5 bosqichli deterministik reja |
| Vositalar (6 ta) | calculator, unit_converter, soil, irrigation, weather, veterinary |
| Xavfsizlik (guardrails + validation) | dosaj → high risk, manba majburiy |
| Skills router (15), differensial tashxis | eval smoke suite: 7 pass, 1 halol skip |
| HTTP API (13 route) + kvotalar | `agroai api` → HTTP 200, status "degraded" (LLM yo'q) |
| Doctor / eval / verify / vision-inspect | barchasi exit 0 |

### Mavjud emas (halol chegara — NOT CONFIGURED / NOT IMPLEMENTED)

* Lokal LLM xulosasi (`InferenceEngine` ulanmagan) — `agroai ask` javob o'rniga KB topilmlarini ko'rsatadi.
* Embeddinglar + vektor qidiruv (mashina tayyor, provayder yo'q).
* PDF/HTML ajratib olish (`pypdf` o'rnatilmagan; interfeys halol rad etadi).
* Vision model, video dekodlash, tashqi qidiruv, avtomatik manba yangilanishlari.
* Telegram bot, admin panel — umuman yo'q.
* Trening runtime (`LoRATrainer.train()` atayin `TrainingNotAvailable`).

### Buzilgan buyruqlar (3 ta CLI xatosi)

| Buyruq | Xato | Sabab | Joy |
| --- | --- | --- | --- |
| `agroai models` | `ImportError: circular import` | `datasets.manager` ↔ `models.fine_tuning.manager` o'zaro import | `agroai/datasets/manager.py:44`, `agroai/models/fine_tuning/manager.py:35` |
| `agroai datasets` | xuddi shu circular import | xuddi shu tsikl | xuddi shu joylar |
| `agroai plans` | `AttributeError: 'Namespace' object has no attribute 'user'` | `plans` subparserida `--user` argumenti e'lon qilinmagan | `agroai/cli/main.py:560` (arg: `:813`) |

### Bilim pipeline holati

```text
[COMPLETE]   Documents → Parser → Cleaning → Chunking → Metadata
[NOT IMPL.]  Embeddings → Vector DB
[COMPLETE]   Retrieval (SQLite FTS5)
```

Bilimlar bazasi mazmuni: **3 ta seed hujjat** (10/13 kategoriya bo'sh) — Faza 2 kutilmoqda.

### Birinchi navbatdagi tuzatishlar

1. **P0:** `datasets ↔ models.fine_tuning` circular import ini uzish (`agroai models`/`datasets` ochiladi).
2. **P0:** `plans` subparseriga `--user` qo'shish.
3. **P0:** Git repozitoriyni ishga tushirish va boshlang'ich commit.
4. **P1:** Har bir subcommand uchun CLI smoke-test (exit kodlari) + README dagi eski test sonini yangilash (237 → 290).
5. **P2:** Haqiqiy bilim kontenti, `pypdf`, `InferenceEngine` ulash (Ollama/llama.cpp).
6. **P3:** Embeddinglar, Telegram bot, admin UI, trening runtime.

## License

MIT
