# `models/` — local model files

This directory is where **base model files** live (typically GGUF, the format
`llama.cpp`/Ollama use). AgroAI never downloads anything here and never commits
model weights: the files are large, and they are yours.

```text
models/
├── qwen2.5-7b-instruct-q4_k_m.gguf      # example: base model
└── agroai-lora/                          # example: future adapter output
```

## What works today

* `agroai doctor` **inspects** this directory for real: it reads the magic bytes
  of each file (so a random file named `model.gguf` is reported as
  `format=unknown`, not as a model), reports the detected format, the guessed
  base-model family and the size on disk.
* `agroai.models.loader` exposes that inspection (`find_model_files`,
  `inspect_model_file`, `detect_model_format`, `guess_model_family`).

## What does not work yet

* **Inference.** No engine is wired up, so requesting an answer with a model
  configured returns an explicit error (`no inference engine available`) instead
  of invented text.
* **Training.** `agroai.models.fine_tuning` has a validated configuration
  schema and a real dataset format, but `LoRATrainer.train()` refuses rather
  than report a run that never happened.

Wiring a real engine means implementing `InferenceEngine` and returning it from
`agroai.models.inference.get_inference_engine` — the agent, CLI, safety layer
and tests all route through that single function.

## Configuration

| Variable | Purpose |
| --- | --- |
| `AGROAI_MODELS_DIR` | where model files are looked for (default: `models/`) |
| `AGROAI_MODEL_PATH` / `AGROAI_LOCAL_MODEL_PATH` | path to one specific model file |
| `AGROAI_LOCAL_MODEL` / `AGROAI_LLM_MODEL` | model name for a local engine (e.g. an Ollama alias) |
| `AGROAI_LLM_ENDPOINT` | base URL of a local inference server |

Nothing here is required for the knowledge base, retrieval, tools or the test
suite — those all work offline without a model.
