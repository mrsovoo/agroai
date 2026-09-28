"""Model loading: what a local model file *is*, before anything runs it.

AgroAI does not train an LLM from scratch. The roadmap's plan is a base
open-weight model (Qwen, Llama, Gemma, ...) plus an AgroAI dataset and
LoRA/fine-tuning on top. This module is the honest, offline part of that story:

* the curated list of base-model families we target,
* real file inspection (GGUF magic bytes, size on disk),
* discovery of the model files a user has actually placed in ``models/``.

It never downloads anything and never claims a model is usable just because a
filename looks promising: format detection reads the file.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Magic bytes at the start of every GGUF file (llama.cpp's format).
GGUF_MAGIC = b"GGUF"

FORMAT_GGUF = "gguf"
FORMAT_SAFETENSORS = "safetensors"
FORMAT_UNKNOWN = "unknown"


class ModelNotFoundError(RuntimeError):
    """Raised when a model file is referenced but does not exist."""


@dataclass(frozen=True)
class ModelFamily:
    """A base-model family AgroAI is designed to work with."""

    key: str
    display_name: str
    vendor: str
    notes: str


#: Curated base-model families (metadata only; nothing is downloaded).
MODEL_FAMILIES: tuple[ModelFamily, ...] = (
    ModelFamily(
        "qwen",
        "Qwen",
        "Alibaba",
        "Strong multilingual coverage; good default for Uzbek/Russian text.",
    ),
    ModelFamily(
        "llama",
        "Llama",
        "Meta",
        "Widely supported by llama.cpp/Ollama; large ecosystem.",
    ),
    ModelFamily(
        "gemma",
        "Gemma",
        "Google",
        "Compact models that fit on modest local hardware.",
    ),
    ModelFamily(
        "mistral",
        "Mistral",
        "Mistral AI",
        "Efficient instruction-tuned models.",
    ),
    ModelFamily(
        "phi",
        "Phi",
        "Microsoft",
        "Very small models for low-resource machines.",
    ),
)

#: Filename substrings used to guess a family (case-insensitive).
_FAMILY_ALIASES: dict[str, tuple[str, ...]] = {
    "qwen": ("qwen",),
    "llama": ("llama", "codellama"),
    "gemma": ("gemma",),
    "mistral": ("mistral", "mixtral"),
    "phi": ("phi-", "phi2", "phi3", "phi4", "phi_"),
}


@dataclass(frozen=True)
class ModelFile:
    """A local model file that was actually inspected on disk."""

    path: str
    size_bytes: int
    format: str
    family: str | None = None

    @property
    def name(self) -> str:
        return Path(self.path).name

    @property
    def size_gb(self) -> float:
        return round(self.size_bytes / (1024 ** 3), 2)

    @property
    def is_supported_format(self) -> bool:
        """Only formats AgroAI can actually load today count as supported."""
        return self.format == FORMAT_GGUF

    def to_dict(self) -> dict[str, object]:
        return {
            "path": self.path,
            "name": self.name,
            "size_bytes": self.size_bytes,
            "size_gb": self.size_gb,
            "format": self.format,
            "family": self.family or "(unknown)",
            "supported": self.is_supported_format,
        }


def get_family(key: str) -> ModelFamily | None:
    """Look up a base-model family by key (``qwen``, ``llama``, ...)."""
    key = (key or "").strip().lower()
    for family in MODEL_FAMILIES:
        if family.key == key:
            return family
    return None


def guess_model_family(filename: str) -> str | None:
    """Guess the base-model family from a filename (``None`` if unclear)."""
    lowered = (filename or "").lower()
    for family_key, aliases in _FAMILY_ALIASES.items():
        if any(alias in lowered for alias in aliases):
            return family_key
    return None


def detect_model_format(path: Path) -> str:
    """Detect the on-disk format of a model file by reading its first bytes."""
    path = Path(path)
    try:
        with path.open("rb") as handle:
            head = handle.read(16)
    except OSError:
        return FORMAT_UNKNOWN

    if head[:4] == GGUF_MAGIC:
        return FORMAT_GGUF
    # safetensors: 8-byte little-endian header length, then a JSON header.
    if len(head) == 16 and head[8:9] == b"{":
        return FORMAT_SAFETENSORS
    return FORMAT_UNKNOWN


def inspect_model_file(path: Path | str) -> ModelFile:
    """Inspect one model file. Raises :class:`ModelNotFoundError` if absent."""
    path = Path(path).expanduser()
    if not path.is_file():
        raise ModelNotFoundError(f"Model file not found: {path}")
    return ModelFile(
        path=str(path.resolve()),
        size_bytes=path.stat().st_size,
        format=detect_model_format(path),
        family=guess_model_family(path.name),
    )


def find_model_files(models_dir: Path | str, *, recursive: bool = False) -> list[ModelFile]:
    """Find model files under ``models_dir`` (sorted, inspected, never guessed).

    A directory that does not exist is an empty result, not an error: a fresh
    checkout has no models yet.
    """
    root = Path(models_dir).expanduser()
    if not root.is_dir():
        return []
    pattern = "**/*" if recursive else "*"
    found: list[ModelFile] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file() or path.name.startswith("."):
            continue
        if detect_model_format(path) == FORMAT_UNKNOWN:
            continue
        found.append(
            ModelFile(
                path=str(path.resolve()),
                size_bytes=path.stat().st_size,
                format=detect_model_format(path),
                family=guess_model_family(path.name),
            )
        )
    return found


__all__ = [
    "FORMAT_GGUF",
    "FORMAT_SAFETENSORS",
    "FORMAT_UNKNOWN",
    "GGUF_MAGIC",
    "MODEL_FAMILIES",
    "ModelFamily",
    "ModelFile",
    "ModelNotFoundError",
    "detect_model_format",
    "find_model_files",
    "get_family",
    "guess_model_family",
    "inspect_model_file",
]
