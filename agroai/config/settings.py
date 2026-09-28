"""Application settings.

Configuration is resolved from, in order of precedence:

1. Explicit environment variables (``AGROAI_*``).
2. A ``.env`` file at the project root (simple parser, no hard dependency).
3. Built-in offline-safe defaults.

No API key is ever required, and none is hardcoded anywhere in the project.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

_ENV_PREFIX = "AGROAI_"

# Knowledge categories the project understands (used for validation + CLI).
KNOWLEDGE_CATEGORIES: tuple[str, ...] = (
    "crops",
    "plant_diseases",
    "pests",
    "livestock",
    "veterinary",
    "poultry",
    "medicines",
    "fertilizers",
    "soil",
    "irrigation",
    "greenhouse",
    "machinery",
    "uzbekistan",
)


def _detect_project_root() -> Path:
    """Find the project root (the directory holding ``pyproject.toml``)."""
    env_home = os.environ.get(f"{_ENV_PREFIX}HOME") or os.environ.get("AGROAI_HOME")
    if env_home:
        return Path(env_home).expanduser().resolve()

    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "pyproject.toml").exists():
            return parent
    # Fallback: agroai/config/settings.py -> parents[2] == project root.
    return here.parents[2]


def parse_env_file(path: Path) -> dict[str, str]:
    """Parse a minimal ``.env`` file.

    Supports ``KEY=value``, optional ``export`` prefix, ``#`` comments and
    quoted values. Deliberately dependency-free so the core stays offline.
    """
    values: dict[str, str] = {}
    if not path.is_file():
        return values

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
            value = value[1:-1]
        if key:
            values[key] = value
    return values


def _env_source(project_root: Path) -> dict[str, str]:
    """Merge ``.env`` values with the real environment (env wins)."""
    merged = parse_env_file(project_root / ".env")
    for key, value in os.environ.items():
        if key.startswith(_ENV_PREFIX):
            merged[key] = value
    return merged


def _get(source: dict[str, str], name: str, default: str = "") -> str:
    return source.get(f"{_ENV_PREFIX}{name}", source.get(name, default))


def _get_bool(source: dict[str, str], name: str, default: bool) -> bool:
    raw = _get(source, name, "")
    if raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(source: dict[str, str], name: str, default: int) -> int:
    raw = _get(source, name, "")
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _get_float(source: dict[str, str], name: str, default: float) -> float:
    raw = _get(source, name, "")
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def _resolve_path(base: Path, value: str, default: str) -> Path:
    chosen = value or default
    path = Path(chosen).expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


@dataclass
class Settings:
    """Resolved, immutable-ish application configuration."""

    project_root: Path
    knowledge_dir: Path
    data_dir: Path
    database_path: Path
    vector_db_path: Path
    documents_dir: Path
    indexes_dir: Path
    models_dir: Path
    datasets_dir: Path

    log_level: str = "INFO"
    default_language: str = "uz"

    #: Safety guardrails: strict mode raises the risk level of agrochemical
    #: questions, because an unverified application rate is the costliest kind
    #: of wrong answer in this domain.
    safety_strict: bool = True
    #: Automatic knowledge updates: only 'none' exists (no network use).
    updater_provider: str = "none"

    llm_provider: str = "local"
    llm_model: str = ""
    local_model_path: str = ""
    llm_endpoint: str = "http://127.0.0.1:11434"
    llm_max_tokens: int = 512
    llm_temperature: float = 0.2

    embedding_provider: str = "none"
    embedding_model_path: str = ""
    embedding_dimensions: int = 384

    retrieval_backend: str = "sqlite"
    vector_store_backend: str = "none"

    chunk_size: int = 800
    chunk_overlap: int = 120
    deduplicate: bool = True

    #: Vision / multimodal. ``none`` means no vision model: files are inspected
    #: (format, resolution, duration) but never described.
    vision_provider: str = "none"
    vision_model_path: str = ""
    #: Performance target for one image analysis, not a guarantee.
    vision_target_latency_seconds: float = 30.0
    vision_max_side: int = 1024
    video_max_frames: int = 12

    #: Trusted external search. Off by default and disabled until a provider is
    #: explicitly implemented - AgroAI Core never opens a network connection.
    external_search_provider: str = "none"
    external_search_enabled: bool = False

    #: Quotas are enforced server-side (never trusted from a client counter).
    default_plan: str = "free"
    enforce_quotas: bool = True

    #: API keys as ``key:user:plan`` entries, comma separated. Empty means local
    #: development mode (no key required). Production must set both of these.
    api_keys: str = ""
    require_api_key: bool = False

    #: Remembering farm context across sessions requires explicit consent.
    memory_consent: bool = False

    #: --- Product layer (API / Web / Telegram) -------------------------- #
    #: HTTP host/port the local API server binds to (agroai.api.main).
    api_host: str = "127.0.0.1"
    api_port: int = 8000
    #: Origins allowed by the local API's CORS (comma separated; * allows all).
    #: Only the Web client needs this during development.
    cors_origins: str = "http://localhost:3000,http://127.0.0.1:3000"
    #: Telegram bot: token read from the environment ONLY (never committed,
    #: never logged). Empty means the bot is not configured.
    telegram_bot_token: str = ""
    telegram_mode: str = "polling"
    telegram_enabled: bool = False
    #: Where Web/Telegram clients find the API.
    web_url: str = "http://localhost:3000"
    api_url: str = "http://127.0.0.1:8000"
    #: Super admin panel password (HTTP /admin). Empty = panel not configured.
    admin_password: str = ""

    categories: tuple[str, ...] = field(default=KNOWLEDGE_CATEGORIES)

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(cls, project_root: Path | None = None) -> "Settings":
        root = (project_root or _detect_project_root()).resolve()
        src = _env_source(root)

        knowledge_dir = _resolve_path(root, _get(src, "KNOWLEDGE_DIR"), "knowledge")
        data_dir = _resolve_path(root, "", "data")
        documents_dir = _resolve_path(root, _get(src, "DOCUMENTS_DIR"), "data/documents")
        indexes_dir = _resolve_path(root, _get(src, "VECTOR_DB_PATH"), "data/indexes")
        database_path = _resolve_path(root, _get(src, "DATABASE_PATH"), "data/database/agroai.db")
        models_dir = _resolve_path(root, _get(src, "MODELS_DIR"), "models")
        datasets_dir = _resolve_path(root, _get(src, "DATASETS_DIR"), "datasets")

        # Alias handling: the spec uses AGROAI_LOCAL_MODEL (name),
        # AGROAI_MODEL_PATH (file) and AGROAI_EMBEDDING_MODEL alongside the
        # internal names; all resolve to the same settings fields.
        llm_model = _get(src, "LLM_MODEL", "") or _get(src, "LOCAL_MODEL", "")
        local_model_path = _get(src, "LOCAL_MODEL_PATH", "") or _get(src, "MODEL_PATH", "")
        embedding_model = _get(src, "EMBEDDING_MODEL_PATH", "") or _get(src, "EMBEDDING_MODEL", "")

        return cls(
            project_root=root,
            knowledge_dir=knowledge_dir,
            data_dir=data_dir,
            database_path=database_path,
            vector_db_path=indexes_dir,
            documents_dir=documents_dir,
            indexes_dir=indexes_dir,
            models_dir=models_dir,
            datasets_dir=datasets_dir,
            log_level=_get(src, "LOG_LEVEL", "INFO").upper(),
            default_language=_get(src, "DEFAULT_LANGUAGE", "uz").lower(),
            safety_strict=_get_bool(src, "SAFETY_STRICT", True),
            updater_provider=_get(src, "UPDATER_PROVIDER", "none").lower(),
            llm_provider=_get(src, "LLM_PROVIDER", "local").lower(),
            llm_model=llm_model,
            local_model_path=local_model_path,
            llm_endpoint=_get(src, "LLM_ENDPOINT", "http://127.0.0.1:11434"),
            llm_max_tokens=_get_int(src, "LLM_MAX_TOKENS", 512),
            llm_temperature=_get_float(src, "LLM_TEMPERATURE", 0.2),
            embedding_provider=_get(src, "EMBEDDING_PROVIDER", "none").lower(),
            embedding_model_path=embedding_model,
            embedding_dimensions=_get_int(src, "EMBEDDING_DIMENSIONS", 384),
            retrieval_backend=_get(src, "RETRIEVAL_BACKEND", "sqlite").lower(),
            vector_store_backend=_get(src, "VECTOR_STORE_BACKEND", "none").lower(),
            chunk_size=_get_int(src, "CHUNK_SIZE", 800),
            chunk_overlap=_get_int(src, "CHUNK_OVERLAP", 120),
            deduplicate=_get_bool(src, "DEDUPLICATE", True),
            vision_provider=_get(src, "VISION_PROVIDER", "none").lower(),
            vision_model_path=_get(src, "VISION_MODEL_PATH", "")
            or _get(src, "VISION_MODEL", ""),
            vision_target_latency_seconds=_get_float(src, "VISION_TARGET_SECONDS", 30.0),
            vision_max_side=_get_int(src, "VISION_MAX_SIDE", 1024),
            video_max_frames=_get_int(src, "VIDEO_MAX_FRAMES", 12),
            external_search_provider=_get(src, "EXTERNAL_SEARCH_PROVIDER", "none").lower(),
            external_search_enabled=_get_bool(src, "EXTERNAL_SEARCH_ENABLED", False),
            default_plan=_get(src, "DEFAULT_PLAN", "free").lower(),
            enforce_quotas=_get_bool(src, "ENFORCE_QUOTAS", True),
            api_keys=_get(src, "API_KEYS", ""),
            require_api_key=_get_bool(src, "REQUIRE_API_KEY", False),
            memory_consent=_get_bool(src, "MEMORY_CONSENT", False),
            api_host=_get(src, "API_HOST", "127.0.0.1"),
            api_port=_get_int(src, "API_PORT", 8000),
            cors_origins=_get(src, "CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"),
            telegram_bot_token=_get(src, "TELEGRAM_BOT_TOKEN", ""),
            telegram_mode=_get(src, "TELEGRAM_MODE", "polling").lower(),
            telegram_enabled=_get_bool(src, "TELEGRAM_ENABLED", False),
            web_url=_get(src, "WEB_URL", "http://localhost:3000"),
            api_url=_get(src, "API_URL", "http://127.0.0.1:8000"),
            admin_password=_get(src, "ADMIN_PASSWORD", ""),
        )

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def ensure_directories(self) -> None:
        """Create the runtime directories if they do not exist yet."""
        for path in (
            self.knowledge_dir,
            self.data_dir,
            self.documents_dir,
            self.indexes_dir,
            self.database_path.parent,
            self.models_dir,
            self.datasets_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def category_directory(self, category: str) -> Path:
        return self.knowledge_dir / category

    def as_public_dict(self) -> dict[str, object]:
        """Return settings safe to print (contains no secrets by design)."""
        return {
            "project_root": str(self.project_root),
            "knowledge_dir": str(self.knowledge_dir),
            "data_dir": str(self.data_dir),
            "database_path": str(self.database_path),
            "vector_db_path": str(self.vector_db_path),
            "documents_dir": str(self.documents_dir),
            "models_dir": str(self.models_dir),
            "datasets_dir": str(self.datasets_dir),
            "log_level": self.log_level,
            "default_language": self.default_language,
            "safety_strict": self.safety_strict,
            "updater_provider": self.updater_provider,
            "llm_provider": self.llm_provider,
            "llm_model": self.llm_model or "(not configured)",
            "local_model_path": self.local_model_path or "(not configured)",
            "llm_endpoint": self.llm_endpoint,
            "embedding_provider": self.embedding_provider,
            "embedding_model_path": self.embedding_model_path or "(not configured)",
            "retrieval_backend": self.retrieval_backend,
            "vector_store_backend": self.vector_store_backend,
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "deduplicate": self.deduplicate,
            "vision_provider": self.vision_provider,
            "vision_model_path": self.vision_model_path or "(not configured)",
            "vision_target_latency_seconds": self.vision_target_latency_seconds,
            "vision_max_side": self.vision_max_side,
            "video_max_frames": self.video_max_frames,
            "external_search_provider": self.external_search_provider,
            "external_search_enabled": self.external_search_enabled,
            "default_plan": self.default_plan,
            "enforce_quotas": self.enforce_quotas,
            "require_api_key": self.require_api_key,
            "api_keys": f"{len(self.api_key_map())} configured" if self.api_keys else "(none - local mode)",
            "memory_consent": self.memory_consent,
            "api_host": self.api_host,
            "api_port": self.api_port,
            "cors_origins": self.cors_origins or "(none)",
            "telegram_token": "CONFIGURED" if self.telegram_bot_token else "(not configured)",
            "telegram_mode": self.telegram_mode,
            "telegram_enabled": self.telegram_enabled,
            "web_url": self.web_url,
            "api_url": self.api_url,
        }

    def api_key_map(self) -> dict[str, tuple[str, str]]:
        """Parse ``AGROAI_API_KEYS`` into ``key -> (user_id, plan)``.

        Format: ``key:user:plan`` entries separated by commas. Malformed entries
        are skipped rather than guessed, and no key is ever logged.
        """
        mapping: dict[str, tuple[str, str]] = {}
        for entry in (self.api_keys or "").split(","):
            entry = entry.strip()
            if not entry:
                continue
            parts = [part.strip() for part in entry.split(":")]
            if len(parts) < 2 or not parts[0] or not parts[1]:
                continue
            plan = parts[2] if len(parts) > 2 and parts[2] else ""
            mapping[parts[0]] = (parts[1], plan)
        return mapping


@lru_cache(maxsize=1)
def _cached_settings() -> Settings:
    return Settings.from_env()


def get_settings() -> Settings:
    """Return the process-wide cached settings singleton."""
    return _cached_settings()


def reset_settings_cache() -> None:
    """Clear the cached settings (used by tests and after config changes)."""
    _cached_settings.cache_clear()
