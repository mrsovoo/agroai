"""Vision contracts: what AgroAI can honestly say about a picture right now.

The multimodal phase is part of the product, so the architecture must exist from
day one. But there is no vision model in this build, and a module that pretends
to recognise "early blight" from bytes would be the single most dangerous thing
in the codebase.

So this package draws a hard line:

* **real, implemented today** - file-level inspection: format, dimensions,
  resolution, size, video container and duration. Read from the actual bytes
  (:mod:`agroai.vision.image`, :mod:`agroai.vision.video`).
* **real, implemented today** - the analysis *pipeline*: inspect -> ask the
  knowledge base what the described symptoms could mean -> report limitations
  (:mod:`agroai.vision.analysis`).
* **contract only** - :class:`VisionEngine`, the interface a real
  plant/animal vision model plugs into. The shipped engine is
  :class:`~agroai.vision.engines.UnconfiguredVisionEngine`: it reports
  ``NOT_CONFIGURED`` and returns **no** observations, so nothing downstream can
  mistake a missing model for a finding.

Vision never asserts a diagnosis either: its output feeds the differential
diagnosis engine as *evidence*, never as a verdict.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

#: Vision result statuses (mirrors the LLM response statuses on purpose).
VISION_OK = "ok"
VISION_NOT_CONFIGURED = "not_configured"
VISION_NOT_IMPLEMENTED = "not_implemented"
VISION_ERROR = "error"


class MediaKind(str, Enum):
    """Whether the input is a still image or a video."""

    IMAGE = "image"
    VIDEO = "video"
    UNKNOWN = "unknown"


@dataclass
class MediaInfo:
    """Objective, byte-derived facts about a media file.

    Every field is read from the file itself - nothing here is a model output.
    ``width``/``height``/``duration_seconds`` are ``None`` when the format is
    recognised but the value could not be parsed; the inspector says which in
    ``notes`` rather than guessing.
    """

    path: str
    kind: MediaKind
    format: str = "unknown"
    mime_type: str = ""
    size_bytes: int = 0
    width: int | None = None
    height: int | None = None
    duration_seconds: float | None = None
    frame_count: int | None = None
    inspected: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def resolution(self) -> str:
        if self.width and self.height:
            return f"{self.width}x{self.height}"
        return "unknown"

    @property
    def megapixels(self) -> float | None:
        if not self.width or not self.height:
            return None
        return round(self.width * self.height / 1_000_000, 2)

    @property
    def aspect_ratio(self) -> float | None:
        if not self.width or not self.height:
            return None
        return round(self.width / self.height, 3)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "kind": self.kind.value,
            "format": self.format,
            "mime_type": self.mime_type,
            "size_bytes": self.size_bytes,
            "width": self.width,
            "height": self.height,
            "resolution": self.resolution,
            "megapixels": self.megapixels,
            "aspect_ratio": self.aspect_ratio,
            "duration_seconds": self.duration_seconds,
            "frame_count": self.frame_count,
            "inspected": self.inspected,
            "notes": list(self.notes),
        }

    def describe(self) -> str:
        """One-line summary of the objective facts."""
        parts = [f"{self.kind.value}: {self.format}"]
        if self.width and self.height:
            parts.append(self.resolution)
            if self.megapixels:
                parts.append(f"{self.megapixels} MP")
        parts.append(f"{self.size_bytes} bytes")
        if self.duration_seconds is not None:
            parts.append(f"{self.duration_seconds:.1f} s")
        return ", ".join(parts)


@dataclass
class VisionObservation:
    """One visual finding proposed by a vision model.

    ``needs_review`` exists because a vision model's output is *evidence*, not a
    diagnosis: low-confidence findings must be routed to a human or to the
    differential engine, never printed as a conclusion.
    """

    label: str
    confidence: float = 0.0
    detail: str = ""
    source: str = ""
    needs_review: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "detail": self.detail,
            "source": self.source,
            "needs_review": self.needs_review,
        }


@dataclass
class VisionResult:
    """Uniform envelope for a vision call (same shape as ``LLMResponse``)."""

    status: str = VISION_NOT_CONFIGURED
    media: MediaInfo | None = None
    observations: list[VisionObservation] = field(default_factory=list)
    engine: str = ""
    detail: str = ""
    usage: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.status == VISION_OK

    @property
    def configured(self) -> bool:
        return self.status != VISION_NOT_CONFIGURED

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "engine": self.engine,
            "detail": self.detail,
            "media": self.media.to_dict() if self.media is not None else None,
            "observations": [item.to_dict() for item in self.observations],
            "usage": dict(self.usage),
        }


class VisionEngine(ABC):
    """Interface a real plant/animal/video vision model must implement.

    Contract:

    * ``is_configured`` is True only when a real model is loaded and usable;
    * an unconfigured engine returns ``VISION_NOT_CONFIGURED`` with an empty
      ``observations`` list - never a plausible-looking guess;
    * every observation carries a confidence and defaults to ``needs_review``;
    * ``analyze_*`` must not raise for an ordinary bad input: it returns an
      ``ERROR`` status with the reason.
    """

    name: str = "base"
    #: Performance target, not a guarantee (documented in docs/PLATFORM.md).
    target_latency_seconds: float = 30.0

    @property
    @abstractmethod
    def is_configured(self) -> bool:
        """True only when a real vision model is available for inference."""

    @abstractmethod
    def analyze_image(
        self, path: str, *, prompt: str = "", context: dict[str, Any] | None = None
    ) -> VisionResult:
        """Analyse one still image."""

    @abstractmethod
    def analyze_video(
        self, path: str, *, prompt: str = "", context: dict[str, Any] | None = None
    ) -> VisionResult:
        """Analyse a video clip (frame selection is the engine's job)."""

    @abstractmethod
    def health_check(self) -> dict[str, Any]:
        """Report whether the engine is configured and reachable."""

    def describe(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "configured": self.is_configured,
            "target_latency_seconds": self.target_latency_seconds,
        }

    @staticmethod
    def not_configured(reason: str, engine: str = "base") -> VisionResult:
        return VisionResult(status=VISION_NOT_CONFIGURED, engine=engine, detail=reason)


__all__ = [
    "MediaInfo",
    "MediaKind",
    "VISION_ERROR",
    "VISION_NOT_CONFIGURED",
    "VISION_NOT_IMPLEMENTED",
    "VISION_OK",
    "VisionEngine",
    "VisionObservation",
    "VisionResult",
]
