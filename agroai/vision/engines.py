"""Vision engine implementations.

Two engines exist, and neither pretends to see:

* :class:`UnconfiguredVisionEngine` - the default. It still does the real work
  that is possible (file inspection, resize planning, frame planning) and returns
  ``NOT_CONFIGURED`` **with an empty observations list**. An empty list is the
  honest signal downstream code checks.
* :class:`PlaceholderVisionEngine` - answers for a provider name that is
  configured but not built into this distribution, with ``NOT_IMPLEMENTED`` and
  a precise explanation. It never raises from the analysis path, because a
  product endpoint must degrade, not crash.

When the real model arrives it implements :class:`~agroai.vision.base.VisionEngine`
and replaces these; nothing else in the codebase changes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from agroai.config.settings import Settings, get_settings
from agroai.vision.base import (
    VISION_ERROR,
    VISION_NOT_CONFIGURED,
    VISION_NOT_IMPLEMENTED,
    MediaInfo,
    MediaKind,
    VisionEngine,
    VisionResult,
)
from agroai.vision.image import MediaReadError, inspect_image
from agroai.vision.preprocessing import plan_frames, plan_resize
from agroai.vision.video import inspect_video

VISION_PROVIDER_NONE = "none"

DEFAULT_REASON = (
    "No vision model is configured. AgroAI inspected the file itself (format, "
    "resolution, size) but cannot describe its content: image understanding is "
    "not part of this build. Set AGROAI_VISION_PROVIDER and a model path when "
    "the multimodal phase is deployed."
)


class UnconfiguredVisionEngine(VisionEngine):
    """The honest default: real inspection, zero content claims."""

    name = VISION_PROVIDER_NONE

    def __init__(self, reason: str = DEFAULT_REASON, *, target_latency_seconds: float = 30.0):
        self._reason = reason
        self.target_latency_seconds = target_latency_seconds

    @property
    def is_configured(self) -> bool:
        return False

    def analyze_image(
        self, path: str, *, prompt: str = "", context: dict[str, Any] | None = None
    ) -> VisionResult:
        media, error = _inspect(path, MediaKind.IMAGE)
        if error:
            return VisionResult(status=VISION_ERROR, engine=self.name, detail=error)
        resize = plan_resize(media.width, media.height) if media else None
        detail = self._reason
        if resize is not None:
            detail += (
                f"\nPreprocessing plan: resize to {resize.to_dict()['target']}."
                if resize.resized
                else f"\nPreprocessing plan: {resize.note} ({resize.to_dict()['target']})"
            )
        return VisionResult(
            status=VISION_NOT_CONFIGURED,
            engine=self.name,
            media=media,
            observations=[],
            detail=detail,
        )

    def analyze_video(
        self, path: str, *, prompt: str = "", context: dict[str, Any] | None = None
    ) -> VisionResult:
        media, error = _inspect(path, MediaKind.VIDEO)
        if error:
            return VisionResult(status=VISION_ERROR, engine=self.name, detail=error)
        detail = self._reason
        if media is not None:
            plan = plan_frames(media.duration_seconds)
            if plan.planned:
                detail += f"\nFrame plan: {plan.frame_count} frame(s) at ~{plan.fps} fps."
            else:
                detail += f"\nFrame plan: unavailable ({plan.note})"
        return VisionResult(
            status=VISION_NOT_CONFIGURED,
            engine=self.name,
            media=media,
            observations=[],
            detail=detail,
        )

    def health_check(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "configured": False,
            "available": False,
            "detail": self._reason,
        }


class PlaceholderVisionEngine(VisionEngine):
    """Answers for a configured-but-unbuilt provider (e.g. ``clip``, ``siglip``)."""

    def __init__(self, provider: str, *, target_latency_seconds: float = 30.0):
        self.name = provider
        self.target_latency_seconds = target_latency_seconds
        self._reason = (
            f"Vision provider '{provider}' is not implemented in this build. "
            "No observations were produced; configure an implemented provider or "
            "wait for the multimodal phase."
        )

    @property
    def is_configured(self) -> bool:
        return False

    def analyze_image(
        self, path: str, *, prompt: str = "", context: dict[str, Any] | None = None
    ) -> VisionResult:
        media, error = _inspect(path, MediaKind.IMAGE)
        status = VISION_ERROR if error else VISION_NOT_IMPLEMENTED
        detail = error or self._reason
        return VisionResult(
            status=status, engine=self.name, media=media, observations=[], detail=detail
        )

    def analyze_video(
        self, path: str, *, prompt: str = "", context: dict[str, Any] | None = None
    ) -> VisionResult:
        media, error = _inspect(path, MediaKind.VIDEO)
        status = VISION_ERROR if error else VISION_NOT_IMPLEMENTED
        detail = error or self._reason
        return VisionResult(
            status=status, engine=self.name, media=media, observations=[], detail=detail
        )

    def health_check(self) -> dict[str, Any]:
        return {
            "engine": self.name,
            "configured": False,
            "available": False,
            "detail": self._reason,
        }


def _inspect(path: str, kind: MediaKind) -> tuple[MediaInfo | None, str]:
    """Inspect a file, converting read failures into an error string."""
    try:
        if kind is MediaKind.IMAGE:
            return inspect_image(Path(path)), ""
        return inspect_video(Path(path)), ""
    except MediaReadError as exc:
        return None, str(exc)


def get_vision_engine(settings: Settings | None = None) -> VisionEngine:
    """Return the configured vision engine.

    Only ``none`` (and an unset provider) map to the real default. Any other name
    yields a placeholder that reports ``NOT_IMPLEMENTED`` - never a silent
    pretend-model.
    """
    settings = settings or get_settings()
    provider = (getattr(settings, "vision_provider", VISION_PROVIDER_NONE) or "").strip().lower()
    target = float(getattr(settings, "vision_target_latency_seconds", 30.0) or 30.0)
    if provider in {"", VISION_PROVIDER_NONE, "disabled", "off"}:
        return UnconfiguredVisionEngine(target_latency_seconds=target)
    return PlaceholderVisionEngine(provider, target_latency_seconds=target)


__all__ = [
    "DEFAULT_REASON",
    "PlaceholderVisionEngine",
    "UnconfiguredVisionEngine",
    "VISION_PROVIDER_NONE",
    "get_vision_engine",
]
