"""Preprocessing *planning* - the arithmetic part, with no image library.

The roadmap's latency budget is:

```text
Image -> Resize -> Preprocess -> Vision inference -> Parallel retrieval -> Answer
```

Resizing and frame selection are geometry and scheduling, and both can be
computed exactly without decoding a single pixel. That is what this module does:
it turns a :class:`~agroai.vision.base.MediaInfo` into a concrete plan (target
dimensions, downscale factor, frame timestamps) that a real preprocessing
implementation executes.

Nothing here fabricates a resized image: no pixels are touched, so nothing can
be silently mis-processed. A future engine calls these helpers, then does the
actual decode with its own library.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: Default longest-side cap for a vision request. Models are trained on
#: hundreds, not thousands, of pixels; sending more costs latency for no gain.
DEFAULT_MAX_SIDE = 1024
#: Hard cap so a phone panorama cannot blow the memory budget.
HARD_MAX_SIDE = 2048
#: Frame-sampling defaults for video (important frames, not every frame).
DEFAULT_TARGET_FPS = 1.0
DEFAULT_MAX_FRAMES = 12
DEFAULT_MIN_FRAMES = 3


@dataclass
class ResizePlan:
    """The exact geometry a resize step should apply."""

    source_width: int
    source_height: int
    target_width: int
    target_height: int
    resized: bool
    downscale_factor: float
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "source": f"{self.source_width}x{self.source_height}",
            "target": f"{self.target_width}x{self.target_height}",
            "resized": self.resized,
            "downscale_factor": round(self.downscale_factor, 4),
            "note": self.note,
        }


@dataclass
class FramePlan:
    """Which frames to sample from a video, and why."""

    planned: bool
    fps: float = 0.0
    frame_count: int = 0
    timestamps: list[float] = field(default_factory=list)
    note: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "planned": self.planned,
            "fps": self.fps,
            "frame_count": self.frame_count,
            "timestamps": [round(value, 3) for value in self.timestamps],
            "note": self.note,
        }


def resize_dimensions(
    width: int, height: int, *, max_side: int = DEFAULT_MAX_SIDE
) -> tuple[int, int]:
    """Return the target size preserving aspect ratio (never upscales)."""
    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    cap = max(64, min(int(max_side), HARD_MAX_SIDE))
    longest = max(width, height)
    if longest <= cap:
        return width, height
    scale = cap / longest
    return max(1, round(width * scale)), max(1, round(height * scale))


def plan_resize(
    width: int | None,
    height: int | None,
    *,
    max_side: int = DEFAULT_MAX_SIDE,
) -> ResizePlan | None:
    """Build a :class:`ResizePlan`, or ``None`` when dimensions are unknown.

    Returning ``None`` (rather than a made-up size) keeps the caller honest: an
    image whose dimensions could not be parsed needs a real decoder first.
    """
    if not width or not height:
        return None
    target_width, target_height = resize_dimensions(width, height, max_side=max_side)
    if (target_width, target_height) == (width, height):
        return ResizePlan(
            source_width=width,
            source_height=height,
            target_width=target_width,
            target_height=target_height,
            resized=False,
            downscale_factor=1.0,
            note="Already within the size budget; no resize needed.",
        )
    return ResizePlan(
        source_width=width,
        source_height=height,
        target_width=target_width,
        target_height=target_height,
        resized=True,
        downscale_factor=target_width / width,
        note=f"Downscale longest side to {max_side}px before inference.",
    )


def plan_frames(
    duration_seconds: float | None,
    *,
    target_fps: float = DEFAULT_TARGET_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    min_frames: int = DEFAULT_MIN_FRAMES,
) -> FramePlan:
    """Choose which timestamps to sample from a video.

    Strategy: ``target_fps`` samples per second, clamped to
    ``[min_frames, max_frames]``, spread evenly so short and long clips both get
    usable coverage. When the duration is unknown the plan is *unplanned* - a
    caller must decode the container first instead of guessing timestamps.
    """
    if duration_seconds is None:
        return FramePlan(
            planned=False,
            note=(
                "Video duration could not be read from the container; sample "
                "frames with a real demuxer instead of guessing timestamps."
            ),
        )
    if duration_seconds <= 0:
        return FramePlan(planned=False, note="Video duration is zero; nothing to sample.")

    max_frames = max(1, min(int(max_frames), DEFAULT_MAX_FRAMES))
    min_frames = max(1, min(int(min_frames), max_frames))

    wanted = round(duration_seconds * max(0.01, target_fps))
    count = max(min_frames, min(wanted or min_frames, max_frames))
    count = min(count, max_frames)

    # Even spacing, always keeping the first sample just inside the clip.
    step = duration_seconds / count
    timestamps = [round(step * (index + 0.5), 3) for index in range(count)]
    return FramePlan(
        planned=True,
        fps=round(count / duration_seconds, 4),
        frame_count=count,
        timestamps=timestamps,
        note=(
            f"{count} frame(s) sampled at ~{count / duration_seconds:.2f} fps "
            "instead of decoding every frame."
        ),
    )


__all__ = [
    "DEFAULT_MAX_FRAMES",
    "DEFAULT_MAX_SIDE",
    "DEFAULT_MIN_FRAMES",
    "DEFAULT_TARGET_FPS",
    "HARD_MAX_SIDE",
    "FramePlan",
    "ResizePlan",
    "plan_frames",
    "plan_resize",
    "resize_dimensions",
]
