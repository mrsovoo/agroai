"""Video inspection - container facts plus a frame-sampling plan.

The roadmap's video path is:

```text
Video -> Frame selection -> Important frames -> Vision analysis -> Temporal
analysis -> Knowledge retrieval -> Agent
```

Selecting frames needs the clip's duration, which for MP4/MOV lives in the
``mvhd`` box **outside the video payload** - so it can be read exactly, right
now, without demuxing a single frame. This module does that, and for other
containers (AVI, WebM/MKV) it reports the container honestly and marks the
duration unknown rather than inventing one.

Frame sampling is then pure arithmetic
(:func:`~agroai.vision.preprocessing.plan_frames`): a handful of evenly spaced
keyframes instead of every frame.

``important frames`` in the roadmap means content-aware selection (motion,
focus, novelty). That requires a decoder and a model, and is **not**
implemented: :func:`select_important_frames` raises instead of returning a
lineup that was never computed.
"""

from __future__ import annotations

import struct
from pathlib import Path

from agroai.vision.base import MediaInfo, MediaKind
from agroai.vision.image import MediaReadError, read_header
from agroai.vision.preprocessing import (
    DEFAULT_MAX_FRAMES,
    DEFAULT_MIN_FRAMES,
    DEFAULT_TARGET_FPS,
    FramePlan,
    plan_frames,
)

#: Containers recognised from their leading bytes.
CONTAINER_MP4 = "mp4"
CONTAINER_QUICKTIME = "mov"
CONTAINER_AVI = "avi"
CONTAINER_MKV = "matroska"
CONTAINER_WEBM = "webm"

SUPPORTED_VIDEO_CONTAINERS: tuple[str, ...] = (
    CONTAINER_MP4,
    CONTAINER_QUICKTIME,
    CONTAINER_AVI,
    CONTAINER_MKV,
    CONTAINER_WEBM,
)

#: A video header can be large (an MP4 with ``moov`` at the end is huge); read
#: enough to reach ``moov``/``mvhd`` in the common fast-start layout.
MAX_VIDEO_HEADER_BYTES = 4_194_304

_MIME_TYPES = {
    CONTAINER_MP4: "video/mp4",
    CONTAINER_QUICKTIME: "video/quicktime",
    CONTAINER_AVI: "video/x-msvideo",
    CONTAINER_MKV: "video/x-matroska",
    CONTAINER_WEBM: "video/webm",
}

_EBML_MAGIC = b"\x1a\x45\xdf\xa3"


def detect_video_container(data: bytes) -> str:
    """Return the video container name, or ``'unknown'``."""
    if len(data) >= 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"qt  ",):
            return CONTAINER_QUICKTIME
        return CONTAINER_MP4
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"AVI ":
        return CONTAINER_AVI
    if data.startswith(_EBML_MAGIC):
        # Matroska and WebM share the EBML header; the DocType string says which.
        lowered = data[:4096].lower()
        if b"webm" in lowered:
            return CONTAINER_WEBM
        return CONTAINER_MKV
    return "unknown"


def _iter_boxes(data: bytes, start: int = 0, end: int | None = None):
    """Yield ``(type, payload_start, payload_end)`` for top-level MP4 boxes."""
    end = len(data) if end is None else min(end, len(data))
    index = start
    while index + 8 <= end:
        size = struct.unpack(">I", data[index : index + 4])[0]
        box_type = data[index + 4 : index + 8]
        header = 8
        if size == 1:  # 64-bit size
            if index + 16 > end:
                return
            size = struct.unpack(">Q", data[index + 8 : index + 16])[0]
            header = 16
        elif size == 0:  # extends to the end of the file
            size = end - index
        if size < header or index + size > end:
            return
        yield box_type, index + header, index + size
        index += size


def _mp4_duration(data: bytes) -> float | None:
    """Read duration from ``moov/mvhd`` (seconds), or ``None`` if unavailable."""
    for box_type, payload_start, payload_end in _iter_boxes(data):
        if box_type != b"moov":
            continue
        for child_type, child_start, child_end in _iter_boxes(data, payload_start, payload_end):
            if child_type != b"mvhd":
                continue
            if child_start + 4 > child_end:
                return None
            version = data[child_start]
            try:
                if version == 1:
                    timescale = struct.unpack(">I", data[child_start + 20 : child_start + 24])[0]
                    duration = struct.unpack(">Q", data[child_start + 24 : child_start + 32])[0]
                else:
                    timescale = struct.unpack(">I", data[child_start + 12 : child_start + 16])[0]
                    duration = struct.unpack(">I", data[child_start + 16 : child_start + 20])[0]
            except struct.error:
                return None
            if not timescale:
                return None
            return round(duration / timescale, 3)
    return None


def inspect_video(
    path: Path | str, *, size: int = MAX_VIDEO_HEADER_BYTES
) -> MediaInfo:
    """Read ``path`` and return byte-derived facts about the video container."""
    path = Path(path)
    data = read_header(path, size)
    container = detect_video_container(data)

    info = MediaInfo(
        path=str(path),
        kind=MediaKind.VIDEO,
        format=container,
        mime_type=_MIME_TYPES.get(container, ""),
        size_bytes=path.stat().st_size,
        inspected=container != "unknown",
    )
    if container == "unknown":
        info.notes.append(
            "Unrecognised video container; AgroAI will not treat this file as a video."
        )
        return info

    if container in (CONTAINER_MP4, CONTAINER_QUICKTIME):
        duration = _mp4_duration(data)
        info.duration_seconds = duration
        if duration is None:
            info.notes.append(
                "Container recognised but the 'moov'/'mvhd' box was not found in the "
                "read window (the file may be non-fast-start); duration unknown."
            )
    else:
        info.notes.append(
            f"{container} duration is not read yet (a demuxer is required); "
            "frame planning is unavailable for this file."
        )
    return info


def plan_video_frames(
    info: MediaInfo,
    *,
    target_fps: float = DEFAULT_TARGET_FPS,
    max_frames: int = DEFAULT_MAX_FRAMES,
    min_frames: int = DEFAULT_MIN_FRAMES,
) -> FramePlan:
    """Plan frame sampling for an inspected video.

    Returns an *unplanned* :class:`FramePlan` when the duration is unknown or the
    file is not a video - never a fabricated timestamp list.
    """
    if info.kind is not MediaKind.VIDEO:
        return FramePlan(planned=False, note="Not a video file; nothing to sample.")
    if not info.inspected:
        return FramePlan(planned=False, note="Container not recognised.")
    return plan_frames(
        info.duration_seconds,
        target_fps=target_fps,
        max_frames=max_frames,
        min_frames=min_frames,
    )


def select_important_frames(*args: object, **kwargs: object) -> None:
    """Not implemented: content-aware frame selection needs a decoder + model.

    Raises :class:`NotImplementedError`. Returning a "selection" that was never
    computed would make every downstream temporal claim meaningless.
    """
    raise NotImplementedError(
        "Content-aware frame selection is not implemented in this phase: it "
        "requires a video decoder plus a model to score motion/novelty. "
        "Use plan_video_frames() for the evenly sampled plan."
    )


def is_video(path: Path | str) -> bool:
    """True when the file's leading bytes are a known video container."""
    try:
        return detect_video_container(read_header(path)) != "unknown"
    except MediaReadError:
        return False


__all__ = [
    "CONTAINER_AVI",
    "CONTAINER_MKV",
    "CONTAINER_MP4",
    "CONTAINER_QUICKTIME",
    "CONTAINER_WEBM",
    "MAX_VIDEO_HEADER_BYTES",
    "MediaReadError",
    "SUPPORTED_VIDEO_CONTAINERS",
    "detect_video_container",
    "inspect_video",
    "is_video",
    "plan_video_frames",
    "select_important_frames",
]
