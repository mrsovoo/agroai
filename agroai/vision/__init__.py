"""Vision: multimodal interfaces with a hard honesty boundary.

What is implemented today (real code, tested):

* **file inspection** - :func:`~agroai.vision.image.inspect_image` reads image
  headers (PNG/JPEG/GIF/BMP/WEBP/TIFF) and
  :func:`~agroai.vision.video.inspect_video` reads video containers plus the
  MP4/MOV ``mvhd`` duration, straight from the bytes.
* **planning** - :func:`~agroai.vision.preprocessing.plan_resize` and
  :func:`~agroai.vision.preprocessing.plan_frames` compute exactly what a
  preprocessing stage should do (no pixels are touched).
* **the pipeline** - :class:`~agroai.vision.analysis.MediaAnalyzer` combines
  inspection, the (possibly unconfigured) vision engine and differential
  diagnosis into one structured answer.

What is a contract only:

* :class:`~agroai.vision.base.VisionEngine` - the interface a real
  plant/animal/video model implements. The shipped engine reports
  ``not_configured`` and returns **zero** observations.
* content-aware frame selection
  (:func:`~agroai.vision.video.select_important_frames`) raises
  ``NotImplementedError``: ranking frames needs a decoder and a model.

The latency target (~30 s) is a performance goal, not a guarantee; it is carried
on the engine so the API layer can report it instead of promising it.
"""

from __future__ import annotations

from agroai.vision.analysis import (
    LIMITATION_NO_VISION,
    MediaAnalysis,
    MediaAnalyzer,
    SAFETY_RULES,
    analyze_media,
)
from agroai.vision.base import (
    VISION_ERROR,
    VISION_NOT_CONFIGURED,
    VISION_NOT_IMPLEMENTED,
    VISION_OK,
    MediaInfo,
    MediaKind,
    VisionEngine,
    VisionObservation,
    VisionResult,
)
from agroai.vision.engines import (
    PlaceholderVisionEngine,
    UnconfiguredVisionEngine,
    get_vision_engine,
)
from agroai.vision.image import (
    MediaReadError,
    detect_image_format,
    inspect_image,
    is_image,
)
from agroai.vision.preprocessing import (
    FramePlan,
    ResizePlan,
    plan_frames,
    plan_resize,
    resize_dimensions,
)
from agroai.vision.video import (
    detect_video_container,
    inspect_video,
    is_video,
    plan_video_frames,
    select_important_frames,
)

__all__ = [
    "FramePlan",
    "LIMITATION_NO_VISION",
    "MediaAnalysis",
    "MediaAnalyzer",
    "MediaInfo",
    "MediaKind",
    "MediaReadError",
    "PlaceholderVisionEngine",
    "ResizePlan",
    "SAFETY_RULES",
    "UnconfiguredVisionEngine",
    "VISION_ERROR",
    "VISION_NOT_CONFIGURED",
    "VISION_NOT_IMPLEMENTED",
    "VISION_OK",
    "VisionEngine",
    "VisionObservation",
    "VisionResult",
    "analyze_media",
    "detect_image_format",
    "detect_video_container",
    "get_vision_engine",
    "inspect_image",
    "inspect_video",
    "is_image",
    "is_video",
    "plan_frames",
    "plan_resize",
    "plan_video_frames",
    "resize_dimensions",
    "select_important_frames",
]
