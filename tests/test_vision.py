"""Vision: real byte inspection, planning, and honest "not configured" answers."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest

from agroai.config.settings import Settings
from agroai.knowledge.documents import Chunk, KnowledgeDocument, RetrievedChunk
from agroai.vision import (
    VISION_ERROR,
    VISION_NOT_CONFIGURED,
    VISION_NOT_IMPLEMENTED,
    MediaAnalyzer,
    MediaKind,
    MediaReadError,
    PlaceholderVisionEngine,
    UnconfiguredVisionEngine,
    detect_image_format,
    detect_video_container,
    get_vision_engine,
    inspect_image,
    inspect_video,
    is_image,
    is_video,
    plan_frames,
    plan_resize,
    plan_video_frames,
    resize_dimensions,
    select_important_frames,
)


# ---------------------------------------------------------------------- #
# Fixtures that build real headers
# ---------------------------------------------------------------------- #
def _png(width: int = 640, height: int = 480) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + struct.pack(">I", 13)
        + b"IHDR"
        + struct.pack(">II", width, height)
        + b"\x08\x06\x00\x00\x00"
    )


def _gif(width: int = 100, height: int = 50) -> bytes:
    return b"GIF89a" + struct.pack("<HH", width, height) + b"\x00" * 4


def _bmp(width: int = 32, height: int = 16) -> bytes:
    return b"BM" + b"\x00" * 16 + struct.pack("<ii", width, height) + b"\x00" * 4


def _jpeg(width: int = 200, height: int = 100) -> bytes:
    # SOI, APP0 segment, then SOF0 with the dimensions.
    app0 = b"\xff\xe0" + struct.pack(">H", 4) + b"\x00\x00"
    sof0 = (
        b"\xff\xc0"
        + struct.pack(">H", 17)
        + b"\x08"
        + struct.pack(">HH", height, width)
        + b"\x03" + b"\x00" * 9
    )
    return b"\xff\xd8" + app0 + sof0


def _box(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload) + 8) + kind + payload


def _mp4(duration_seconds: float = 30.0, timescale: int = 1000) -> bytes:
    ftyp = _box(b"ftyp", b"isom" + b"\x00\x00\x02\x00" + b"isomiso2")
    mvhd_payload = (
        b"\x00\x00\x00\x00"          # version 0 + flags
        + b"\x00" * 8                # creation + modification
        + struct.pack(">I", timescale)
        + struct.pack(">I", int(duration_seconds * timescale))
        + b"\x00" * 80
    )
    moov = _box(b"moov", _box(b"mvhd", mvhd_payload))
    return ftyp + moov


def _write(tmp_path: Path, name: str, data: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


# ---------------------------------------------------------------------- #
# Image inspection
# ---------------------------------------------------------------------- #
def test_png_dimensions_are_read_from_the_header(tmp_path: Path) -> None:
    info = inspect_image(_write(tmp_path, "leaf.png", _png(4032, 3024)))
    assert info.kind is MediaKind.IMAGE
    assert (info.width, info.height) == (4032, 3024)
    assert info.format == "png"
    assert info.mime_type == "image/png"
    assert info.megapixels == pytest.approx(12.19, abs=0.01)
    assert info.inspected is True
    assert info.resolution == "4032x3024"
    assert "png" in info.describe()


def test_jpeg_gif_bmp_dimensions(tmp_path: Path) -> None:
    assert inspect_image(_write(tmp_path, "a.jpg", _jpeg(200, 100))).width == 200
    assert inspect_image(_write(tmp_path, "b.gif", _gif(100, 50))).height == 50
    assert inspect_image(_write(tmp_path, "c.bmp", _bmp(32, 16))).height == 16


def test_unknown_signature_is_not_treated_as_an_image(tmp_path: Path) -> None:
    info = inspect_image(_write(tmp_path, "notes.txt", b"just text"))
    assert info.format == "unknown"
    assert info.inspected is False
    assert info.notes
    assert info.width is None
    assert detect_image_format(b"just text") == "unknown"
    assert is_image(tmp_path / "notes.txt") is False


def test_missing_and_empty_files_raise(tmp_path: Path) -> None:
    with pytest.raises(MediaReadError):
        inspect_image(tmp_path / "nope.png")
    empty = _write(tmp_path, "empty.png", b"")
    with pytest.raises(MediaReadError):
        inspect_image(empty)
    assert is_image(empty) is False


def test_truncated_header_reports_unknown_dimensions(tmp_path: Path) -> None:
    info = inspect_image(_write(tmp_path, "short.png", b"\x89PNG\r\n\x1a\n" + b"\x00" * 4))
    assert info.format == "png"
    assert info.width is None
    assert any("could not be parsed" in note for note in info.notes)


# ---------------------------------------------------------------------- #
# Video inspection
# ---------------------------------------------------------------------- #
def test_mp4_duration_is_read_from_mvhd(tmp_path: Path) -> None:
    info = inspect_video(_write(tmp_path, "clip.mp4", _mp4(30.0)))
    assert info.kind is MediaKind.VIDEO
    assert info.format == "mp4"
    assert info.duration_seconds == pytest.approx(30.0)
    assert info.inspected is True


def test_video_containers_detected() -> None:
    assert detect_video_container(_mp4()) == "mp4"
    assert detect_video_container(b"RIFF" + b"\x00" * 4 + b"AVI ") == "avi"
    assert detect_video_container(b"\x1a\x45\xdf\xa3" + b"webm" + b"\x00" * 100) == "webm"
    assert detect_video_container(b"\x1a\x45\xdf\xa3" + b"\x00" * 20) == "matroska"
    assert detect_video_container(b"nope") == "unknown"


def test_unknown_container_is_reported_not_guessed(tmp_path: Path) -> None:
    info = inspect_video(_write(tmp_path, "file.bin", b"not a video"))
    assert info.inspected is False
    assert info.duration_seconds is None
    assert info.notes
    assert plan_video_frames(info).planned is False


def test_non_mp4_video_has_no_duration_yet(tmp_path: Path) -> None:
    data = b"RIFF" + struct.pack("<I", 100) + b"AVI " + b"\x00" * 32
    info = inspect_video(_write(tmp_path, "clip.avi", data))
    assert info.format == "avi"
    assert info.duration_seconds is None
    assert any("demuxer" in note for note in info.notes)
    assert is_video(tmp_path / "clip.avi") is True


# ---------------------------------------------------------------------- #
# Planning
# ---------------------------------------------------------------------- #
def test_resize_plan_never_upscales_and_preserves_aspect() -> None:
    assert resize_dimensions(500, 400, max_side=1024) == (500, 400)
    wide = resize_dimensions(4032, 3024, max_side=1024)
    assert wide[0] == 1024
    assert wide[1] == pytest.approx(768, abs=2)

    plan = plan_resize(4032, 3024)
    assert plan is not None
    assert plan.resized is True
    assert plan.target_width == 1024
    assert 0 < plan.downscale_factor < 1

    small = plan_resize(300, 200)
    assert small is not None and small.resized is False
    assert plan_resize(None, 100) is None


def test_frame_plan_is_evenly_spaced_and_clamped() -> None:
    plan = plan_frames(20.0, target_fps=1.0, max_frames=12, min_frames=3)
    assert plan.planned is True
    assert plan.frame_count == 12
    assert plan.timestamps == sorted(plan.timestamps)
    assert all(0 < value < 20.0 for value in plan.timestamps)

    short = plan_frames(1.0)
    assert short.frame_count == 3  # min_frames

    assert plan_frames(None).planned is False
    assert plan_frames(0.0).planned is False


def test_video_frame_plan_uses_the_read_duration(tmp_path: Path) -> None:
    info = inspect_video(_write(tmp_path, "clip.mp4", _mp4(10.0)))
    plan = plan_video_frames(info)
    assert plan.planned is True
    assert plan.frame_count == 10


def test_content_aware_selection_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError):
        select_important_frames("clip.mp4")


# ---------------------------------------------------------------------- #
# Engines
# ---------------------------------------------------------------------- #
def test_unconfigured_engine_inspects_but_never_observes(tmp_path: Path) -> None:
    engine = UnconfiguredVisionEngine()
    assert engine.is_configured is False
    result = engine.analyze_image(str(_write(tmp_path, "leaf.png", _png())))
    assert result.status == VISION_NOT_CONFIGURED
    assert result.observations == []
    assert result.media is not None and result.media.width == 640
    assert "Preprocessing plan" in result.detail
    assert engine.health_check()["configured"] is False


def test_unconfigured_engine_reports_missing_files_as_errors(tmp_path: Path) -> None:
    result = UnconfiguredVisionEngine().analyze_image(str(tmp_path / "missing.png"))
    assert result.status == VISION_ERROR
    assert result.media is None


def test_placeholder_engine_is_not_implemented(tmp_path: Path) -> None:
    engine = PlaceholderVisionEngine("clip")
    result = engine.analyze_video(str(_write(tmp_path, "clip.mp4", _mp4())))
    assert result.status == VISION_NOT_IMPLEMENTED
    assert result.observations == []
    assert "not implemented" in result.detail


def test_vision_engine_factory(settings: Settings) -> None:
    engine = get_vision_engine(settings)
    assert engine.name == "none"
    assert engine.is_configured is False
    settings.vision_provider = "clip"
    assert get_vision_engine(settings).name == "clip"


# ---------------------------------------------------------------------- #
# Analysis pipeline
# ---------------------------------------------------------------------- #
class _StubRetriever:
    name = "stub"

    def __init__(self, text: str) -> None:
        self._text = text

    def search(self, query: str, top_k: int = 5, categories=None):
        document = KnowledgeDocument(
            id="d1", title="Early blight", content=self._text, source="FAO"
        )
        chunk = Chunk(id="d1:0", document_id="d1", chunk_index=0, content=self._text)
        return [RetrievedChunk(chunk=chunk, score=1.0, document=document)]

    def health_check(self):
        return None


def test_analysis_combines_inspection_and_differential(tmp_path: Path) -> None:
    path = _write(tmp_path, "leaf.png", _png(1200, 900))
    analyzer = MediaAnalyzer(retriever=_StubRetriever("sariq dog'lar va nekroz"))
    analysis = analyzer.analyze(str(path), question="pomidor bargida sariq dog'lar")
    assert analysis.status == VISION_NOT_CONFIGURED
    assert analysis.vision_available is False
    assert analysis.media is not None and analysis.media.width == 1200
    assert analysis.diagnosis is not None and analysis.diagnosis.hypotheses
    assert any("vision model" in item for item in analysis.limitations)
    rendered = analysis.render()
    assert "Kuzatilgan belgilar" in rendered
    assert "Nima qilmaslik kerak" in rendered
    assert "vision" in " ".join(analysis.pipeline).lower()
    assert analysis.to_dict()["status"] == VISION_NOT_CONFIGURED


def test_analysis_of_an_unknown_file_does_not_crash(tmp_path: Path) -> None:
    path = _write(tmp_path, "thing.bin", b"####")
    analysis = MediaAnalyzer(retriever=_StubRetriever("nekroz")).analyze(str(path))
    assert analysis.status == VISION_ERROR
    assert analysis.media is None
    assert any("aniqlanmadi" in item for item in analysis.limitations)


def test_analysis_without_a_differential_engine(tmp_path: Path) -> None:
    path = _write(tmp_path, "leaf.png", _png())
    analysis = MediaAnalyzer().analyze(str(path))
    assert analysis.diagnosis is None
    assert any("retriever" in item for item in analysis.limitations)
    assert analysis.latency_target_seconds == 30.0


def test_video_analysis_reports_the_frame_plan(tmp_path: Path) -> None:
    path = _write(tmp_path, "clip.mp4", _mp4(20.0))
    analysis = MediaAnalyzer(retriever=_StubRetriever("nekroz")).analyze(str(path))
    assert analysis.media is not None and analysis.media.kind is MediaKind.VIDEO
    assert "Frame plan" in (analysis.vision.detail if analysis.vision else "")
