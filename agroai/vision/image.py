"""Image inspection - real, byte-level, no model.

This module reads an image's *header* and reports what is objectively there:
format, width, height, resolution and byte size. It decodes no pixels and makes
no claims about content. That distinction matters: "tomato leaf, 4032x3024
JPEG" is a fact; "early blight" would be a model output and is never produced
here.

Supported (and parsed for dimensions): PNG, JPEG, GIF, BMP, WEBP.
Recognised but without dimension parsing: TIFF,
whose IFD layout is parsed on a best-effort basis and reported as unknown when
it cannot be read - it is never guessed.

A file whose signature matches nothing is reported as ``unknown`` with a note,
instead of being silently treated as an image.
"""

from __future__ import annotations

import struct
from pathlib import Path

from agroai.vision.base import MediaInfo, MediaKind

#: How many bytes to read for header inspection. Headers are tiny; a truncated
#: or mislabelled file still gets a truthful answer.
MAX_HEADER_BYTES = 65_536


class MediaReadError(RuntimeError):
    """Raised when a media file cannot be read (missing, unreadable, empty)."""


#: Signature (prefix) -> (format name, mime type, dimension parser name).
_SIGNATURES: tuple[tuple[bytes, str, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", "png", "image/png"),
    (b"\xff\xd8\xff", "jpeg", "image/jpeg"),
    (b"GIF87a", "gif", "image/gif"),
    (b"GIF89a", "gif", "image/gif"),
    (b"BM", "bmp", "image/bmp"),
    (b"II*\x00", "tiff", "image/tiff"),
    (b"MM\x00*", "tiff", "image/tiff"),
)

SUPPORTED_IMAGE_FORMATS: tuple[str, ...] = ("png", "jpeg", "gif", "bmp", "webp", "tiff")


def read_header(path: Path | str, size: int = MAX_HEADER_BYTES) -> bytes:
    """Read up to ``size`` bytes from a file, raising a clear error otherwise."""
    path = Path(path)
    if not path.is_file():
        raise MediaReadError(f"Media file not found: {path}")
    try:
        with path.open("rb") as handle:
            data = handle.read(size)
    except OSError as exc:  # pragma: no cover - depends on the filesystem
        raise MediaReadError(f"Cannot read {path}: {exc}") from exc
    if not data:
        raise MediaReadError(f"Media file is empty: {path}")
    return data


# ---------------------------------------------------------------------- #
# Dimension parsers (one per format)
# ---------------------------------------------------------------------- #
def _png_size(data: bytes) -> tuple[int | None, int | None]:
    # 8-byte signature, 4-byte length, 'IHDR', then width/height as BE uint32.
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None, None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def _gif_size(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 10:
        return None, None
    width, height = struct.unpack("<HH", data[6:10])
    return width, height


def _bmp_size(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 26:
        return None, None
    width, height = struct.unpack("<ii", data[18:26])
    return abs(width), abs(height)


def _jpeg_size(data: bytes) -> tuple[int | None, int | None]:
    """Walk JPEG markers to the Start-Of-Frame segment."""
    if len(data) < 4:
        return None, None
    index = 2  # skip SOI
    length = len(data)
    while index + 9 < length:
        if data[index] != 0xFF:
            index += 1
            continue
        marker = data[index + 1]
        if marker in (0xFF, 0x00):  # fill byte / escaped FF
            index += 1
            continue
        if 0xD0 <= marker <= 0xD9 or marker == 0x01:  # standalone markers
            index += 2
            continue
        segment_length = struct.unpack(">H", data[index + 2 : index + 4])[0]
        # SOF0..SOF15, excluding DHT (0xC4), JPG (0xC8) and DAC (0xCC).
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            height, width = struct.unpack(">HH", data[index + 5 : index + 9])
            return width, height
        if segment_length < 2:
            return None, None
        index += 2 + segment_length
    return None, None


def _webp_size(data: bytes) -> tuple[int | None, int | None]:
    if len(data) < 30 or data[:4] != b"RIFF" or data[8:12] != b"WEBP":
        return None, None
    chunk = data[12:16]
    if chunk == b"VP8X":
        width = int.from_bytes(data[24:27], "little") + 1
        height = int.from_bytes(data[27:30], "little") + 1
        return width, height
    if chunk == b"VP8L" and len(data) > 25 and data[20] == 0x2F:
        bits = int.from_bytes(data[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    if chunk == b"VP8 " and len(data) > 30 and data[23:26] == b"\x9d\x01\x2a":
        width = int.from_bytes(data[26:28], "little") & 0x3FFF
        height = int.from_bytes(data[28:30], "little") & 0x3FFF
        return width, height
    return None, None


def _tiff_size(data: bytes) -> tuple[int | None, int | None]:
    """Best-effort TIFF IFD walk for ImageWidth (256) / ImageLength (257)."""
    if len(data) < 8:
        return None, None
    endian = "<" if data[:2] == b"II" else ">"
    try:
        if struct.unpack(endian + "H", data[2:4])[0] != 42:
            return None, None
        offset = struct.unpack(endian + "I", data[4:8])[0]
        if offset + 2 > len(data):
            return None, None
        entries = struct.unpack(endian + "H", data[offset : offset + 2])[0]
        width = height = None
        for index in range(min(entries, 64)):
            entry = offset + 2 + index * 12
            if entry + 12 > len(data):
                break
            tag, field_type, count = struct.unpack(endian + "HHI", data[entry : entry + 8])
            if tag not in (256, 257):
                continue
            raw = data[entry + 8 : entry + 12]
            if field_type == 3:  # SHORT
                value = struct.unpack(endian + "H", raw[:2])[0]
            elif field_type == 4:  # LONG
                value = struct.unpack(endian + "I", raw)[0]
            else:
                continue
            if count > 1:
                continue
            if tag == 256:
                width = value
            else:
                height = value
        return width, height
    except (struct.error, IndexError):
        return None, None


_DIMENSION_PARSERS = {
    "png": _png_size,
    "jpeg": _jpeg_size,
    "gif": _gif_size,
    "bmp": _bmp_size,
    "webp": _webp_size,
    "tiff": _tiff_size,
}

_MIME_TYPES = {
    "png": "image/png",
    "jpeg": "image/jpeg",
    "gif": "image/gif",
    "bmp": "image/bmp",
    "webp": "image/webp",
    "tiff": "image/tiff",
}


def detect_image_format(data: bytes) -> str:
    """Return the image format name, or ``'unknown'``."""
    for signature, name, _mime in _SIGNATURES:
        if data.startswith(signature):
            return name
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return "unknown"


def inspect_image(path: Path | str, *, size: int = MAX_HEADER_BYTES) -> MediaInfo:
    """Read ``path`` and return byte-derived facts about the image.

    Raises :class:`MediaReadError` when the file is missing or empty: reporting a
    non-existent file as a zero-byte image would be worse than failing.
    """
    path = Path(path)
    data = read_header(path, size)
    fmt = detect_image_format(data)

    info = MediaInfo(
        path=str(path),
        kind=MediaKind.IMAGE,
        format=fmt,
        mime_type=_MIME_TYPES.get(fmt, ""),
        size_bytes=path.stat().st_size,
        inspected=fmt != "unknown",
    )
    if fmt == "unknown":
        info.notes.append(
            "Unrecognised image signature; AgroAI will not treat this file as an image."
        )
        return info

    width, height = _DIMENSION_PARSERS[fmt](data)
    info.width, info.height = width, height
    if width is None or height is None:
        info.notes.append(
            f"{fmt.upper()} header recognised but dimensions could not be parsed; "
            "a decoder is required."
        )
    return info


def is_image(path: Path | str) -> bool:
    """True when the file's signature is a known image format."""
    try:
        return detect_image_format(read_header(path)) != "unknown"
    except MediaReadError:
        return False


__all__ = [
    "MAX_HEADER_BYTES",
    "MediaReadError",
    "SUPPORTED_IMAGE_FORMATS",
    "detect_image_format",
    "inspect_image",
    "is_image",
    "read_header",
]
