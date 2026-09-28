"""Tests for TXT and Markdown ingestion, chunking and content hashing."""

from __future__ import annotations

from pathlib import Path

import pytest

from agroai.config.settings import Settings
from agroai.ingestion import (
    DocumentReadError,
    UnimplementedReader,
    UnsupportedFormatError,
    get_reader,
    read_document,
)
from agroai.knowledge.models import iter_chunks


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_txt_reader(tmp_path: Path, settings: Settings) -> None:
    path = _write(tmp_path, "note.txt", "Pomidor barglari sarg'aydi.\nIkkinchi qator.\n")
    document = read_document(path, settings)
    assert document.title == "Pomidor barglari sarg'aydi."
    assert "Ikkinchi qator" in document.content
    assert document.metadata["format"] == "txt"
    assert document.path == str(path.resolve())
    assert document.content_hash


def test_markdown_reader_strips_markup(tmp_path: Path, settings: Settings) -> None:
    path = _write(
        tmp_path,
        "guide.md",
        "# Pomidor kasalligi\n\nBu **muhim** `matn`.\n\n```\ncode here\n```\n\nYakun.\n",
    )
    document = read_document(path, settings)
    assert document.title == "Pomidor kasalligi"
    assert "#" not in document.content
    assert "**" not in document.content
    assert "`" not in document.content
    assert "muhim matn" in document.content
    assert "code here" in document.content  # fenced text kept for search


def test_markdown_reader_strips_blockquote_markers_but_keeps_code(
    tmp_path: Path, settings: Settings
) -> None:
    path = _write(
        tmp_path,
        "note.md",
        "> **Muhim**: bu misol fayl.\n\n```\n> qator kod ichida saqlanadi\n```\n",
    )
    document = read_document(path, settings)
    assert document.content.startswith("Muhim: bu misol fayl.")
    assert "> qator kod ichida saqlanadi" in document.content  # fences untouched


def test_reader_category_inferred_from_knowledge_tree(
    tmp_path: Path, settings: Settings
) -> None:
    target = settings.knowledge_dir / "pests"
    target.mkdir(parents=True)
    path = target / "aphid.txt"
    path.write_text("Shira zararkunandasi haqida.", encoding="utf-8")
    document = read_document(path, settings)
    assert document.category == "pests"


def test_reader_outside_knowledge_tree_is_general(
    tmp_path: Path, settings: Settings
) -> None:
    path = _write(tmp_path, "random.txt", "Matn")
    assert read_document(path, settings).category == "general"


def test_missing_file_raises(tmp_path: Path, settings: Settings) -> None:
    with pytest.raises(DocumentReadError):
        read_document(tmp_path / "nope.txt", settings)


def test_unsupported_extension(tmp_path: Path, settings: Settings) -> None:
    path = _write(tmp_path, "table.xlsx", "binary-ish")
    with pytest.raises(UnsupportedFormatError):
        read_document(path, settings)


def test_future_formats_report_not_implemented(
    tmp_path: Path, settings: Settings
) -> None:
    path = _write(tmp_path, "doc.pdf", "%PDF-1.4 fake")
    reader = get_reader(path)
    assert isinstance(reader, UnimplementedReader)
    with pytest.raises(NotImplementedError):
        reader.read(path, settings)


def test_identical_content_produces_identical_hash(
    tmp_path: Path, settings: Settings
) -> None:
    first = read_document(_write(tmp_path, "a.txt", "bir xil matn"), settings)
    second = read_document(_write(tmp_path, "b.txt", "bir xil matn"), settings)
    assert first.content_hash == second.content_hash
    assert first.id != second.id  # ids differ, dedup is by hash


def test_chunking_long_document_preserves_metadata(
    tmp_path: Path, settings: Settings
) -> None:
    path = _write(tmp_path, "long.txt", "qator matn. " * 300)
    document = read_document(path, settings)
    chunks = list(iter_chunks(document, chunk_size=600, overlap=80))
    assert len(chunks) > 1
    for chunk in chunks:
        assert chunk.document_id == document.id
        assert chunk.metadata["format"] == "txt"


def test_non_utf8_file_raises(tmp_path: Path, settings: Settings) -> None:
    path = tmp_path / "broken.txt"
    path.write_bytes(b"\xff\xfe\xfa")
    with pytest.raises(DocumentReadError):
        read_document(path, settings)
