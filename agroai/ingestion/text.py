"""Plain-text ingestion: TXT and Markdown (the formats implemented today)."""

from __future__ import annotations

from pathlib import Path

from agroai.config.settings import Settings
from agroai.ingestion.base import DocumentReadError, DocumentReader
from agroai.knowledge.documents import KnowledgeDocument
from agroai.knowledge.metadata import content_hash, parse_front_matter


class TextLikeReader(DocumentReader):
    """Shared UTF-8 reading for plain-text style formats."""

    def read(self, path: Path, settings: Settings) -> KnowledgeDocument:
        if not path.is_file():
            raise DocumentReadError(f"File not found: {path}")
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentReadError(f"{path} is not valid UTF-8 text: {exc}") from exc

        fields, text = parse_front_matter(raw)
        digest = content_hash(text)
        return self._base_document(
            path,
            settings,
            content=text,
            digest=digest,
            title=self._title_from_text(path, text),
            fields=fields,
        )


class TxtReader(TextLikeReader):
    """Plain-text reader."""

    format_name = "txt"
    extensions = (".txt",)


class MarkdownReader(DocumentReader):
    """Markdown reader.

    Markdown is stored as normalized plain text: markdown emphasis/backtick
    markers are stripped and headings keep their text so they stay searchable.
    """

    format_name = "markdown"
    extensions = (".md", ".markdown")

    # Inline markers stripped before storage (headings are kept as text).
    _INLINE_STRIP = ("**", "__", "*", "`")

    def read(self, path: Path, settings: Settings) -> KnowledgeDocument:
        if not path.is_file():
            raise DocumentReadError(f"File not found: {path}")
        try:
            raw = path.read_text(encoding="utf-8")
        except UnicodeDecodeError as exc:
            raise DocumentReadError(f"{path} is not valid UTF-8 text: {exc}") from exc

        fields, body = parse_front_matter(raw)
        title = self._title_from_text(path, body)
        text = self._normalize(body)

        digest = content_hash(text)
        return self._base_document(
            path,
            settings,
            content=text,
            digest=digest,
            title=title,
            fields=fields,
        )

    def _normalize(self, raw: str) -> str:
        lines: list[str] = []
        in_fence = False
        for line in raw.splitlines():
            stripped = line.strip()
            if stripped.startswith("```"):
                in_fence = not in_fence
                lines.append("")  # keep code fences as blank separators
                continue
            if in_fence:
                lines.append(line)
                continue
            if stripped.startswith("#"):
                # Keep heading text, drop the '#' markers.
                stripped = stripped.lstrip("#").strip()
            elif stripped.startswith(">"):
                # Blockquote markers are markup: keep the quoted text only.
                stripped = stripped.lstrip(">").strip()
            for marker in self._INLINE_STRIP:
                stripped = stripped.replace(marker, "")
            lines.append(stripped.rstrip())
        # Collapse 3+ consecutive blank lines to one blank line.
        normalized: list[str] = []
        blanks = 0
        for line in lines:
            blanks = blanks + 1 if not line.strip() else 0
            if blanks <= 1:
                normalized.append(line)
        return "\n".join(normalized).strip()


__all__ = ["MarkdownReader", "TextLikeReader", "TxtReader"]
