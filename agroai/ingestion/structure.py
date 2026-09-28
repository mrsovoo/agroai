"""Document structure detection: chapters and sections, without a model.

The roadmap's ingestion pipeline is:

```text
PDF -> Document Parser -> Full Text -> Chapter Detection -> Section Detection
    -> Table Extraction -> Cleaning -> Deduplication -> Metadata
    -> Semantic Chunking -> Embedding -> Knowledge Base
```

Chapter/section detection is pure text analysis, so it can be real **today** and
works for every format the pipeline can read (a converted PDF, a markdown book,
a plain-text manual). It recognises:

* Markdown headings (``#``..``######``),
* numbered headings (``3``, ``3.1``, ``3.1.2``),
* chapter words in Uzbek/Russian/English (bob, глава, chapter, qism, bo'lim...),
* short ALL-CAPS lines (the usual plain-text rendering of a book heading).

Everything is a regex over the text, so the result is inspectable and
reproducible. Table extraction is *not* implemented: a table without layout
information is a guess, and :func:`extract_tables` says so instead of producing
one.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

KIND_MARKDOWN = "markdown"
KIND_NUMBERED = "numbered"
KIND_CHAPTER = "chapter"
KIND_CAPS = "caps"

MIN_TITLE_CHARS = 3
MAX_TITLE_CHARS = 120

_MARKDOWN_RE = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")
_NUMBERED_RE = re.compile(r"^(\d+(?:\.\d+){0,3})[.)]?\s+(\S.*?)\s*$")
_CHAPTER_RE = re.compile(
    r"^\s*(bob|bo'lim|bolim|qism|глава|раздел|часть|chapter|part|section|unit)"
    r"\s*([0-9]+|[IVXLC]+)?\s*[.:)\-–—]?\s*(.*?)\s*$",
    re.IGNORECASE,
)
_CAPS_RE = re.compile(r"^[A-ZА-ЯЁ0-9][A-ZА-ЯЁ0-9 ,'’\-()/]{3,79}$")


@dataclass
class Section:
    """One detected chapter/section, with its character span in the document."""

    title: str
    kind: str
    level: int = 1
    number: str = ""
    start: int = 0
    end: int = 0
    line: int = 0

    @property
    def char_count(self) -> int:
        return max(0, self.end - self.start)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def render(self) -> str:
        indent = "  " * max(0, self.level - 1)
        label = f"{self.number} {self.title}".strip()
        return f"{indent}{label}"


@dataclass
class DocumentStructure:
    """The structure of one document: its sections plus a flat outline."""

    sections: list[Section] = field(default_factory=list)
    kind_counts: dict[str, int] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.sections)

    @property
    def titles(self) -> list[str]:
        return [section.title for section in self.sections]

    def outline(self, limit: int | None = None) -> str:
        chosen = self.sections[:limit] if limit else self.sections
        return "\n".join(section.render() for section in chosen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sections": [section.to_dict() for section in self.sections],
            "count": len(self.sections),
            "kind_counts": dict(self.kind_counts),
        }


def _clean_title(raw: str) -> str:
    title = re.sub(r"\s+", " ", (raw or "").strip())
    title = title.strip(".#*_ ")
    return title


def _looks_like_numbered_heading(number: str, title: str) -> bool:
    """Reject things like "2 kg of fertilizer" that merely start with a number."""
    parts = number.split(".")
    if len(parts) > 4:
        return False
    if not title:
        # A bare number line is only a heading at the top level (e.g. "3").
        return len(parts) == 1
    if len(parts) > 1:
        return True
    words = title.split()
    if len(words) > 12:
        return False
    # A single number followed by prose is usually a list item, not a heading.
    return title[0].isupper() or title.endswith(":")


def detect_sections(
    text: str,
    *,
    max_title_chars: int = MAX_TITLE_CHARS,
    allow_caps_headings: bool = True,
) -> DocumentStructure:
    """Detect chapters and sections in ``text`` (regex only, fully offline)."""
    text = text or ""
    lines = text.splitlines()
    offsets: list[int] = []
    position = 0
    for line in lines:
        offsets.append(position)
        position += len(line) + 1  # +1 for the newline

    found: list[Section] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or len(stripped) > max_title_chars:
            continue

        match = _MARKDOWN_RE.match(stripped)
        if match:
            found.append(
                Section(
                    title=_clean_title(match.group(2)),
                    kind=KIND_MARKDOWN,
                    level=len(match.group(1)),
                    start=offsets[index],
                    line=index + 1,
                )
            )
            continue

        match = _CHAPTER_RE.match(stripped)
        if match and (match.group(3) or match.group(2)):
            number = (match.group(2) or "").strip()
            title = _clean_title(match.group(3)) or _clean_title(match.group(1))
            found.append(
                Section(
                    title=title,
                    kind=KIND_CHAPTER,
                    level=1 if not number or "." not in number else 2,
                    number=f"{match.group(1).capitalize()} {number}".strip(),
                    start=offsets[index],
                    line=index + 1,
                )
            )
            continue

        match = _NUMBERED_RE.match(stripped)
        if match and _looks_like_numbered_heading(match.group(1), match.group(2)):
            number = match.group(1)
            found.append(
                Section(
                    title=_clean_title(match.group(2)),
                    kind=KIND_NUMBERED,
                    level=min(4, number.count(".") + 1),
                    number=number,
                    start=offsets[index],
                    line=index + 1,
                )
            )
            continue

        if allow_caps_headings and _CAPS_RE.match(stripped) and any(
            character.isalpha() for character in stripped
        ):
            words = stripped.split()
            # All-caps prose is usually a warning; a heading is short.
            if len(words) <= 10 and len(stripped) >= MIN_TITLE_CHARS:
                found.append(
                    Section(
                        title=_clean_title(stripped),
                        kind=KIND_CAPS,
                        level=1,
                        start=offsets[index],
                        line=index + 1,
                    )
                )

    # Fill in each section's end offset from the next section's start.
    for index, section in enumerate(found):
        section.end = found[index + 1].start if index + 1 < len(found) else len(text)

    structure = DocumentStructure(sections=found)
    counts: dict[str, int] = {}
    for section in found:
        counts[section.kind] = counts.get(section.kind, 0) + 1
    structure.kind_counts = counts
    return structure


def section_text(text: str, section: Section) -> str:
    """Return the body text of one section."""
    return (text or "")[section.start : section.end]


def section_index(structure: DocumentStructure) -> str:
    """Return a compact outline (used in reports and the CLI)."""
    if not structure.sections:
        return "(no chapters or sections detected)"
    return structure.outline(limit=40)


def extract_tables(*args: object, **kwargs: object) -> None:
    """Not implemented: tables need layout information, not guesswork.

    Raises :class:`NotImplementedError`. A "table" reconstructed from reflowed
    text can invert a dosage column, which is unacceptable in this domain.
    """
    raise NotImplementedError(
        "Table extraction is not implemented in this phase: it requires layout "
        "information from the source document. Reflowed text must not be turned "
        "into a table, because a mis-read dose column is dangerous."
    )


__all__ = [
    "DocumentStructure",
    "KIND_CAPS",
    "KIND_CHAPTER",
    "KIND_MARKDOWN",
    "KIND_NUMBERED",
    "MAX_TITLE_CHARS",
    "MIN_TITLE_CHARS",
    "Section",
    "detect_sections",
    "extract_tables",
    "section_index",
    "section_text",
]
