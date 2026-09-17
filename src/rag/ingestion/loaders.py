"""Format-specific loaders that normalise input into a common LoadedDocument.

Structure is retained deliberately: heading hierarchy and page numbers are what make
structure-aware chunking possible in Phase 2, and they cannot be recovered later.

Each loader returns plain text plus a list of `sections` — one entry per detected
heading, recording where in the text it starts. The chunker uses these offsets to
split on structural boundaries instead of blindly every N tokens.
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from rag.models import DocumentMeta

MIN_EXTRACTABLE_CHARS = 50
"""Below this, a document is treated as having no usable text layer.

A 13-page PDF that yields 3 characters is a scan or photo, not a text document.
Indexing it would add an empty entry that silently degrades retrieval, so loaders
raise instead. Chosen conservatively: a genuinely tiny text file is rare, a
scanned PDF that leaks a few stray characters is common.
"""

MIN_CHARS_PER_PAGE = 200
"""Below this average, a multi-page PDF is treated as a scan.

The absolute floor above catches a completely empty file but not a scan that leaks
a few thousand characters of running headers. A 48-page document yielding 5,566
characters — 116 per page — passed that check and indexed three chunks of pure
boilerplate. Density is the honest test: a real text page runs 1,500-3,000
characters, so 200 is far below anything legitimate.
"""

BOILERPLATE_MIN_PAGES = 3
"""Pages a document needs before repeated-line detection is worth running."""

BOILERPLATE_THRESHOLD = 0.6
"""Fraction of pages a line must appear on to count as a header or footer.

Running headers appear on nearly every page; a sentence that happens to recur twice
in a long document does not. Set high enough that ordinary repetition survives,
low enough to catch boilerplate missing from title and index pages.
"""

MAX_BOILERPLATE_LINE_CHARS = 120
"""Longest line considered as possible boilerplate.

Headers and footers are short. A long repeated passage is far more likely to be
real content — a recurring definition, a standard disclaimer — and removing it
would silently delete text the user may ask about.
"""


class EmptyTextLayerError(ValueError):
    """Raised when a file parses correctly but contains no extractable text.

    Almost always a scanned or photographed document needing OCR. Callers should
    skip the file and log the path rather than indexing an empty document.
    """


class UnsupportedFormatError(ValueError):
    """Raised when no loader is registered for a file extension."""


class LoadedDocument:
    def __init__(self, text: str, meta: DocumentMeta, sections: list[dict[str, Any]] | None = None):
        self.text = text
        self.meta = meta
        self.sections = sections or []

    def __repr__(self) -> str:
        return (
            f"LoadedDocument(path={self.meta.source_path!r}, "
            f"chars={len(self.text)}, sections={len(self.sections)})"
        )


def content_hash(path: Path) -> str:
    """SHA-256 of the raw file bytes.

    Used for incremental re-indexing: an unchanged hash means the document does not
    need re-embedding. Hashing bytes rather than extracted text means a re-export of
    the same content counts as a change, which is the safer direction to err in.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def _normalise(text: str) -> str:
    """Collapse runaway blank lines and trailing whitespace without losing paragraphs."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _guard_empty(text: str, path: Path, hint: str) -> None:
    if len(text.strip()) < MIN_EXTRACTABLE_CHARS:
        raise EmptyTextLayerError(
            f"{path.name}: only {len(text.strip())} characters extracted. {hint}"
        )


def find_boilerplate(pages: list[str]) -> set[str]:
    """Identify running headers and footers by how often they repeat.

    Detection is by frequency rather than by pattern, so no filename, course code or
    page-number format is hardcoded — the same logic handles any document.

    This matters more than it sounds. Boilerplate does not merely waste tokens: a
    chunk that is mostly repeated header text resembles every other page, so it wins
    retrieval slots on weak queries and pushes real content out of the results.
    """
    if len(pages) < BOILERPLATE_MIN_PAGES:
        return set()

    counts: dict[str, int] = {}
    for page in pages:
        for line in {ln.strip() for ln in page.splitlines() if ln.strip()}:
            if len(line) <= MAX_BOILERPLATE_LINE_CHARS:
                counts[line] = counts.get(line, 0) + 1

    needed = max(BOILERPLATE_MIN_PAGES, int(len(pages) * BOILERPLATE_THRESHOLD))
    return {line for line, count in counts.items() if count >= needed}


def _strip_boilerplate(page: str, boilerplate: set[str]) -> str:
    """Remove boilerplate lines, and bare page numbers left stranded beside them."""
    kept = [
        line
        for line in page.splitlines()
        if line.strip() and line.strip() not in boilerplate and not line.strip().isdigit()
    ]
    return _normalise("\n".join(kept))


class Loader(ABC):
    extensions: tuple[str, ...] = ()
    doc_type: str = "unknown"

    @abstractmethod
    def load(self, path: Path) -> LoadedDocument: ...

    def _meta(self, path: Path, title: str | None, page_count: int | None = None) -> DocumentMeta:
        return DocumentMeta(
            source_path=str(path),
            title=title or path.stem,
            doc_type=self.doc_type,
            content_hash=content_hash(path),
            ingested_at=datetime.now(UTC),
            page_count=page_count,
        )


class PdfLoader(Loader):
    """Extracts text from a PDF, one page at a time, preserving page numbers.

    Page numbers are captured here because they are the only citation anchor a PDF
    offers — there is no heading hierarchy to rely on. Each page becomes a section
    entry so a chunk can later say "page 7" rather than just naming the file.
    """

    extensions = (".pdf",)
    doc_type = "pdf"

    def load(self, path: Path) -> LoadedDocument:
        from pypdf import PdfReader

        reader = PdfReader(str(path))

        # First pass: extract every page, so repeated lines can be counted across the
        # whole document. Boilerplate cannot be identified from one page in isolation.
        raw_pages = [_normalise(page.extract_text() or "") for page in reader.pages]
        self._guard_density(raw_pages, path)

        boilerplate = find_boilerplate(raw_pages)

        parts: list[str] = []
        sections: list[dict[str, Any]] = []
        cursor = 0

        for number, raw in enumerate(raw_pages, start=1):
            page_text = _strip_boilerplate(raw, boilerplate) if boilerplate else raw
            if not page_text:
                continue
            sections.append(
                {"heading": f"Page {number}", "level": 1, "page": number, "start": cursor}
            )
            parts.append(page_text)
            cursor += len(page_text) + 2

        text = "\n\n".join(parts)
        _guard_empty(
            text,
            path,
            "This looks like a scanned or photographed PDF with no text layer. "
            "It needs OCR before it can be indexed.",
        )

        title = None
        if reader.metadata and reader.metadata.title:
            title = str(reader.metadata.title).strip() or None

        return LoadedDocument(text, self._meta(path, title, len(reader.pages)), sections)

    @staticmethod
    def _guard_density(pages: list[str], path: Path) -> None:
        """Reject a PDF whose pages are too sparse to be anything but a scan."""
        if len(pages) < BOILERPLATE_MIN_PAGES:
            return
        total = sum(len(page.strip()) for page in pages)
        per_page = total / len(pages)
        if per_page < MIN_CHARS_PER_PAGE:
            raise EmptyTextLayerError(
                f"{path.name}: {total} characters across {len(pages)} pages "
                f"({per_page:.0f} per page). A text page runs 1,500-3,000 characters, so "
                "this is a scanned or photographed document. It needs OCR before indexing."
            )


class TextLoader(Loader):
    """Plain text. No structure to recover, so the whole file is one section."""

    extensions = (".txt",)
    doc_type = "text"

    def load(self, path: Path) -> LoadedDocument:
        text = _normalise(path.read_text(encoding="utf-8", errors="replace"))
        _guard_empty(text, path, "The file appears to be empty.")
        return LoadedDocument(text, self._meta(path, None), [])


class MarkdownLoader(Loader):
    """Markdown, keeping the heading hierarchy.

    Headings are the highest-value structural signal available: a chunk that knows it
    sits under "Setup > Installation" can carry that path as a contextual header, which
    measurably improves retrieval over the bare chunk text.
    """

    extensions = (".md", ".markdown")
    doc_type = "markdown"

    _heading = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)

    def load(self, path: Path) -> LoadedDocument:
        raw = _normalise(path.read_text(encoding="utf-8", errors="replace"))
        _guard_empty(raw, path, "The file appears to be empty.")

        sections: list[dict[str, Any]] = [
            {
                "heading": match.group(2),
                "level": len(match.group(1)),
                "page": None,
                "start": match.start(),
            }
            for match in self._heading.finditer(raw)
        ]

        title = sections[0]["heading"] if sections and sections[0]["level"] == 1 else None
        return LoadedDocument(raw, self._meta(path, title), sections)


class HtmlLoader(Loader):
    """HTML, stripped to text with h1-h6 retained as sections."""

    extensions = (".html", ".htm")
    doc_type = "html"

    def load(self, path: Path) -> LoadedDocument:
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        headings = [h.get_text(strip=True) for h in soup.find_all(re.compile(r"^h[1-6]$"))]
        levels = [int(h.name[1]) for h in soup.find_all(re.compile(r"^h[1-6]$"))]

        text = _normalise(soup.get_text("\n"))
        _guard_empty(text, path, "No readable text found in the markup.")

        sections: list[dict[str, Any]] = []
        for heading, level in zip(headings, levels, strict=True):
            start = text.find(heading)
            if start != -1:
                sections.append({"heading": heading, "level": level, "page": None, "start": start})

        title = soup.title.get_text(strip=True) if soup.title else None
        return LoadedDocument(text, self._meta(path, title), sections)


class DocxLoader(Loader):
    """Word documents, using paragraph styles to detect headings."""

    extensions = (".docx",)
    doc_type = "docx"

    def load(self, path: Path) -> LoadedDocument:
        import docx

        document = docx.Document(str(path))
        parts: list[str] = []
        sections: list[dict[str, Any]] = []
        cursor = 0

        for paragraph in document.paragraphs:
            content = paragraph.text.strip()
            if not content:
                continue
            style = (paragraph.style.name or "") if paragraph.style else ""
            if style.startswith("Heading"):
                level = int(style.split()[-1]) if style.split()[-1].isdigit() else 1
                sections.append({"heading": content, "level": level, "page": None, "start": cursor})
            parts.append(content)
            cursor += len(content) + 1

        text = _normalise("\n".join(parts))
        _guard_empty(text, path, "The document contains no readable paragraphs.")
        return LoadedDocument(text, self._meta(path, None), sections)


LOADERS: tuple[Loader, ...] = (
    PdfLoader(),
    TextLoader(),
    MarkdownLoader(),
    HtmlLoader(),
    DocxLoader(),
)

_REGISTRY: dict[str, Loader] = {
    extension: loader for loader in LOADERS for extension in loader.extensions
}

SUPPORTED_EXTENSIONS: frozenset[str] = frozenset(_REGISTRY)


def loader_for(path: Path) -> Loader:
    """Dispatch to the loader registered for this file extension."""
    loader = _REGISTRY.get(path.suffix.lower())
    if loader is None:
        raise UnsupportedFormatError(
            f"No loader for '{path.suffix}'. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
        )
    return loader


def load(path: Path) -> LoadedDocument:
    """Load any supported document. Convenience wrapper over `loader_for`."""
    return loader_for(path).load(path)
