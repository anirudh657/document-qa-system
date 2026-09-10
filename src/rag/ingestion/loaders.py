"""Format-specific loaders that normalise input into a common Document.

Structure is retained deliberately: heading hierarchy and page numbers are what make
structure-aware chunking possible in Phase 2, and they cannot be recovered later.

TODO(phase-1): implement PdfLoader, MarkdownLoader, HtmlLoader, TextLoader, DocxLoader.
"""

from abc import ABC, abstractmethod
from pathlib import Path

from rag.models import DocumentMeta


class LoadedDocument:
    def __init__(self, text: str, meta: DocumentMeta, sections: list[dict] | None = None):
        self.text = text
        self.meta = meta
        self.sections = sections or []


class Loader(ABC):
    extensions: tuple[str, ...] = ()

    @abstractmethod
    def load(self, path: Path) -> LoadedDocument: ...


def loader_for(path: Path) -> Loader:
    """Dispatch to the loader registered for this file extension."""
    raise NotImplementedError("TODO(phase-1): loader registry")
