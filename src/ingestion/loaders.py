"""Load page-aware text from supported study documents."""

from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}


@dataclass(frozen=True)
class DocumentPage:
    """A document page with citation metadata."""

    text: str
    source: str
    page: int


def load_document(path: Path) -> list[DocumentPage]:
    """Read PDF pages or a plain-text file into citation-ready records."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")

    if suffix == ".pdf":
        try:
            reader = PdfReader(str(path))
            pages = [
                DocumentPage(text=_clean(page.extract_text() or ""), source=path.name, page=i)
                for i, page in enumerate(reader.pages, start=1)
            ]
        except Exception as exc:
            raise ValueError(f"Could not read PDF {path.name}: {exc}") from exc
    else:
        try:
            content = path.read_text(encoding="utf-8-sig")
        except UnicodeError as exc:
            raise ValueError(f"File must be UTF-8 text: {path.name}") from exc
        pages = [DocumentPage(text=_clean(content), source=path.name, page=1)]

    nonempty = [page for page in pages if page.text]
    if not nonempty:
        raise ValueError(
            f"No extractable text in {path.name}. Scanned PDFs need OCR (not supported yet)."
        )
    return nonempty


def _clean(text: str) -> str:
    """Normalize whitespace without merging separate lines."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())
