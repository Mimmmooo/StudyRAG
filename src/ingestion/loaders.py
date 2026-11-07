"""Load page-aware text from text PDFs, scanned PDFs and study notes."""

import subprocess
from dataclasses import dataclass
from pathlib import Path

import fitz

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
OCR_DPI = 140
OCR_TIMEOUT_SECONDS = 30


@dataclass(frozen=True)
class DocumentPage:
    """A document page with citation metadata."""

    text: str
    source: str
    page: int


def _clean(text: str) -> str:
    """Normalize whitespace without merging separate lines."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _ocr_page(page: fitz.Page) -> str:
    """Read one rasterized page using free Bengali/English Tesseract OCR."""
    pixmap = page.get_pixmap(
        matrix=fitz.Matrix(OCR_DPI / 72, OCR_DPI / 72),
        alpha=False,
    )
    try:
        result = subprocess.run(
            ["tesseract", "stdin", "stdout", "-l", "ben+eng", "--psm", "3"],
            input=pixmap.tobytes("png"),
            capture_output=True,
            timeout=OCR_TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        raise ValueError(
            "This PDF needs OCR. Tesseract with Bengali and English "
            "language data is required on the server."
        ) from exc
    except subprocess.TimeoutExpired as exc:
        raise ValueError(
            f"OCR timed out on page {page.number + 1}. Try a smaller PDF."
        ) from exc
    if result.returncode:
        raise ValueError(
            f"OCR failed on page {page.number + 1}. "
            "Make sure Bengali and English language packs are installed."
        )
    return _clean(result.stdout.decode("utf-8", errors="replace"))


def _load_pdf(path: Path) -> list[DocumentPage]:
    """Process one PDF page at a time to avoid holding page images in RAM."""
    pages: list[DocumentPage] = []
    try:
        with fitz.open(path) as document:
            if document.needs_pass:
                raise ValueError(f"Password-protected PDF: {path.name}")
            for index, page in enumerate(document, start=1):
                text = _clean(page.get_text("text"))
                if not text:
                    text = _ocr_page(page)
                if text:
                    pages.append(DocumentPage(text, path.name, index))
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Could not read PDF {path.name}: {exc}") from exc
    if not pages:
        raise ValueError(f"No extractable text in {path.name}, even after OCR.")
    return pages


def load_document(path: Path) -> list[DocumentPage]:
    """Read PDF pages or UTF-8 plain text with original source/page labels."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported file type: {suffix}")
    if suffix == ".pdf":
        return _load_pdf(path)
    try:
        content = path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise ValueError(f"File must be UTF-8 text: {path.name}") from exc
    text = _clean(content)
    if not text:
        raise ValueError(f"No extractable text in {path.name}.")
    return [DocumentPage(text=text, source=path.name, page=1)]
