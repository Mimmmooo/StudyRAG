"""Load page-aware text from text PDFs, scanned PDFs and study notes."""

import hashlib
import json
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path

import fitz

SUPPORTED_SUFFIXES = {".pdf", ".txt", ".md"}
OCR_DPI = 130
OCR_TIMEOUT_SECONDS = 30
MIN_SEARCHABLE_TEXT_CHARS = 80
SCANNED_IMAGE_AREA_FRACTION = 0.65


@dataclass(frozen=True)
class DocumentPage:
    """A document page with citation metadata."""

    text: str
    source: str
    page: int


def _clean(text: str) -> str:
    """Normalize whitespace without merging separate lines."""
    return "\n".join(line.strip() for line in text.splitlines() if line.strip())


def _is_image_dominant(page: fitz.Page, text: str) -> bool:
    """Spot scanned pages whose tiny text layer contains only labels or page numbers."""
    if len(text.strip()) >= MIN_SEARCHABLE_TEXT_CHARS:
        return False
    page_area = page.rect.get_area()
    if page_area <= 0:
        return False
    for image in page.get_image_info():
        bounds = image.get("bbox")
        if bounds is None:
            continue
        visible = fitz.Rect(bounds) & page.rect
        if not visible.is_empty and (
            visible.get_area() / page_area >= SCANNED_IMAGE_AREA_FRACTION
        ):
            return True
    return False


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
            env={**os.environ, "OMP_THREAD_LIMIT": "1"},
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


def _cache_path(path: Path) -> Path:
    """Locate a local, versioned OCR cache by the PDF contents."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    root = Path(os.getenv("STUDYRAG_OCR_CACHE_DIR", str(Path.home() / ".cache" / "studyrag" / "ocr")))
    return root / f"{digest.hexdigest()}-dpi{OCR_DPI}-v2.json"


def _read_cached_pages(cache: Path, source: str) -> list[DocumentPage] | None:
    """Read reusable page text, falling back to OCR if the cache is damaged."""
    if not cache.is_file():
        return None
    try:
        rows = json.loads(cache.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            return None
        return [
            DocumentPage(row["text"], source, int(row["page"]))
            for row in rows
            if isinstance(row.get("text"), str) and row["text"].strip()
        ] or None
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return None


def _save_cached_pages(cache: Path, pages: list[DocumentPage]) -> None:
    """Save extracted text locally only after the full PDF succeeds."""
    try:
        cache.parent.mkdir(parents=True, exist_ok=True)
        temp = cache.with_suffix(".tmp")
        temp.write_text(
            json.dumps([{"text": row.text, "page": row.page} for row in pages], ensure_ascii=False),
            encoding="utf-8",
        )
        temp.replace(cache)
    except OSError:
        pass  # Caching is optional; never fail indexing due to cache permissions.


def _load_pdf(path: Path) -> list[DocumentPage]:
    """Process one PDF page at a time to avoid holding page images in RAM."""
    cache = _cache_path(path)
    cached = _read_cached_pages(cache, path.name)
    if cached is not None:
        return cached
    pages: list[DocumentPage] = []
    try:
        with fitz.open(path) as document:
            if document.needs_pass:
                raise ValueError(f"Password-protected PDF: {path.name}")
            for index, page in enumerate(document, start=1):
                text = _clean(page.get_text("text"))
                if not text or _is_image_dominant(page, text):
                    scanned_text = _ocr_page(page)
                    if len(scanned_text) > len(text):
                        text = scanned_text
                if text:
                    pages.append(DocumentPage(text, path.name, index))
    except (ValueError, RuntimeError) as exc:
        raise ValueError(f"Could not read PDF {path.name}: {exc}") from exc
    if not pages:
        raise ValueError(f"No extractable text in {path.name}, even after OCR.")
    _save_cached_pages(cache, pages)
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
