"""Exercise image-only PDF fallback without requiring OCR in CI."""

from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter

from src.ingestion.loaders import load_document


def test_scanned_page_runs_ocr_and_keeps_original_page(tmp_path: Path) -> None:
    """A PDF page without a text layer is indexable through the OCR adapter."""
    path = tmp_path / "scanned.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=400, height=600)
    writer.add_blank_page(width=400, height=600)
    with path.open("wb") as stream:
        writer.write(stream)
    with patch(
        "src.ingestion.loaders._ocr_page",
        side_effect=["বাংলা বইয়ের প্রথম পৃষ্ঠা", "Second page"],
    ) as ocr:
        pages = load_document(path)
    assert ocr.call_count == 2
    assert [(p.source, p.page) for p in pages] == [
        ("scanned.pdf", 1),
        ("scanned.pdf", 2),
    ]
    assert "বাংলা" in pages[0].text
    assert pages[1].text == "Second page"


def test_blank_scan_without_text_fails_clearly(tmp_path: Path) -> None:
    """A blank OCR result should not create an empty searchable index."""
    path = tmp_path / "blank.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=400, height=600)
    with path.open("wb") as stream:
        writer.write(stream)
    with patch("src.ingestion.loaders._ocr_page", return_value=""):
        try:
            load_document(path)
        except ValueError as exc:
            assert "No extractable text" in str(exc)
        else:
            raise AssertionError("Expected a clear error for a blank PDF")


def test_reupload_uses_cached_ocr_without_reprocessing(tmp_path: Path, monkeypatch) -> None:
    """The same scanned PDF should be OCR'd once across separate sessions."""
    monkeypatch.setenv("STUDYRAG_OCR_CACHE_DIR", str(tmp_path / "ocr-cache"))
    pdf = tmp_path / "book.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=400, height=600)
    with pdf.open("wb") as stream:
        writer.write(stream)
    with patch("src.ingestion.loaders._ocr_page", return_value="বাংলা বইয়ের লেখা") as ocr:
        first = load_document(pdf)
        second = load_document(pdf)
    assert first == second
    assert ocr.call_count == 1
