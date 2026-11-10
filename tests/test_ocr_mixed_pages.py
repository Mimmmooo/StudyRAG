"""Realistic OCR detection tests for scanned pages with a sparse text layer."""

from pathlib import Path
from unittest.mock import patch

import fitz

from src.ingestion.loaders import load_document


def _picture() -> bytes:
    """Create a tiny synthetic image without external files or OCR engines."""
    pixmap = fitz.Pixmap(fitz.csRGB, fitz.IRect(0, 0, 24, 24), False)
    pixmap.clear_with(235)
    return pixmap.tobytes("png")


def test_full_page_scan_with_selectable_page_number_uses_ocr(
    tmp_path: Path, monkeypatch,
) -> None:
    """A page number must not hide scanned Bengali content."""
    monkeypatch.setenv("STUDYRAG_OCR_CACHE_DIR", str(tmp_path / "cache"))
    pdf = tmp_path / "scanned-with-page-number.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=600)
    page.insert_image(page.rect, stream=_picture())
    page.insert_text((20, 25), "12")
    document.save(pdf)
    document.close()

    with patch(
        "src.ingestion.loaders._ocr_page",
        return_value="কাকতাড়ুয়া গল্পের লেখক সত্যজিৎ রায়",
    ) as ocr:
        first = load_document(pdf)
        second = load_document(pdf)

    assert [(item.source, item.page) for item in first] == [(pdf.name, 1)]
    assert "সত্যজিৎ রায়" in first[0].text
    assert second == first
    ocr.assert_called_once()


def test_small_image_on_text_pdf_does_not_force_ocr(
    tmp_path: Path, monkeypatch,
) -> None:
    """A logo on an otherwise readable short page is not a scanned page."""
    monkeypatch.setenv("STUDYRAG_OCR_CACHE_DIR", str(tmp_path / "cache"))
    pdf = tmp_path / "text-with-logo.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=600)
    page.insert_image(fitz.Rect(30, 40, 70, 80), stream=_picture())
    page.insert_text((25, 120), "Physics is the study of matter.")
    document.save(pdf)
    document.close()

    with patch("src.ingestion.loaders._ocr_page") as ocr:
        pages = load_document(pdf)

    assert "Physics is the study of matter." in pages[0].text
    ocr.assert_not_called()


def test_full_page_background_with_meaningful_text_keeps_text_layer(
    tmp_path: Path, monkeypatch,
) -> None:
    """A large background photo should not trigger OCR on readable notes."""
    monkeypatch.setenv("STUDYRAG_OCR_CACHE_DIR", str(tmp_path / "cache"))
    pdf = tmp_path / "text-over-background.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=600)
    page.insert_image(page.rect, stream=_picture())
    for line in range(6):
        page.insert_text(
            (25, 35 + 22 * line),
            f"Lesson number {line}: this page contains searchable lecture notes.",
            fontsize=9,
        )
    document.save(pdf)
    document.close()

    with patch("src.ingestion.loaders._ocr_page") as ocr:
        pages = load_document(pdf)

    assert "searchable lecture notes." in pages[0].text
    ocr.assert_not_called()


def test_short_ocr_result_does_not_discard_existing_text(
    tmp_path: Path, monkeypatch,
) -> None:
    """Poor OCR should not replace a more useful text-layer snippet."""
    monkeypatch.setenv("STUDYRAG_OCR_CACHE_DIR", str(tmp_path / "cache"))
    pdf = tmp_path / "mixed.pdf"
    document = fitz.open()
    page = document.new_page(width=400, height=600)
    page.insert_image(page.rect, stream=_picture())
    page.insert_text((20, 25), "Section 12")
    document.save(pdf)
    document.close()

    with patch("src.ingestion.loaders._ocr_page", return_value="") as ocr:
        pages = load_document(pdf)

    assert pages[0].text == "Section 12"
    ocr.assert_called_once()
