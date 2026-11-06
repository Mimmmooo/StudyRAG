"""Fast unit tests for loading and splitting documents."""

from pathlib import Path

import pytest

from src.ingestion.chunker import chunk_pages
from src.ingestion.loaders import DocumentPage, load_document


def test_load_txt(tmp_path: Path) -> None:
    """Plain text retains filename and page-one metadata."""
    path = tmp_path / "lecture.txt"
    path.write_text("  Machine learning\n\n Regression  ", encoding="utf-8")
    assert load_document(path) == [
        DocumentPage("Machine learning\nRegression", "lecture.txt", 1)
    ]


def test_load_md(tmp_path: Path) -> None:
    """Markdown is supported without network calls."""
    path = tmp_path / "lecture.md"
    path.write_text("# Week 1", encoding="utf-8")
    assert load_document(path)[0].text == "# Week 1"


def test_empty_file_rejected(tmp_path: Path) -> None:
    """Empty documents should report a useful error."""
    path = tmp_path / "empty.txt"
    path.write_text(" \n ", encoding="utf-8")
    with pytest.raises(ValueError, match="No extractable text"):
        load_document(path)


def test_unsupported_file_rejected(tmp_path: Path) -> None:
    """Only supported types should be accepted."""
    path = tmp_path / "notes.csv"
    path.write_text("a,b", encoding="utf-8")
    with pytest.raises(ValueError, match="Unsupported"):
        load_document(path)


def test_chunks_keep_page_metadata() -> None:
    """Splitting never changes the original page reference."""
    pages = [DocumentPage("A " * 80, "notes.pdf", 7)]
    chunks = chunk_pages(pages, chunk_size=40, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(chunk.source == "notes.pdf" and chunk.page == 7 for chunk in chunks)


def test_bad_chunk_settings() -> None:
    """Overlap must be smaller than chunk size."""
    with pytest.raises(ValueError):
        chunk_pages([], chunk_size=10, chunk_overlap=10)
