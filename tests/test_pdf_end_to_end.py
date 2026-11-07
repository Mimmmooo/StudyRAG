"""Real PDF extraction to indexing regression tests (no external API required)."""

from pathlib import Path
from unittest.mock import Mock

from pypdf import PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

from src.config import Settings
from src.ingestion.service import EMBED_BATCH_SIZE, ingest_files
from src.vectorstore.chroma_store import ChromaStore


def _write_text_pdf(path: Path) -> None:
    """Create a PDF with selectable text, like a typical lecture handout."""
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): font}),
    })
    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 72 720 Td (ChromaDB stores vectors for note retrieval.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    with path.open("wb") as output:
        writer.write(output)


def test_real_pdf_is_extracted_embedded_and_cited(tmp_path: Path) -> None:
    """Actual PDF text should flow through ingestion into searchable Chroma."""
    pdf = tmp_path / "lecture.pdf"
    _write_text_pdf(pdf)
    embedder = Mock()
    embedder.embed_documents.side_effect = lambda texts: [[1.0, 0.0] for _ in texts]
    store = ChromaStore(tmp_path / "index")
    counts = ingest_files([pdf], Settings(), embedder, store)
    assert counts["lecture.pdf"] >= 1
    result = store.search([1.0, 0.0])[0]
    assert "ChromaDB stores vectors" in result["text"]
    assert result["source"] == "lecture.pdf"
    assert result["page"] == 1


def test_large_embedding_work_is_bounded_in_batches() -> None:
    """Never submit an entire large document as one embedding request."""
    from src.ingestion.service import _embed_in_batches

    embedder = Mock()
    embedder.embed_documents.side_effect = lambda texts: [[1.0] for _ in texts]
    vectors = _embed_in_batches(["test"] * (EMBED_BATCH_SIZE * 3 + 1), embedder)
    assert len(vectors) == EMBED_BATCH_SIZE * 3 + 1
    assert embedder.embed_documents.call_count == 4
    assert all(len(call.args[0]) <= EMBED_BATCH_SIZE for call in embedder.embed_documents.call_args_list)
