"""Split page-aware documents into overlapping chunks."""

from dataclasses import dataclass

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.ingestion.loaders import DocumentPage


@dataclass(frozen=True)
class TextChunk:
    """A text fragment that retains its source and page."""

    text: str
    source: str
    page: int
    chunk_index: int


def chunk_pages(
    pages: list[DocumentPage], chunk_size: int = 800, chunk_overlap: int = 120
) -> list[TextChunk]:
    """Split each page separately so citations never cross page boundaries."""
    if chunk_size <= 0 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Require chunk_size > 0 and 0 <= chunk_overlap < chunk_size")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks: list[TextChunk] = []
    for page in pages:
        for index, text in enumerate(splitter.split_text(page.text)):
            if text.strip():
                chunks.append(
                    TextChunk(text=text, source=page.source, page=page.page, chunk_index=index)
                )
    return chunks
