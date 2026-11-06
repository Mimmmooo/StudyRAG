"""Coordinate document loading, chunking, embedding and storage."""

from pathlib import Path

from src.config import Settings
from src.embeddings.embedder import LocalEmbedder
from src.ingestion.chunker import chunk_pages
from src.ingestion.loaders import load_document
from src.vectorstore.chroma_store import ChromaStore


def ingest_files(
    paths: list[Path],
    settings: Settings,
    embedder: LocalEmbedder,
    store: ChromaStore,
) -> dict[str, int]:
    """Index each supported document and return its chunk count."""
    counts: dict[str, int] = {}
    for path in paths:
        pages = load_document(path)
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        vectors = embedder.embed_documents([chunk.text for chunk in chunks])
        counts[path.name] = store.add_chunks(chunks, vectors)
    return counts
