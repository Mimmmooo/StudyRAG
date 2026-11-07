"""Coordinate document loading, chunking, embedding and storage."""

from pathlib import Path

from src.config import Settings
from src.embeddings.embedder import LocalEmbedder
from src.ingestion.chunker import chunk_pages
from src.ingestion.loaders import load_document
from src.vectorstore.chroma_store import ChromaStore

EMBED_BATCH_SIZE = 32


def _embed_in_batches(
    texts: list[str], embedder: LocalEmbedder
) -> list[list[float]]:
    """Avoid embedding a large textbook in one expensive model call."""
    vectors: list[list[float]] = []
    for start in range(0, len(texts), EMBED_BATCH_SIZE):
        vectors.extend(
            embedder.embed_documents(texts[start : start + EMBED_BATCH_SIZE])
        )
    return vectors


def ingest_files(
    paths: list[Path],
    settings: Settings,
    embedder: LocalEmbedder,
    store: ChromaStore,
) -> dict[str, int]:
    """Index supported documents and preserve the previous index on parse failures."""
    if len({path.name for path in paths}) != len(paths):
        raise ValueError("Document filenames must be unique.")
    prepared = []
    for path in paths:
        pages = load_document(path)
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        vectors = _embed_in_batches([chunk.text for chunk in chunks], embedder)
        prepared.append((path.name, chunks, vectors))

    counts: dict[str, int] = {}
    for name, chunks, vectors in prepared:
        store.delete_source(name)
        for start in range(0, len(chunks), EMBED_BATCH_SIZE):
            store.add_chunks(
                chunks[start : start + EMBED_BATCH_SIZE],
                vectors[start : start + EMBED_BATCH_SIZE],
            )
        counts[name] = len(chunks)
    return counts
