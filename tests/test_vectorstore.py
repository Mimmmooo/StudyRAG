"""Chroma integration tests using deterministic fake vectors."""

from pathlib import Path

import pytest

from src.ingestion.chunker import TextChunk
from src.vectorstore.chroma_store import ChromaStore


def test_upsert_search_clear(tmp_path: Path) -> None:
    """Chunks can be stored, searched and deleted locally."""
    store = ChromaStore(tmp_path / "chroma")
    chunk = TextChunk("Photosynthesis uses light.", "biology.txt", 1, 0)
    assert store.add_chunks([chunk], [[1.0, 0.0, 0.0]]) == 1
    assert store.add_chunks([chunk], [[1.0, 0.0, 0.0]]) == 1
    assert store.count() == 1
    result = store.search([1.0, 0.0, 0.0])[0]
    assert result["source"] == "biology.txt"
    assert result["page"] == 1
    store.clear()
    assert store.count() == 0


def test_mismatched_vectors(tmp_path: Path) -> None:
    """A missing embedding cannot silently corrupt storage."""
    store = ChromaStore(tmp_path / "chroma")
    with pytest.raises(ValueError, match="counts differ"):
        store.add_chunks([TextChunk("hello", "a.txt", 1, 0)], [])
