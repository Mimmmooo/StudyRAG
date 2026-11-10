"""Cached bilingual keyword retrieval, ranking and update regressions."""

from pathlib import Path
from unittest.mock import patch

import pytest

from src.ingestion.chunker import TextChunk
from src.vectorstore.chroma_store import ChromaStore


def _add(store: ChromaStore, source: str, page: int, text: str) -> None:
    """Insert a small document without downloading an embedding model."""
    store.add_chunks([TextChunk(text, source, page, 0)], [[1.0, 0.0]])


def test_bengali_exact_title_beats_unrelated_page(tmp_path: Path) -> None:
    """A searched story title ranks above generic Bengali author passages."""
    store = ChromaStore(tmp_path / "index")
    _add(store, "book.pdf", 8, "একটি গল্পের লেখক সম্পর্কে আলোচনা।")
    _add(store, "book.pdf", 15, "কাকতাড়ুয়া\nসত্যজিৎ রায়\nগল্পটি শুরু হয়।")
    result = store.search_keywords("কাকতাড়ুয়া গল্পের লেখক কে?", limit=2)
    assert result[0]["page"] == 15
    assert "সত্যজিৎ রায়" in result[0]["text"]


def test_keyword_search_never_matches_partial_words(tmp_path: Path) -> None:
    """A keyword must not match an unrelated word containing its letters."""
    store = ChromaStore(tmp_path / "index")
    _add(store, "notes.txt", 1, "catapult")
    assert store.search_keywords("cat", limit=1) == []


def test_english_question_finds_exact_matching_terms(tmp_path: Path) -> None:
    """The lexical fallback also works for English documents."""
    store = ChromaStore(tmp_path / "index")
    _add(store, "notes.txt", 1, "ChromaDB stores vectors for similarity search.")
    results = store.search_keywords("What does ChromaDB store?", limit=1)
    assert results[0]["source"] == "notes.txt"


def test_cache_does_not_reload_collection_for_every_question(tmp_path: Path) -> None:
    """Repeated questions reuse the cached document/token index."""
    store = ChromaStore(tmp_path / "index")
    _add(store, "book.txt", 1, "বাংলা বইয়ের বিজ্ঞান শিক্ষা")
    assert store.search_keywords("বিজ্ঞান")
    with patch.object(store.collection, "get", side_effect=AssertionError("reloaded")):
        assert store.search_keywords("বিজ্ঞান")


def test_replacing_document_invalidates_keyword_cache(tmp_path: Path) -> None:
    """Replacing a PDF must not keep stale excerpts or title matches."""
    store = ChromaStore(tmp_path / "index")
    _add(store, "book.pdf", 1, "কাকতাড়ুয়া")
    assert store.search_keywords("কাকতাড়ুয়া")
    store.delete_source("book.pdf")
    _add(store, "book.pdf", 1, "বৃষ্টির গান")
    assert store.search_keywords("কাকতাড়ুয়া") == []
    assert store.search_keywords("বৃষ্টির") 


def test_clear_rebuilds_the_keyword_index(tmp_path: Path) -> None:
    """Clearing notes also clears lexical search evidence."""
    store = ChromaStore(tmp_path / "index")
    _add(store, "book.pdf", 1, "আকাশ")
    assert store.search_keywords("আকাশ")
    store.clear()
    assert store.search_keywords("আকাশ") == []


def test_keyword_search_rejects_invalid_limit(tmp_path: Path) -> None:
    """Reject invalid result counts before accessing the vector database."""
    store = ChromaStore(tmp_path / "index")
    with pytest.raises(ValueError, match="positive"):
        store.search_keywords("আকাশ", limit=0)
