"""Offline coverage of the no-key, no-network retrieval mode."""

from unittest.mock import Mock

from src.llm.extractive_provider import ExtractiveProvider
from src.llm.factory import create_provider
from src.rag.pipeline import RAGPipeline


def test_extractive_provider_is_available_without_key() -> None:
    assert isinstance(create_provider("extractive"), ExtractiveProvider)


def test_extractive_provider_returns_numbered_source() -> None:
    embedder = Mock()
    embedder.embed_query.return_value = [0.1, 0.2]
    store = Mock()
    store.search.return_value = [
        {
            "text": "Photosynthesis converts light into chemical energy.",
            "source": "biology.pdf",
            "page": 4,
            "distance": 0.2,
        }
    ]
    result = RAGPipeline(embedder, store, ExtractiveProvider()).ask("Photosynthesis?")
    assert "Photosynthesis converts light" in result.answer
    assert result.answer.endswith("[1]")
    assert result.citations[0].page == 4
