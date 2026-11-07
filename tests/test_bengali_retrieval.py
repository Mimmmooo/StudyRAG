"""Bengali textbook title retrieval without a paid API or external network."""

from pathlib import Path
from unittest.mock import Mock

from src.ingestion.chunker import TextChunk
from src.llm.extractive_provider import ExtractiveProvider
from src.rag.pipeline import RAGPipeline
from src.vectorstore.chroma_store import ChromaStore


def test_bengali_title_retrieval_overrides_misleading_vector_rank(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "bengali")
    chunks = [
        TextChunk("কাকতাড়ুয়া\nসত্যজিৎ রায়\nগল্পটি গ্রামে শুরু হয়।", "আনন্দপাঠ.pdf", 6, 0),
        TextChunk("একজন ছেলে গ্রামের পথে হাঁটছিল।", "আনন্দপাঠ.pdf", 7, 0),
    ]
    store.add_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
    embedder = Mock()
    embedder.embed_query.return_value = [0.0, 1.0]
    result = RAGPipeline(
        embedder, store, ExtractiveProvider(), top_k=2, max_distance=2.0
    ).ask("কাকতাড়ুয়া গল্পের লেখক কে?")
    assert result.citations[0].page == 6
    assert "সত্যজিৎ রায়" in result.answer


def test_no_keyword_overlap_does_not_invent_sources(tmp_path: Path) -> None:
    store = ChromaStore(tmp_path / "nohit")
    store.add_chunks([TextChunk("বাংলা সাহিত্য", "test.pdf", 1, 0)], [[1.0, 0.0]])
    assert store.search_keywords("photosynthesis?", limit=3) == []
