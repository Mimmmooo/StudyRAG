"""Bengali textbook title retrieval without a paid API or external network."""

from pathlib import Path
from unittest.mock import Mock

from src.ingestion.chunker import TextChunk
from src.llm.extractive_provider import ExtractiveProvider
from src.rag.evidence import explicit_author_from_passage
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


def test_explicit_title_author_from_ocr_heading() -> None:
    """A short standalone author line immediately after the title is evidence."""
    text = "কাকতাড়ুয়া\nসত্যজিৎ রায়\nএকদিন মাঠের পাশে একটি ছেলে দাঁড়িয়েছিল।"
    assert explicit_author_from_passage("কাকতাড়ুয়া গল্পের লেখক কে?", text) == "সত্যজিৎ রায়"


def test_author_is_not_guessed_from_story_character() -> None:
    """A character mentioned in story prose is not proof of authorship."""
    text = "কাকতাড়ুয়া\nএকদিন মাঠে হাঁটছিল রবীন্দ্রনাথ ঠাকুর।\nগল্পের শুরু।"
    assert explicit_author_from_passage("কাকতাড়ুয়া গল্পের লেখক কে?", text) is None


def test_title_author_answer_bypasses_unreliable_model(tmp_path: Path) -> None:
    """Explicit author evidence should not require Qwen3 to format a citation."""
    store = ChromaStore(tmp_path / "explicit-author")
    store.add_chunks(
        [TextChunk("কাকতাড়ুয়া\nসত্যজিৎ রায়\nগল্পের শুরু।", "বই.pdf", 5, 0)],
        [[1.0, 0.0]],
    )
    embedder = Mock()
    embedder.embed_query.return_value = [1.0, 0.0]
    model = Mock()
    result = RAGPipeline(embedder, store, model, top_k=1, max_distance=2.0).ask(
        "কাকতাড়ুয়া গল্পের লেখক কে?"
    )
    assert result.answer == "লেখক: সত্যজিৎ রায়"
    assert result.citations[0].page == 5
    model.generate.assert_not_called()


def test_unlabeled_author_in_unrelated_paragraph_is_not_extracted() -> None:
    """Do not assign names merely because they appear near a story word."""
    text = "কাকতাড়ুয়া গল্পটি শুরু হয়।\nএকদিন সত্যজিৎ রায় মাঠে যান।"
    assert explicit_author_from_passage("কাকতাড়ুয়া গল্পের লেখক কে?", text) is None
