"""Phase 8: follow-ups, larger documents, and multi-file regression tests."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.ingestion.chunker import chunk_pages
from src.ingestion.loaders import DocumentPage, load_document
from src.rag.pipeline import NOT_FOUND, RAGPipeline
from src.ui_service import StudySession
from src.vectorstore.chroma_store import ChromaStore


def _pipeline() -> tuple[RAGPipeline, Mock, Mock]:
    """Construct a deterministic RAG pipeline without API calls."""
    embedder = Mock()
    embedder.embed_query.return_value = [1.0, 0.0]
    store = Mock()
    store.search.return_value = [
        {
            "text": "ChromaDB stores vectors.",
            "source": "notes.pdf",
            "page": 3,
            "distance": 0.5,
        }
    ]
    llm = Mock()
    llm.generate.return_value = "It stores vectors. [1]"
    return RAGPipeline(embedder, store, llm), embedder, llm


def test_followup_uses_previous_question_for_retrieval() -> None:
    """A short follow-up carries previous user question into vector search."""
    rag, embedder, llm = _pipeline()
    result = rag.ask(
        "What does it store?",
        history=[
            ("user", "What is ChromaDB?"),
            ("assistant", "It is a vector database."),
        ],
    )
    assert result.citations[0].page == 3
    assert embedder.embed_query.call_args.args[0] == (
        "What is ChromaDB?\nWhat does it store?"
    )
    assert "Previous conversation:" in llm.generate.call_args.args[0]


def test_history_is_bounded() -> None:
    """Old turns and unusually long message bodies must be truncated."""
    rag, embedder, llm = _pipeline()
    history = [("user", "old")] + [("assistant", "X" * 1000)] * 6
    rag.ask("Question", history=history)
    prompt = llm.generate.call_args.args[0]
    assert "old" not in prompt
    assert "X" * 501 not in prompt
    embedder.embed_query.assert_called_once_with("Question")


def test_previous_assistant_answer_is_not_retrieval_query() -> None:
    """Search text uses previous user wording, not model hallucinations."""
    rag, embedder, _ = _pipeline()
    rag.ask(
        "Tell me more",
        history=[("user", "Photosynthesis"), ("assistant", "Unrelated guesses")],
    )
    embedder.embed_query.assert_called_once_with("Photosynthesis\nTell me more")


def test_missing_evidence_does_not_call_llm_for_followup() -> None:
    """Conversation history cannot override the no-document-evidence rule."""
    rag, _, llm = _pipeline()
    rag.store.search.return_value = []
    result = rag.ask("And why?", history=[("assistant", "A prior guess")])
    assert result.answer == NOT_FOUND
    llm.generate.assert_not_called()


def test_many_pages_keep_original_page_number() -> None:
    """A long PDF-like document is chunked independently on each page."""
    pages = [
        DocumentPage(f"Chapter {page}\n" + "Learning data. " * 120, "large.pdf", page)
        for page in range(1, 26)
    ]
    chunks = chunk_pages(pages, chunk_size=160, chunk_overlap=30)
    assert len(chunks) > 100
    assert {chunk.page for chunk in chunks} == set(range(1, 26))
    assert all(chunk.source == "large.pdf" for chunk in chunks)


def test_mixed_file_index_retains_distinct_sources(tmp_path: Path) -> None:
    """Two documents share a vector index but not their citation metadata."""
    from src.ingestion.chunker import TextChunk

    store = ChromaStore(tmp_path / "vectors")
    chunks = [
        TextChunk("Cats study biology", "biology.txt", 1, 0),
        TextChunk("Gravity is physics", "physics.md", 1, 0),
    ]
    store.add_chunks(chunks, [[1.0, 0.0], [0.0, 1.0]])
    matches = store.search([1.0, 0.0], limit=2)
    assert {item["source"] for item in matches} == {"biology.txt", "physics.md"}


def test_missing_pdf_is_useful_error(tmp_path: Path) -> None:
    """Unreadable documents return an actionable error."""
    invalid = tmp_path / "invalid.pdf"
    invalid.write_bytes(b"not a valid pdf")
    with pytest.raises(ValueError, match="Could not read PDF"):
        load_document(invalid)


def test_ui_service_forwards_history_and_top_k() -> None:
    """The service passes UI controls without implementing RAG logic."""
    session = StudySession.__new__(StudySession)
    session.store = Mock()
    session.store.count.return_value = 2
    session.embedder = Mock()
    session.settings = Mock()
    with (
        patch("src.ui_service.create_provider", return_value=Mock()),
        patch("src.ui_service.RAGPipeline") as rag_class,
    ):
        rag_class.return_value.ask.return_value = Mock(answer="answer", citations=())
        rag_class.return_value.last_matches = []
        rag_class.return_value.last_model_response = "answer"
        session.ask(
            "And why?",
            "ollama",
            history=[("user", "Explain vectors")],
            top_k=5,
        )
    assert rag_class.call_args.kwargs == {"top_k": 5, "max_distance": 2.0}
    rag_class.return_value.ask.assert_called_once_with(
        "And why?", history=[("user", "Explain vectors")]
    )
