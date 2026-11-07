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
    session.settings = Mock(max_distance=2.0)
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


def test_reindex_replaces_old_chunks(tmp_path: Path) -> None:
    """Replacing a source removes outdated chunks without touching other files."""
    from src.ingestion.chunker import TextChunk

    store = ChromaStore(tmp_path / "replace")
    store.add_chunks(
        [TextChunk("old content", "notes.txt", 1, 0),
         TextChunk("keep this", "other.txt", 1, 0)],
        [[1.0, 0.0], [0.0, 1.0]],
    )
    store.delete_source("notes.txt")
    store.add_chunks([TextChunk("new content", "notes.txt", 1, 0)], [[1.0, 0.0]])
    documents = store.collection.get(include=["documents"])["documents"]
    assert sorted(documents) == ["keep this", "new content"]


def test_ingestion_prepares_all_files_before_mutating_index(tmp_path: Path) -> None:
    """Bad second file must not erase or replace a previously indexed source."""
    from src.config import Settings
    from src.ingestion.service import ingest_files

    store = Mock()
    embedder = Mock()
    embedder.embed_documents.return_value = [[1.0, 0.0]]
    good = tmp_path / "good.txt"
    good.write_text("Valid notes", encoding="utf-8")
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"not a pdf")
    with pytest.raises(ValueError, match="Could not read PDF"):
        ingest_files([good, bad], Settings(), embedder, store)
    store.delete_source.assert_not_called()
    store.add_chunks.assert_not_called()


@pytest.mark.parametrize("answer", [
    "Fact with invented source [9]",
    "Fact without a citation",
])
def test_ungrounded_citation_markers_are_rejected(answer: str) -> None:
    """Only answers that cite an available passage can be returned."""
    rag, _, llm = _pipeline()
    llm.generate.return_value = answer
    result = rag.ask("What does it store?")
    assert result.answer == NOT_FOUND
    assert result.citations == ()


def test_valid_citation_markers_still_work() -> None:
    """Valid citations pass the new answer validation."""
    rag, _, llm = _pipeline()
    llm.generate.return_value = "Vectors [1]"
    assert rag.ask("What does it store?").answer == "Vectors [1]"


def test_numbered_questions_are_retrieved_independently() -> None:
    """Multiple pasted questions must not share one unrelated vector search."""
    rag, embedder, llm = _pipeline()
    result = rag.ask("1. কোথায় ঘটেছে? 2. মূল চরিত্র কারা?")
    assert embedder.embed_query.call_count == 2
    assert [call.args[0] for call in embedder.embed_query.call_args_list] == [
        "কোথায় ঘটেছে?", "মূল চরিত্র কারা?",
    ]
    assert llm.generate.call_count == 2
    assert result.answer.count("It stores vectors.") == 2
    assert len(result.citations) == 2
    assert "[1]" in result.answer and "[2]" in result.answer


def test_independent_question_ignores_long_previous_batch() -> None:
    """Long numbered user prompts should never contaminate new retrieval."""
    rag, embedder, _ = _pipeline()
    rag.ask(
        "কাকতাড়ুয়া গল্পের ঘটনাগুলো কোথায় ঘটেছে?",
        history=[("user", "1. গল্পের লেখক কে? 2. প্রধান চরিত্র কারা?")],
    )
    embedder.embed_query.assert_called_once_with(
        "কাকতাড়ুয়া গল্পের ঘটনাগুলো কোথায় ঘটেছে?"
    )


def test_single_referential_followup_keeps_context() -> None:
    """Short Bengali pronoun follow-ups still resolve earlier questions."""
    rag, embedder, _ = _pipeline()
    rag.ask("সে কেন এমন করেছিল?", history=[("user", "গল্পের প্রধান চরিত্র কে?")])
    embedder.embed_query.assert_called_once_with(
        "গল্পের প্রধান চরিত্র কে?\\nসে কেন এমন করেছিল?".replace("\\n", "\n")
    )
