"""Offline unit tests for grounded retrieval and citations."""

from unittest.mock import Mock

import pytest

from src.rag.pipeline import NOT_FOUND, RAGPipeline


def _match(text: str, distance: float = 0.2) -> dict:
    return {
        "text": text,
        "source": "lecture.pdf",
        "page": 4,
        "distance": distance,
    }


def _pipeline(matches: list[dict], answer: str = "Vectors are stored. [1]") -> tuple:
    embedder = Mock()
    embedder.embed_query.return_value = [0.1, 0.2]
    store = Mock()
    store.search.return_value = matches
    llm = Mock()
    llm.generate.return_value = answer
    return RAGPipeline(embedder, store, llm), embedder, store, llm


def test_answer_contains_source_page() -> None:
    """Retrieved passages are included in the model input."""
    rag, embedder, store, llm = _pipeline([_match("ChromaDB stores vectors.")])
    result = rag.ask("What does ChromaDB store?")
    assert result.answer == "Vectors are stored. [1]"
    assert result.citations[0].source == "lecture.pdf"
    assert result.citations[0].page == 4
    assert "ChromaDB stores vectors." in llm.generate.call_args.args[0]
    assert "File: lecture.pdf, page 4" in llm.generate.call_args.args[0]
    embedder.embed_query.assert_called_once()
    store.search.assert_called_once_with([0.1, 0.2], limit=3)


def test_no_matches_abstains() -> None:
    """Without retrieved notes, no model call is needed."""
    rag, _, _, llm = _pipeline([])
    result = rag.ask("Unknown question?")
    assert result.answer == NOT_FOUND
    assert result.citations == ()
    llm.generate.assert_not_called()


def test_distant_matches_abstain() -> None:
    """Distant results are filtered out."""
    rag, _, _, llm = _pipeline([_match("Unrelated.", 1.7)])
    assert rag.ask("Question?").answer == NOT_FOUND
    llm.generate.assert_not_called()


def test_model_abstention_removes_citations() -> None:
    """An abstention does not present supporting citations."""
    rag, _, _, _ = _pipeline([_match("Unrelated.")], NOT_FOUND)
    assert rag.ask("Question?").citations == ()


def test_empty_model_answer_abstains() -> None:
    """Blank generated text is not a useful answer."""
    rag, _, _, _ = _pipeline([_match("Content")], "  ")
    assert rag.ask("Question?").answer == NOT_FOUND


def test_blank_question_rejected() -> None:
    """Whitespace-only questions are invalid."""
    rag, _, _, _ = _pipeline([])
    with pytest.raises(ValueError, match="empty"):
        rag.ask("  ")


@pytest.mark.parametrize("top_k,max_distance", [(0, 1.0), (3, -0.1)])
def test_bad_retrieval_settings(top_k: int, max_distance: float) -> None:
    """Reject invalid retrieval settings."""
    with pytest.raises(ValueError):
        RAGPipeline(Mock(), Mock(), Mock(), top_k=top_k, max_distance=max_distance)


def test_multiple_citations_keep_pages() -> None:
    """Each passage keeps its own page reference."""
    rag, _, _, _ = _pipeline(
        [_match("First"), {**_match("Second"), "page": 5}]
    )
    assert [citation.page for citation in rag.ask("Question?").citations] == [4, 5]


def test_debug_capture_for_matching_passage() -> None:
    """Expose the actual retrieval distance and model response for UI debugging."""
    rag, _, _, _ = _pipeline([_match("Stored vectors.", distance=1.1261)])
    assert rag.ask("What does ChromaDB store?").citations
    assert rag.last_matches[0]["distance"] == 1.1261
    assert rag.last_model_response == "Vectors are stored. [1]"


def test_debug_capture_without_passages() -> None:
    """When retrieval is empty, no model answer should be recorded."""
    rag, _, _, _ = _pipeline([])
    assert rag.ask("Unknown?").answer == NOT_FOUND
    assert rag.last_matches == []
    assert rag.last_model_response is None
