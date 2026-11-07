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
    assert [citation.page for citation in rag.ask("Question?").citations] == [4]


def test_debug_capture_for_matching_passage() -> None:
    """Expose the actual retrieval distance and model response for UI debugging."""
    rag, _, _, _ = _pipeline([_match("Stored vectors.", distance=1.1261)])
    rag.max_distance = 2.0
    assert rag.ask("What does ChromaDB store?").citations
    assert rag.last_matches[0]["distance"] == 1.1261
    assert rag.last_model_response == "Vectors are stored. [1]"


def test_debug_capture_without_passages() -> None:
    """When retrieval is empty, no model answer should be recorded."""
    rag, _, _, _ = _pipeline([])
    assert rag.ask("Unknown?").answer == NOT_FOUND
    assert rag.last_matches == []
    assert rag.last_model_response is None


def test_citations_include_only_passages_referenced_by_model() -> None:
    """Model citing second hit must not display unrelated first or third hits."""
    rag, _, _, _ = _pipeline(
        [_match("first"), {**_match("second"), "page": 5},
         {**_match("third"), "page": 7}],
        "Relevant fact [2]",
    )
    result = rag.ask("What?")
    assert result.answer == "Relevant fact [1]"
    assert [item.page for item in result.citations] == [5]


def test_author_name_must_appear_in_cited_evidence() -> None:
    """Reject fluent but unsupported author attributions."""
    rag, _, _, _ = _pipeline(
        [_match("কাকতাড়ুয়া\\nসত্যজিৎ রায়\\nগল্পের শুরু", 0.1)],
        "কাকতাড়ুয়া গল্পের লেখক রবীন্দ্রনাথ ঠাকুর [1]",
    )
    assert not rag.ask("কাকতাড়ুয়া গল্পের লেখক কে?").citations


def test_author_attribution_accepts_matching_source() -> None:
    """Source-backed Bengali author answer retains its citation."""
    rag, _, _, _ = _pipeline(
        [_match("কাকতাড়ুয়া\\nসত্যজিৎ রায়\\nগল্পের শুরু", 0.1)],
        "কাকতাড়ুয়া গল্পের লেখক সত্যজিৎ রায় [1]",
    )
    assert rag.ask("কাকতাড়ুয়া গল্পের লেখক কে?").citations[0].page == 4


def test_author_answer_rejects_invented_page_number() -> None:
    """A model must not invent a page number outside its cited excerpt."""
    rag, _, _, _ = _pipeline(
        [_match("কাকতাড়ুয়া\\nসত্যজিৎ রায়", 0.1)],
        "কাকতাড়ুয়া গল্পের লেখক সত্যজিৎ রায় ১৭ পৃষ্ঠায় ছিলেন [1]",
    )
    assert not rag.ask("কাকতাড়ুয়া গল্পের লেখক কে?").citations


def test_retrieval_failure_reports_clear_reason() -> None:
    rag, _, _, _ = _pipeline([_match("Irrelevant", 1.8)])
    assert rag.ask("Where?").answer == NOT_FOUND
    assert "relevance threshold" in rag.last_failure_reason


def test_missing_source_marker_reports_reason() -> None:
    rag, _, _, _ = _pipeline([_match("Relevant text")], "Answer without citation.")
    assert rag.ask("What?").answer == NOT_FOUND
    assert "source marker" in rag.last_failure_reason


def test_unverified_author_reports_specific_reason() -> None:
    rag, _, _, _ = _pipeline(
        [_match("কাকতাড়ুয়া\\nসত্যজিৎ রায়", 0.1)],
        "লেখক: রবীন্দ্রনাথ ঠাকুর [1]",
    )
    assert rag.ask("কাকতাড়ুয়া গল্পের লেখক কে?").answer == NOT_FOUND
    assert "author" in rag.last_failure_reason
