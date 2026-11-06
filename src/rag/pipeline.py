"""Retrieve study passages and generate source-aware answers."""

from typing import Protocol

from src.llm.base import BaseLLMProvider
from src.rag.models import Citation, RAGAnswer
from src.vectorstore.chroma_store import ChromaStore

NOT_FOUND = "I couldn't find an answer in the uploaded notes."


class QueryEmbedder(Protocol):
    """Encode questions with the same model used for indexing."""

    def embed_query(self, text: str) -> list[float]:
        """Encode one question."""


class RAGPipeline:
    """Combine retrieval with a provider-neutral text generator."""

    def __init__(
        self,
        embedder: QueryEmbedder,
        store: ChromaStore,
        llm: BaseLLMProvider,
        *,
        top_k: int = 3,
        max_distance: float = 1.0,
    ) -> None:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if max_distance < 0:
            raise ValueError("max_distance must not be negative")
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.top_k = top_k
        self.max_distance = max_distance
        self.last_matches: list[dict] = []
        self.last_model_response: str | None = None

    def ask(self, question: str) -> RAGAnswer:
        """Answer from nearby notes or return a clear no-answer result."""
        if not question.strip():
            raise ValueError("Question cannot be empty")
        matches = self.store.search(
            self.embedder.embed_query(question), limit=self.top_k
        )
        self.last_matches = matches
        self.last_model_response = None
        relevant = [
            item
            for item in matches
            if isinstance(item.get("distance"), (float, int))
            and item["distance"] <= self.max_distance
            and isinstance(item.get("text"), str)
            and item["text"].strip()
        ]
        if not relevant:
            return RAGAnswer(NOT_FOUND, ())
        citations = tuple(
            Citation(item["source"], item["page"], item["text"])
            for item in relevant
        )
        passages = "\n\n".join(
            f"[{index}] File: {item.source}, page {item.page}\n{item.text}"
            for index, item in enumerate(citations, start=1)
        )
        prompt = (
            "Use only the study passages below to answer the question. "
            "If the passages do not contain an answer, say exactly: "
            f"{NOT_FOUND}\n"
            "Cite supporting passages using [1], [2], etc. "
            "Do not add unsupported facts.\n\n"
            f"Study passages:\n{passages}\n\n"
            f"Question: {question}\nAnswer:"
        )
        answer = self.llm.generate(prompt).strip()
        self.last_model_response = answer
        if not answer or NOT_FOUND.lower() in answer.lower():
            return RAGAnswer(NOT_FOUND, ())
        return RAGAnswer(answer, citations)
