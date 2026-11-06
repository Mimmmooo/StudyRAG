"""RAG result models."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Citation:
    """Source reference for an answer."""

    source: str
    page: int
    text: str


@dataclass(frozen=True)
class RAGAnswer:
    """Answer with source references."""

    answer: str
    citations: tuple[Citation, ...]
