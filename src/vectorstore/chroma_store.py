"""Persist and search text chunks in a local Chroma collection."""

import hashlib
import math
import re
import unicodedata
from pathlib import Path

import chromadb

from src.ingestion.chunker import TextChunk

_STOPWORDS = {
    "গল্পের", "লেখক", "লেখকের", "কে", "কী", "কি", "হয়", "সম্পর্কে", "বলো",
    "what", "who", "the", "does", "where", "when", "which",
}


def _terms(text: str) -> list[str]:
    """Tokenize English and Bengali without matching unrelated substrings."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    return re.findall(r"[\w\u0980-\u09ff]+", normalized)


def _query_terms(text: str) -> list[str]:
    """Keep distinct meaningful terms in their original order."""
    return list(dict.fromkeys(
        term for term in _terms(text)
        if len(term) >= 3 and term not in _STOPWORDS
    ))


class ChromaStore:
    """Chroma storage with deterministic document chunk identifiers."""

    def __init__(self, persist_dir: Path, collection_name: str = "studyrag") -> None:
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        self._keyword_rows: list[tuple[str, dict, list[str]]] | None = None
        self._postings: dict[str, list[int]] = {}

    def add_chunks(self, chunks: list[TextChunk], vectors: list[list[float]]) -> int:
        """Upsert embeddings and page metadata into Chroma."""
        if len(chunks) != len(vectors):
            raise ValueError("Chunk and embedding counts differ")
        if not chunks:
            return 0
        ids = [
            hashlib.sha256(
                f"{chunk.source}:{chunk.page}:{chunk.chunk_index}:{chunk.text}".encode()
            ).hexdigest()
            for chunk in chunks
        ]
        self.collection.upsert(
            ids=ids,
            documents=[chunk.text for chunk in chunks],
            embeddings=vectors,
            metadatas=[
                {"source": chunk.source, "page": chunk.page, "chunk_index": chunk.chunk_index}
                for chunk in chunks
            ],
        )
        self._keyword_rows = None
        return len(chunks)

    def delete_source(self, source: str) -> None:
        """Remove all indexed chunks for a document before replacing it."""
        matches = self.collection.get(where={"source": source}, include=[])
        if matches["ids"]:
            self.collection.delete(ids=matches["ids"])
            self._keyword_rows = None

    def search(self, query_vector: list[float], limit: int = 3) -> list[dict]:
        """Return matching chunks and their citation metadata."""
        if limit < 1:
            raise ValueError("limit must be positive")
        if self.collection.count() == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=min(limit, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        return [
            {"text": text, "source": meta["source"], "page": meta["page"], "distance": dist}
            for text, meta, dist in zip(
                result["documents"][0], result["metadatas"][0], result["distances"][0]
            )
        ]

    def _build_keyword_index(self) -> None:
        """Cache an inverted index so questions avoid repeated full Chroma reads."""
        data = self.collection.get(include=["documents", "metadatas"])
        rows: list[tuple[str, dict, list[str]]] = []
        postings: dict[str, list[int]] = {}
        for document, metadata in zip(data["documents"], data["metadatas"]):
            terms = _terms(document)
            position = len(rows)
            rows.append((document, metadata, terms))
            for token in set(terms):
                postings.setdefault(token, []).append(position)
        self._keyword_rows = rows
        self._postings = postings

    def search_keywords(self, question: str, limit: int = 3) -> list[dict]:
        """Rank exact Bengali/English terms using cached, length-aware evidence."""
        if limit < 1:
            raise ValueError("limit must be positive")
        query = _query_terms(question)
        if not query:
            return []
        if self._keyword_rows is None:
            self._build_keyword_index()
        rows = self._keyword_rows
        if not rows:
            return []
        candidates = set().union(
            *(self._postings.get(term, []) for term in query)
        )
        total = len(rows)
        ranked: list[tuple[float, dict]] = []
        for index in candidates:
            document, metadata, terms = rows[index]
            occurrences = set(terms)
            matched = [term for term in query if term in occurrences]
            if not matched:
                continue
            score = sum(
                math.log((total + 1) / (len(self._postings[term]) + 1)) + 1
                for term in matched
            )
            score += 2 * len(matched) / len(query)
            # Prefer a title containing query words together over scattered hits.
            if len(query) > 1 and " ".join(query) in " ".join(terms):
                score += 3
            ranked.append((score, {
                "text": document,
                "source": metadata["source"],
                "page": metadata["page"],
                "distance": 0.0,
            }))
        ranked.sort(
            key=lambda item: (
                -item[0], item[1]["source"], item[1]["page"]
            )
        )
        return [item for _, item in ranked[:limit]]

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete stored chunks without dropping the collection."""
        all_ids = self.collection.get(include=[])["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
        self._keyword_rows = None
