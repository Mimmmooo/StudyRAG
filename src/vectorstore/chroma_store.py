"""Persist and search text chunks in a local Chroma collection."""

import hashlib
import re
from pathlib import Path

import chromadb

from src.ingestion.chunker import TextChunk


class ChromaStore:
    """Chroma storage with deterministic document chunk identifiers."""

    def __init__(self, persist_dir: Path, collection_name: str = "studyrag") -> None:
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(name=collection_name)

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
        return len(chunks)

    def delete_source(self, source: str) -> None:
        """Remove all indexed chunks for a document before replacing it."""
        matches = self.collection.get(where={"source": source}, include=[])
        if matches["ids"]:
            self.collection.delete(ids=matches["ids"])

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

    def search_keywords(self, question: str, limit: int = 3) -> list[dict]:
        """Find Bengali/English phrases directly when embeddings miss titles."""
        tokens = [
            token for token in re.findall(r"[\\w\\u0980-\\u09ff]+", question.casefold())
            if len(token) >= 3
            and token not in {"গল্পের", "লেখক", "লেখকের", "কে", "কী", "কি", "হয়", "সম্পর্কে", "বলো", "what", "who", "the", "does"}
        ]
        if not tokens or not self.collection.count():
            return []
        # Retrieve stored text only, and never expose source documents externally.
        data = self.collection.get(include=["documents", "metadatas"])
        scored: list[tuple[int, dict]] = []
        for document, metadata in zip(data["documents"], data["metadatas"]):
            normalized = document.casefold()
            score = sum(1 for token in tokens if token in normalized)
            if score:
                scored.append((score, {
                    "text": document, "source": metadata["source"],
                    "page": metadata["page"], "distance": 0.0,
                }))
        scored.sort(key=lambda item: (-item[0], item[1]["page"]))
        return [item for _, item in scored[:limit]]

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete stored chunks without dropping the collection."""
        all_ids = self.collection.get(include=[])["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
