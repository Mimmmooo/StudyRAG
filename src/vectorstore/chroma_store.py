"""Persist and search text chunks in a local Chroma collection."""

import hashlib
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

    def count(self) -> int:
        """Return the number of stored chunks."""
        return self.collection.count()

    def clear(self) -> None:
        """Delete stored chunks without dropping the collection."""
        all_ids = self.collection.get(include=[])["ids"]
        if all_ids:
            self.collection.delete(ids=all_ids)
