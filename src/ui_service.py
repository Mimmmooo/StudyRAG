"""Application operations used by the Streamlit presentation layer."""

import tempfile
from pathlib import Path
from typing import Protocol

from src.config import Settings
from src.embeddings.embedder import LocalEmbedder
from src.ingestion.service import ingest_files
from src.llm.factory import ProviderName, create_provider
from src.rag.models import RAGAnswer
from src.rag.pipeline import RAGPipeline
from src.vectorstore.chroma_store import ChromaStore

MAX_UPLOAD_BYTES = 15 * 1024 * 1024


class UploadedFile(Protocol):
    """Minimum interface required from a Streamlit uploaded file."""

    name: str

    def getvalue(self) -> bytes:
        """Read the uploaded file contents."""


class StudySession:
    """Keep uploaded notes in a private temporary index for one browser session."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()
        self._temporary = tempfile.TemporaryDirectory(prefix="studyrag-")
        self._root = Path(self._temporary.name)
        self.store = ChromaStore(self._root / "chroma")
        self.embedder = LocalEmbedder(self.settings.embedding_model)
        self.files: list[str] = []
        self.last_debug: dict[str, object] = {}

    def ingest(self, uploads: list[UploadedFile]) -> dict[str, int]:
        """Validate and index uploaded notes without persisting originals."""
        if not uploads:
            raise ValueError("Choose at least one PDF, TXT or MD file.")
        names = [Path(upload.name).name for upload in uploads]
        if len(names) != len(set(names)):
            raise ValueError("Uploaded files must have unique names.")
        payloads: list[tuple[str, bytes]] = []
        for upload, name in zip(uploads, names):
            if Path(name).suffix.lower() not in {".pdf", ".txt", ".md"}:
                raise ValueError(f"Unsupported file type: {name}")
            data = upload.getvalue()
            if len(data) > MAX_UPLOAD_BYTES:
                raise ValueError(f"{name} exceeds the 15 MB limit.")
            if not data:
                raise ValueError(f"{name} is empty.")
            payloads.append((name, data))
        paths: list[Path] = []
        for name, data in payloads:
            path = self._root / name
            path.write_bytes(data)
            paths.append(path)
        try:
            counts = ingest_files(paths, self.settings, self.embedder, self.store)
        finally:
            for path in paths:
                path.unlink(missing_ok=True)
        self.files = sorted(set(self.files).union(counts))
        self.last_debug = {}
        return counts

    def ask(
        self,
        question: str,
        provider: ProviderName,
        api_key: str | None = None,
        history: list[tuple[str, str]] | None = None,
        top_k: int = 3,
    ) -> RAGAnswer:
        """Run grounded retrieval using the selected LLM provider."""
        if self.store.count() == 0:
            raise ValueError("Upload and index notes before asking questions.")
        llm = create_provider(
            provider, api_key=api_key, settings=self.settings
        )
        pipeline = RAGPipeline(
            self.embedder, self.store, llm, top_k=top_k, max_distance=self.settings.max_distance
        )
        self.last_debug = {}
        result = pipeline.ask(question, history=history)
        self.last_debug = {
            "indexed_chunks": self.store.count(),
            "matches": [
                {
                    "source": item.get("source"),
                    "page": item.get("page"),
                    "distance": round(item["distance"], 4),
                }
                for item in pipeline.last_matches
                if isinstance(item.get("distance"), (int, float))
            ],
            "model_answer": pipeline.last_model_response,
            "threshold": pipeline.max_distance,
        }
        return result

    def close(self) -> None:
        """Remove the session's temporary document index."""
        self._temporary.cleanup()
