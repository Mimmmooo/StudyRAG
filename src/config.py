"""Environment-backed, validated StudyRAG settings."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    """Non-secret runtime settings for local development."""

    chroma_dir: Path = Path(os.getenv("STUDYRAG_CHROMA_DIR", ".chroma"))
    embedding_model: str = os.getenv(
        "STUDYRAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
    )
    chunk_size: int = int(os.getenv("STUDYRAG_CHUNK_SIZE", "800"))
    chunk_overlap: int = int(os.getenv("STUDYRAG_CHUNK_OVERLAP", "120"))
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen2.5:3b")

    def __post_init__(self) -> None:
        """Reject invalid chunk configurations early."""
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero")
        if not 0 <= self.chunk_overlap < self.chunk_size:
            raise ValueError("chunk_overlap must be between 0 and chunk_size - 1")
