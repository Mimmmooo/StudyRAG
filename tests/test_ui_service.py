"""Offline tests for the Streamlit-independent application service."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from src.ui_service import MAX_UPLOAD_BYTES, StudySession


def _upload(name: str, data: bytes) -> Mock:
    upload = Mock()
    upload.name = name
    upload.getvalue.return_value = data
    return upload


def test_rejects_unsupported_file() -> None:
    """Only PDF, TXT and Markdown uploads are accepted."""
    session = StudySession.__new__(StudySession)
    with pytest.raises(ValueError, match="Unsupported"):
        session.ingest([_upload("notes.exe", b"hello")])


def test_rejects_large_upload() -> None:
    """Large uploads fail before writing to disk."""
    session = StudySession.__new__(StudySession)
    with pytest.raises(ValueError, match="15 MB"):
        session.ingest([_upload("notes.txt", b"x" * (MAX_UPLOAD_BYTES + 1))])


def test_rejects_duplicate_names() -> None:
    """Avoid mixing citations for files with identical names."""
    session = StudySession.__new__(StudySession)
    with pytest.raises(ValueError, match="unique names"):
        session.ingest([_upload("notes.txt", b"a"), _upload("notes.txt", b"b")])


def test_rejects_empty_upload() -> None:
    """An empty file cannot be indexed."""
    session = StudySession.__new__(StudySession)
    with pytest.raises(ValueError, match="empty"):
        session.ingest([_upload("notes.txt", b"")])


def test_ingestion_deletes_original_upload(tmp_path: Path) -> None:
    """Source files are removed after indexing, leaving only the index."""
    session = StudySession.__new__(StudySession)
    session._root = tmp_path
    session.settings = Mock()
    session.embedder = Mock()
    session.store = Mock()
    session.files = []
    with patch("src.ui_service.ingest_files", return_value={"notes.txt": 1}):
        counts = session.ingest([_upload("notes.txt", b"sample")])
    assert counts == {"notes.txt": 1}
    assert session.files == ["notes.txt"]
    assert not (tmp_path / "notes.txt").exists()


def test_ask_requires_index() -> None:
    """No question is sent to a provider before indexing."""
    session = StudySession.__new__(StudySession)
    session.store = Mock()
    session.store.count.return_value = 0
    with pytest.raises(ValueError, match="Upload and index"):
        session.ask("question", "ollama")
