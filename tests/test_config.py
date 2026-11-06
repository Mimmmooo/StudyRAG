"""Tests for Phase 1 configuration."""

from pathlib import Path

import pytest

from src.config import Settings


def test_settings_are_valid() -> None:
    """A valid configuration should be usable."""
    settings = Settings(chroma_dir=Path(".chroma"), chunk_size=800, chunk_overlap=120)
    assert settings.chunk_overlap < settings.chunk_size


@pytest.mark.parametrize("size,overlap", [(0, 0), (100, 100), (100, -1)])
def test_invalid_chunk_settings_raise(size: int, overlap: int) -> None:
    """Reject chunk settings that could break splitting."""
    with pytest.raises(ValueError):
        Settings(chunk_size=size, chunk_overlap=overlap)
