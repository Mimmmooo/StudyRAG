"""Provider-neutral language-model interface."""

from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """A safe, user-facing language-model request failure."""


class BaseLLMProvider(ABC):
    """One contract for local and cloud language models."""

    @abstractmethod
    def generate(self, prompt: str) -> str:
        """Return model-generated text for a prompt."""
