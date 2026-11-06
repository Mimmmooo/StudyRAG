"""Construct a model provider without coupling callers to its implementation."""

import os
from typing import Literal

from src.config import Settings
from src.llm.base import BaseLLMProvider, LLMError
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.ollama_provider import OllamaProvider

ProviderName = Literal["ollama", "gemini", "groq"]


def create_provider(
    provider: ProviderName,
    *,
    api_key: str | None = None,
    model: str | None = None,
    settings: Settings | None = None,
) -> BaseLLMProvider:
    """Choose a provider; explicit BYOK keys override environment variables."""
    settings = settings or Settings()
    if provider == "ollama":
        return OllamaProvider(
            model=model or settings.ollama_model,
            base_url=settings.ollama_base_url,
        )
    if provider == "gemini":
        key = api_key if api_key is not None else os.getenv("GEMINI_API_KEY", "")
        return GeminiProvider(key, model=model or "gemini-2.5-flash")
    if provider == "groq":
        key = api_key if api_key is not None else os.getenv("GROQ_API_KEY", "")
        return GroqProvider(key, model=model or "llama-3.1-8b-instant")
    raise LLMError(f"Unknown LLM provider: {provider}")
