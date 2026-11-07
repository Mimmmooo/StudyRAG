"""Choose a working default based on available inference services."""

from src.llm.factory import ProviderName


def select_default_provider(
    *,
    ollama_available: bool,
    groq_key: str | None = None,
    gemini_key: str | None = None,
) -> ProviderName:
    """Prefer local Ollama, then configured cloud AI, then keyless retrieval."""
    if ollama_available:
        return "ollama"
    if groq_key:
        return "groq"
    if gemini_key:
        return "gemini"
    return "extractive"
