"""Check which answer mode is available without exposing credentials."""

from src.provider_selection import select_default_provider


def test_local_ollama_is_preferred() -> None:
    assert select_default_provider(ollama_available=True, groq_key="secret") == "ollama"


def test_hosted_app_prefers_configured_ai_provider() -> None:
    assert select_default_provider(ollama_available=False, groq_key="secret") == "groq"
    assert select_default_provider(ollama_available=False, gemini_key="secret") == "gemini"


def test_no_ai_service_falls_back_to_keyless_mode() -> None:
    assert select_default_provider(ollama_available=False) == "extractive"
