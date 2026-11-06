"""Offline tests for the interchangeable LLM provider implementations."""

from unittest.mock import Mock, patch

import pytest
import requests

from scripts.llm_smoke import DemoProvider
from src.llm.base import BaseLLMProvider, LLMError
from src.llm.factory import create_provider
from src.llm.gemini_provider import GeminiProvider
from src.llm.groq_provider import GroqProvider
from src.llm.http import post_json
from src.llm.ollama_provider import OllamaProvider


def test_common_interface() -> None:
    """Every provider implements the same abstract contract."""
    assert isinstance(DemoProvider(), BaseLLMProvider)
    assert DemoProvider().generate("hi") == "Demo provider received: hi"


def test_ollama_request() -> None:
    """Ollama uses a local non-streaming endpoint."""
    with patch("src.llm.ollama_provider.post_json", return_value={"response": "Hello"}) as send:
        assert OllamaProvider().generate("hi") == "Hello"
    assert send.call_args.args[0] == "http://localhost:11434/api/generate"
    assert send.call_args.args[1]["stream"] is False


def test_gemini_request() -> None:
    """Gemini sends the key as a header, not as a URL parameter."""
    reply = {"candidates": [{"content": {"parts": [{"text": "Answer"}]}}]}
    with patch("src.llm.gemini_provider.post_json", return_value=reply) as send:
        assert GeminiProvider("secret").generate("question") == "Answer"
    assert "secret" not in send.call_args.args[0]
    assert send.call_args.kwargs["headers"]["x-goog-api-key"] == "secret"


def test_groq_request() -> None:
    """Groq uses the chat-completions response shape."""
    reply = {"choices": [{"message": {"content": "Answer"}}]}
    with patch("src.llm.groq_provider.post_json", return_value=reply):
        assert GroqProvider("secret").generate("question") == "Answer"


@pytest.mark.parametrize("provider", ["gemini", "groq"])
def test_missing_key(provider: str) -> None:
    """Cloud providers refuse missing keys."""
    with pytest.raises(LLMError, match="key is required"):
        create_provider(provider, api_key="")


def test_factory_selects_ollama() -> None:
    """Factory can select a fully local model."""
    assert isinstance(create_provider("ollama"), OllamaProvider)


def test_rate_limit_is_friendly() -> None:
    """HTTP 429 returns a useful message without leaking response bodies."""
    response = Mock(status_code=429)
    with (
        patch("src.llm.http.requests.post", return_value=response),
        pytest.raises(LLMError, match="rate limit"),
    ):
        post_json("https://example.com", {})


def test_connection_failure_is_friendly() -> None:
    """Network failures become user-facing errors."""
    with (
        patch("src.llm.http.requests.post", side_effect=requests.ConnectionError),
        pytest.raises(LLMError, match="Cannot connect"),
    ):
        post_json("https://example.com", {})
