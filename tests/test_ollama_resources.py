"""Verify resource-bounded Ollama generation without running a model."""

from unittest.mock import patch

import pytest

from src.llm.base import LLMError
from src.llm.ollama_provider import OllamaProvider


def test_ollama_disables_thinking_and_limits_context() -> None:
    with patch("src.llm.ollama_provider.post_json", return_value={"response": "বাংলা [1]"}) as post:
        result = OllamaProvider(model="qwen3:4b").generate("A question")
    assert result == "বাংলা [1]"
    payload = post.call_args.args[1]
    assert payload["model"] == "qwen3:4b"
    assert payload["think"] is False
    assert payload["options"]["num_ctx"] == 2048
    assert payload["options"]["num_predict"] == 256
    assert post.call_args.kwargs["timeout"] == 240


def test_ollama_timeout_explains_ram_and_alternatives() -> None:
    with patch(
        "src.llm.ollama_provider.post_json",
        side_effect=LLMError("Model request timed out. Try again."),
    ):
        with pytest.raises(LLMError, match="reduce passages to 2-3"):
            OllamaProvider(model="qwen3:4b").generate("question")
