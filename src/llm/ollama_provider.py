"""Ollama local model implementation."""

import re

from src.llm.base import BaseLLMProvider, LLMError
from src.llm.http import post_json


class OllamaProvider(BaseLLMProvider):
    """Generate answers with an Ollama model running on the user's computer."""

    def __init__(
        self, model: str = "qwen2.5:3b", base_url: str = "http://localhost:11434"
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str) -> str:
        """Request a non-streaming local completion."""
        try:
            data = post_json(
                f"{self.base_url}/api/generate",
                {
                    "model": self.model,
                    "prompt": (
                        ("/no_think\n" if self.model.startswith("qwen3") else "")
                        + "Answer the question directly in its language. Do not write analysis.\n"
                        + prompt
                    ),
                    "stream": False,
                    "think": False,
                    "keep_alive": "5m",
                    "options": {
                        "temperature": 0,
                        "num_ctx": 2048,
                        "num_predict": 256,
                    },
                },
                timeout=240,
            )
        except LLMError as exc:
            if "Cannot connect" in str(exc):
                raise LLMError(
                    "Ollama is not reachable. Start Ollama and pull the selected model."
                ) from exc
            if "timed out" in str(exc).lower():
                raise LLMError(
                    "Local Ollama timed out after 240 seconds. "
                    "On an 8 GB computer, reduce passages to 2-3 or choose "
                    "a smaller model such as qwen2.5:3b."
                ) from exc
            raise
        answer = data.get("response")
        if not isinstance(answer, str):
            raise LLMError("Ollama returned no response. Check the model name.")
        # Some Qwen versions may still emit explicit thinking tags.
        cleaned = re.sub(r"<think>.*?</think>", "", answer, flags=re.DOTALL).strip()
        if "<think>" in cleaned or "</think>" in cleaned:
            return ""
        return cleaned
