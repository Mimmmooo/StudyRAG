"""Ollama local model implementation."""

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
                {"model": self.model, "prompt": prompt, "stream": False, "options": {"temperature": 0}},
                timeout=180,
            )
        except LLMError as exc:
            if "Cannot connect" in str(exc):
                raise LLMError(
                    "Ollama is not reachable. Start Ollama and pull the selected model."
                ) from exc
            raise
        answer = data.get("response")
        if not isinstance(answer, str):
            raise LLMError("Ollama returned no response. Check the model name.")
        return answer.strip()
