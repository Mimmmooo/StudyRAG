"""Groq chat-completions provider."""

from src.llm.base import BaseLLMProvider, LLMError
from src.llm.http import post_json


class GroqProvider(BaseLLMProvider):
    """Generate answers using a user's Groq API key."""

    def __init__(
        self, api_key: str, model: str = "llama-3.1-8b-instant"
    ) -> None:
        if not api_key.strip():
            raise LLMError("Groq API key is required.")
        self._api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        """Request one chat completion."""
        data = post_json(
            "https://api.groq.com/openai/v1/chat/completions",
            {"model": self.model, "messages": [{"role": "user", "content": prompt}]},
            headers={"Authorization": f"Bearer {self._api_key}"},
        )
        try:
            answer = data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError("Groq returned no text. Check model availability.") from exc
        if not isinstance(answer, str):
            raise LLMError("Groq returned an unexpected response.")
        return answer.strip()
