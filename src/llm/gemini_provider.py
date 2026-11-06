"""Gemini API provider using the public generateContent REST endpoint."""

from urllib.parse import quote

from src.llm.base import BaseLLMProvider, LLMError
from src.llm.http import post_json


class GeminiProvider(BaseLLMProvider):
    """Generate answers with a user-provided Gemini API key."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash") -> None:
        if not api_key.strip():
            raise LLMError("Gemini API key is required.")
        self._api_key = api_key
        self.model = model

    def generate(self, prompt: str) -> str:
        """Call Gemini without writing or logging the key."""
        url = (
            "https://generativelanguage.googleapis.com/v1beta/models/"
            f"{quote(self.model, safe='')}:generateContent"
        )
        data = post_json(
            url,
            {"contents": [{"parts": [{"text": prompt}]}]},
            headers={"x-goog-api-key": self._api_key},
        )
        try:
            parts = data["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError, AttributeError) as exc:
            raise LLMError("Gemini returned no text (possibly blocked or quota-limited).") from exc
        if not text.strip():
            raise LLMError("Gemini returned an empty response.")
        return text.strip()
