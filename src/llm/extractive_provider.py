"""Free, keyless retrieval-only answers for hosted demonstrations."""

import re

from src.llm.base import BaseLLMProvider


class ExtractiveProvider(BaseLLMProvider):
    """Quote the nearest source without sending data to an external LLM."""

    def generate(self, prompt: str) -> str:
        """Return a short source excerpt with its original citation marker."""
        match = re.search(
            r"Study passages:\n\[1\] File: [^\n]+\n(.*?)(?=\n\n\[2\] File:|\n\nQuestion:)",
            prompt,
            flags=re.DOTALL,
        )
        if not match:
            return ""
        passage = " ".join(match.group(1).split())
        if not passage:
            return ""
        return f"Closest passage from your notes: {passage[:600]} [1]"
