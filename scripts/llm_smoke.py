"""Phase 3 provider smoke test, offline by default."""

import argparse

from src.llm.base import BaseLLMProvider, LLMError
from src.llm.factory import create_provider


class DemoProvider(BaseLLMProvider):
    """No-network demonstration of the common provider interface."""

    def generate(self, prompt: str) -> str:
        """Return a predictable message without any API or model."""
        return f"Demo provider received: {prompt}"


def main() -> None:
    """Test interface offline or opt in to a real model call."""
    parser = argparse.ArgumentParser(description="Test StudyRAG LLM providers")
    parser.add_argument("--provider", choices=["demo", "ollama", "gemini", "groq"], default="demo")
    parser.add_argument("--prompt", default="Explain retrieval-augmented generation in one sentence.")
    parser.add_argument("--model", default=None)
    args = parser.parse_args()
    try:
        provider = (
            DemoProvider()
            if args.provider == "demo"
            else create_provider(args.provider, model=args.model)
        )
        print(provider.generate(args.prompt))
    except LLMError as exc:
        parser.exit(1, f"Provider error: {exc}\n")


if __name__ == "__main__":
    main()
