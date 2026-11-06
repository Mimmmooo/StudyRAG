"""Test retrieval and source citations from the terminal."""

import argparse

from src.config import Settings
from src.embeddings.embedder import LocalEmbedder
from src.llm.base import BaseLLMProvider, LLMError
from src.llm.factory import create_provider
from src.rag.pipeline import RAGPipeline
from src.vectorstore.chroma_store import ChromaStore


class DemoLLM(BaseLLMProvider):
    """Offline model stand-in for testing the retrieval pipeline."""

    def generate(self, prompt: str) -> str:
        """Show that a prompt reached the generator."""
        return "Demo: retrieved context reached the generator. [1]"


def main() -> None:
    """Run a question against indexed notes."""
    parser = argparse.ArgumentParser(description="StudyRAG RAG smoke test")
    parser.add_argument("--question", default="What does ChromaDB store?")
    parser.add_argument(
        "--provider",
        choices=["demo", "ollama", "gemini", "groq"],
        default="demo",
    )
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--max-distance", type=float, default=2.0)
    args = parser.parse_args()
    settings = Settings()
    store = ChromaStore(settings.chroma_dir)
    if store.count() == 0:
        parser.exit(
            1, "No notes indexed. Run: python -m scripts.ingest examples/sample_notes.txt\n"
        )
    llm = DemoLLM() if args.provider == "demo" else create_provider(args.provider)
    rag = RAGPipeline(
        LocalEmbedder(settings.embedding_model),
        store,
        llm,
        top_k=args.top_k,
        max_distance=args.max_distance,
    )
    try:
        result = rag.ask(args.question)
    except (LLMError, ValueError) as exc:
        parser.exit(1, f"RAG error: {exc}\n")
    print(f"Answer: {result.answer}")
    for index, citation in enumerate(result.citations, start=1):
        print(
            f"[{index}] {citation.source}, page {citation.page}: "
            f"{citation.text[:160]}"
        )


if __name__ == "__main__":
    main()
