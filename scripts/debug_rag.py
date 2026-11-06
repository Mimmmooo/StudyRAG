"""Inspect retrieval distances and raw Ollama output for a local test question."""

from src.config import Settings
from src.embeddings.embedder import LocalEmbedder
from src.llm.factory import create_provider
from src.rag.pipeline import RAGPipeline
from src.vectorstore.chroma_store import ChromaStore


class TracingLLM:
    """Show the model's response without hiding the underlying result."""

    def __init__(self) -> None:
        self.provider = create_provider("ollama")

    def generate(self, prompt: str) -> str:
        """Print the actual model response for diagnosis."""
        result = self.provider.generate(prompt)
        print(f"Raw Ollama response: {result!r}")
        return result


def main() -> None:
    """Display nearest notes, their distances, and the final answer."""
    settings = Settings()
    store = ChromaStore(settings.chroma_dir)
    embedder = LocalEmbedder(settings.embedding_model)
    question = "What does ChromaDB store?"
    print(f"Indexed chunks: {store.count()}")
    matches = store.search(embedder.embed_query(question), limit=3)
    for index, match in enumerate(matches, start=1):
        print(
            f"Match {index}: distance={match['distance']:.4f}, "
            f"source={match['source']}, page={match['page']}"
        )
        print(f"Passage: {match['text'][:400]}")
    if not matches:
        print("No matches: index is empty or unavailable.")
        return
    answer = RAGPipeline(
        embedder, store, TracingLLM(), max_distance=2.0
    ).ask(question)
    print(f"Final answer: {answer.answer}")
    print(f"Citations: {len(answer.citations)}")


if __name__ == "__main__":
    main()
