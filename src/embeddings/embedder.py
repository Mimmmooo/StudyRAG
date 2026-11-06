"""Local, free sentence-transformer embedding adapter."""

from functools import lru_cache

from sentence_transformers import SentenceTransformer


@lru_cache(maxsize=2)
def _model(model_name: str) -> SentenceTransformer:
    """Reuse the model instead of reloading it for every document."""
    return SentenceTransformer(model_name)


class LocalEmbedder:
    """Convert text into vectors using a locally cached model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of chunks."""
        if not texts:
            return []
        return _model(self.model_name).encode(texts, show_progress_bar=False).tolist()

    def embed_query(self, text: str) -> list[float]:
        """Embed a query with the same model used for documents."""
        return self.embed_documents([text])[0]
