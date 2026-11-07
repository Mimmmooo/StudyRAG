"""Local CPU-friendly embeddings with optional sentence-transformers."""

from functools import lru_cache

MULTILINGUAL_ONNX = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"


@lru_cache(maxsize=2)
def _fast_model(model_name: str):
    """Load an ONNX embedding model once per process."""
    from fastembed import TextEmbedding

    return TextEmbedding(model_name=model_name)


@lru_cache(maxsize=2)
def _sentence_model(model_name: str):
    """Load the optional PyTorch embedding model once per process."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError(
            "Sentence Transformers requires: pip install -r requirements-local.txt"
        ) from exc
    return SentenceTransformer(model_name)


class LocalEmbedder:
    """Encode documents and queries using the same local model."""

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed study chunks as floating-point vectors."""
        if not texts:
            return []
        if self.model_name.startswith("sentence-transformers/") and self.model_name != MULTILINGUAL_ONNX:
            return _sentence_model(self.model_name).encode(
                texts, show_progress_bar=False
            ).tolist()
        return [vector.tolist() for vector in _fast_model(self.model_name).embed(texts)]

    def embed_query(self, text: str) -> list[float]:
        """Embed questions using the same model as uploaded documents."""
        if self.model_name.startswith("sentence-transformers/") and self.model_name != MULTILINGUAL_ONNX:
            return self.embed_documents([text])[0]
        return next(iter(_fast_model(self.model_name).query_embed(text))).tolist()
