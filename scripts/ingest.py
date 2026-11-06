"""CLI smoke test for the Phase 2 ingestion pipeline."""

import argparse
from pathlib import Path

from src.config import Settings
from src.embeddings.embedder import LocalEmbedder
from src.ingestion.service import ingest_files
from src.vectorstore.chroma_store import ChromaStore


def main() -> None:
    """Index files, optionally run a similarity search, or clear the store."""
    parser = argparse.ArgumentParser(description="StudyRAG Phase 2 document indexing")
    parser.add_argument("files", nargs="*", type=Path, help="PDF, TXT or MD files")
    parser.add_argument("--query", help="Search indexed chunks by meaning")
    parser.add_argument("--clear", action="store_true", help="Clear the knowledge base")
    args = parser.parse_args()

    settings = Settings()
    store = ChromaStore(settings.chroma_dir)
    if args.clear:
        store.clear()
        print("Knowledge base cleared.")
    if args.files:
        embedder = LocalEmbedder(settings.embedding_model)
        counts = ingest_files(args.files, settings, embedder, store)
        for filename, count in counts.items():
            print(f"Indexed {filename}: {count} chunks")
    if args.query:
        embedder = LocalEmbedder(settings.embedding_model)
        results = store.search(embedder.embed_query(args.query))
        if not results:
            print("No indexed notes found.")
        for result in results:
            print(f"\n[{result['source']} page {result['page']}] {result['text']}")
    print(f"Total stored chunks: {store.count()}")


if __name__ == "__main__":
    main()
