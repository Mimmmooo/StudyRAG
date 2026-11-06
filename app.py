"""Streamlit entry point; the RAG interface arrives in Phase 5."""

import streamlit as st

from src.config import Settings


def main() -> None:
    """Render a minimal setup confirmation page."""
    st.set_page_config(page_title="StudyRAG", page_icon="📚")
    st.title("📚 StudyRAG")
    st.info("Phase 1 setup complete. Document chat will be added in later phases.")
    settings = Settings()
    st.caption(f"Embedding model configured: {settings.embedding_model}")


if __name__ == "__main__":
    main()
