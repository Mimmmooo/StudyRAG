# Changelog

Project history is organized by development phases. The repository is public and MIT licensed. No cloud deployment is required.

## Unreleased: Phase 8 final polish

- Recent conversational history is included in the RAG prompt for follow-up questions.
- Retrieval includes the previous user question when available, helping with references such as "What does it store?"
- Previous conversation is explicitly not accepted as factual evidence; citations still come from retrieved source passages.
- Chat history is bounded to a small number of recent turns, with message-size limits.
- Refined Streamlit layout, source labels, passage count, indexed document listing, new-chat button and retrieval-count slider.
- Added tests for follow-ups, bounded history, large multi-page documents, multiple file citations and service-to-pipeline wiring.
- Removed an obsolete debugging script and refreshed README/architecture notes.
- Retained optional local Ollama, Gemini/Groq BYOK, Ruff, pytest and GitHub Actions.

## Phases 1–7

- Setup and modular Python package.
- PDF, TXT and MD ingestion with page-aware chunks.
- Local sentence-transformer embeddings and ChromaDB vector retrieval.
- Provider-neutral local Ollama and optional Gemini/Groq adapters.
- Source-grounded RAG answers with filename and page citations.
- Streamlit chat UI with per-session temporary index.
- Automated tests, CI on Python 3.11/3.12, MIT License and documentation.

## Release notes

A first tagged release will be appropriate after the Phase 8 tests pass locally and on GitHub Actions. Until then, the `main` branch is the working open-source source of truth.
