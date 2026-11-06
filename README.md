# 📚 StudyRAG

**Free, open-source chat with your own study notes, with filename + page citations.**

Upload text-based PDF, TXT or Markdown files. StudyRAG splits and embeds them locally, retrieves relevant passages with ChromaDB, then asks a selected LLM to answer using those passages. The chat UI shows the answer and expandable source excerpts.

**Project status:** Phase 7 documentation and deployment preparation. Local end-to-end Ollama + Streamlit smoke test passed; 45 automated tests passed locally in Phase 6, with successful GitHub Actions on Python 3.11 and 3.12. **Public cloud deployment is experimental and not yet verified.**

## Features

- File upload for PDF (text layer), TXT and Markdown
- Page-aware chunking, local embeddings, and ChromaDB similarity search
- Ollama for completely local generation, or optional Gemini/Groq free-tier BYOK
- File/page citations and expandable retrieved passages for each answer
- Per-browser-session temporary document index and chat display history
- Friendly error messages and no-answer diagnostics
- Offline tests, Ruff linting, GitHub Actions on Python 3.11 and 3.12
- MIT license

## Quick start: local (recommended)

**Prerequisites:** Python 3.11/3.12, macOS/Linux, and optionally [Ollama](https://ollama.com/download) for local generation. The first embedding run downloads a free sentence-transformer model.

```bash
git clone https://github.com/Mimmmooo/StudyRAG.git
cd StudyRAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
python -m streamlit run app.py
```

Open **http://localhost:8501**.

For local Ollama, install it and run:

```bash
ollama pull qwen2.5:3b
ollama list
curl http://localhost:11434/api/tags
```

Start Ollama if it is not running. On macOS the Ollama app may already run in the background; an `address already in use` error from `ollama serve` usually means the server is already listening.

Alternatively, choose **Gemini** or **Groq** in the StudyRAG sidebar and enter your own optional free-tier key. Never commit or paste keys in a public issue. Check each provider's current limits and terms.

### First browser test

1. Upload `examples/sample_notes.txt`.
2. Click **Index selected files**.
3. Choose an available model provider.
4. Ask `What does ChromaDB store?`.
5. Check that the answer mentions vector storage/similar-passage search and has **Source: sample_notes.txt, page 1**.

This was verified locally using Ollama `qwen2.5:3b`. Live Gemini/Groq and public hosting have not been end-to-end verified.

### Update and retest an existing checkout

```bash
cd ~/StudyRAG
git pull origin main
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
python -m streamlit run app.py
```

## Free deployment

**Local:** works offline after the model downloads, requires no paid services or API key.

**Streamlit Community Cloud:** a free GitHub-based hosting option, but not yet validated for StudyRAG's PyTorch/sentence-transformers/ChromaDB memory needs. Hosted `localhost:11434` is **not** your laptop's Ollama. Use the user-provided Gemini/Groq option when testing the hosted app. Avoid confidential notes on public/shared infrastructure until its security limitations are addressed.

**Read the full [deployment guide](docs/DEPLOYMENT.md)** for setup, cloud steps, privacy warnings and troubleshooting. Official Streamlit docs: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy.

## Architecture

```text
Upload -> loaders (PDF/TXT/MD + page)
       -> page-aware chunks
       -> local sentence-transformer embeddings
       -> ChromaDB (text + filename + page)

Question -> embed -> retrieve -> filter by distance
         -> grounded context prompt -> LLM provider
         -> answer + source citations -> Streamlit chat
```

The presentation layer (`app.py`) is separate from ingestion, embeddings, vector storage, provider adapters and the RAG pipeline. See the [architecture guide](docs/ARCHITECTURE.md) for the module-by-module design.

## Command-line checks

```bash
python -m scripts.ingest examples/sample_notes.txt
python -m scripts.rag_smoke
python -m scripts.rag_smoke --provider ollama
python -m scripts.debug_rag
python -m ruff check .
python -m pytest -q
```

The default `rag_smoke` uses a demo model response, not factual generation. `--provider ollama` requires a running Ollama server and downloaded model. CLI ingestion stores the vector index in the gitignored `.chroma/` directory; the web app uses a **different, temporary per-browser-session** index.

## Limitations and security

- Scanned/image-only PDFs require OCR (not implemented). TXT/MD citations use page 1.
- A 15 MB per-file limit is checked in the application. Original uploaded files are removed after ingestion, but extracted text and vectors stay in the session's temporary index while active.
- Public hosting receives uploaded notes. Using Gemini/Groq sends selected passages to those providers. Do not upload confidential files without reviewing their terms and privacy policies.
- Retrieved citations are candidate evidence, not automated proof of every claim; model answers can still be incorrect. The retrieval distance cutoff is currently a demo default, not a calibrated relevance score.
- Chat history is displayed in the browser session but not sent to the LLM as multi-turn context.
- The app does not yet implement user authentication, permanent account storage, or independently audited multi-tenant isolation. Not production-ready for unrestricted public uploads.
- Never commit `.env`, API keys, private notes or local Chroma data. `.gitignore` excludes common secret and local-index paths.

## Roadmap

- [x] Phase 1: Project setup
- [x] Phase 2: Ingestion, embeddings and ChromaDB
- [x] Phase 3: Ollama, Gemini and Groq provider interface
- [x] Phase 4: Retrieval, grounded generation and citations
- [x] Phase 5: Streamlit upload and chat UI
- [x] Phase 6: Regression tests and CI
- [x] Phase 7: Documentation and deployment preparation
- [ ] Phase 8: Stretch goals

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setting up checks and contributing without exposing credentials. Feedback and reproducible bug reports are welcome; remove all API keys and private document content before sharing logs.

Licensed under [MIT](LICENSE).
