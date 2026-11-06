# 📚 StudyRAG

**A free, open-source study assistant for chatting with your own notes, with file and page citations.**

StudyRAG runs on your computer. Upload text-based PDFs, TXT or Markdown files; it finds relevant passages and asks a language model to answer using your notes. Follow-up questions can refer to earlier turns. Source excerpts remain visible under each answer.

**Status:** Open-source portfolio project. No cloud deployment is required. The GitHub repository is public and MIT licensed.

## Features

- 📄 Upload multiple PDF, TXT and Markdown notes (15 MB per file)
- 🧠 Local sentence-transformer embeddings and ChromaDB retrieval
- 💬 Follow-up-aware chat with recent conversational context
- 🔖 Expandable citations with original filename, page number and passage text
- 🎛️ Adjustable number of retrieved passages, 1–5
- 🔌 Free local Ollama; optional Gemini/Groq free-tier API keys (BYOK)
- 🔒 Temporary per-browser-session index; clear notes and chat when finished
- ✅ Ruff, pytest, GitHub Actions on Python 3.11 and 3.12

## Run StudyRAG locally

Requirements: **Python 3.11 or 3.12** and a Mac/Linux computer. You can use the free Ollama app for AI responses, without any API key.

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

### Start a local model (recommended)

Install [Ollama](https://ollama.com/download), then download the free model:

```bash
ollama pull qwen2.5:3b
ollama list
curl http://localhost:11434/api/tags
```

On macOS, Ollama often runs in the background. If `ollama serve` reports that port 11434 is already in use but the `curl` command returns model data, it is already running. Select **ollama** in StudyRAG.

Alternatively select **gemini** or **groq** and enter your own free-tier API key in the password field. These optional providers transmit retrieved note passages to their services.

### First test

1. Upload `examples/sample_notes.txt` in the sidebar.
2. Click **Index selected files**.
3. Select Ollama (or a configured alternative).
4. Ask **What does ChromaDB store?**
5. The answer should mention vector storage and similar-passage search. Expand **[1] sample_notes.txt · page 1** to inspect the evidence.
6. Follow up with **What are they used for?** to try the multi-turn context feature.

Use **New chat** to reset only the conversation, or **Clear notes and chat** to remove the session index too.

### Update an existing checkout

```bash
cd ~/StudyRAG
git pull origin main
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
python -m streamlit run app.py
```

## How it works

```text
Upload files → Extract pages → Split into chunks → Local embeddings → ChromaDB
                                                               ↓
Question + recent turns → Embed query → Retrieve matching passages
                                               ↓
                                 Ollama / Gemini / Groq
                                               ↓
                                 Answer + [1], [2] citations
```

UI logic stays in `app.py`; document ingestion, embeddings, vector search, model adapters, and the RAG pipeline are separate modules under `src/`.

For technical details, see [Architecture](docs/ARCHITECTURE.md).

## Test the code

```bash
python -m ruff check .
python -m pytest -q
python -m scripts.ingest examples/sample_notes.txt
python -m scripts.rag_smoke
```

The default smoke script uses a demo provider and does not generate a factual answer. `python -m scripts.rag_smoke --provider ollama` tests a live Ollama model. GitHub Actions checks Ruff and offline pytest on Python 3.11 and 3.12.

## Limitations and privacy

- Scanned/image-only PDFs need OCR, which is not yet included.
- TXT and MD files use page 1 as a citation placeholder.
- Retrieved citations show candidate evidence; they do not automatically verify every generated claim.
- Retrieval distance uses a demo cutoff and is not calibrated for all documents.
- Follow-up awareness uses a limited number of recent messages, not permanent conversation memory.
- Original uploads are removed after indexing, but extracted passages and embeddings remain in the temporary session index until cleared. The CLI index is stored in a separate local `.chroma/` folder.
- Never commit API keys, `.env` files, private notes or vector databases. For especially sensitive notes, choose local Ollama rather than an external provider.

## Project progress

- [x] Phase 1: Setup
- [x] Phase 2: PDF/TXT/MD ingestion, embeddings, ChromaDB
- [x] Phase 3: Ollama, Gemini, Groq provider adapters
- [x] Phase 4: Grounded RAG with citations
- [x] Phase 5: Streamlit upload and chat
- [x] Phase 6: Automated tests and CI
- [x] Phase 7: README and architecture documentation
- [x] Phase 8: Follow-up-aware chat, UI improvements, regression tests

Cloud deployment is intentionally not part of this project's current goal. If you want to experiment with it later, see the optional [Deployment Guide](docs/DEPLOYMENT.md).

## Contribute and license

Pull requests and bug reports are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) and remove any secrets/private text from logs. The project is licensed under [MIT](LICENSE); you can use, modify, and redistribute it under that license.
