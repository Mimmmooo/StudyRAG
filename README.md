# StudyRAG

Free, open-source study assistant for asking questions about your own notes, with source file and page citations.

**Status: Phase 3 complete (LLM provider adapters).** The Streamlit screen remains a setup placeholder; grounded RAG answers and chat UI are not implemented yet.

## Features available now
- Parse text-based PDF, TXT, and MD files
- Keep source filename and PDF page number for each chunk
- Configurable chunk size and overlap
- Generate free local sentence-transformer embeddings
- Persist embeddings in local ChromaDB; similarity search and reset via CLI
- Interchangeable Ollama, Gemini and Groq LLM providers behind a common interface\n- API keys supplied via environment variables or per-call BYOK; never stored in the vector DB\n- Automated unit/integration tests

## Quick start (macOS, Python 3.12)

```bash
git clone https://github.com/Mimmmooo/StudyRAG.git
cd StudyRAG
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m ruff check .
python -m streamlit run app.py
```

If you already have a local checkout, run `git pull origin main` inside it instead of cloning. If you previously extracted a ZIP (not a git clone), download the new ZIP from GitHub or use the Phase 2 ZIP provided in chat.

## Phase 2: test document ingestion

```bash
python -m scripts.ingest examples/sample_notes.txt
python -m scripts.ingest --query "What does ChromaDB store?"
python -m scripts.ingest --clear
```

The first embedding command downloads the free `all-MiniLM-L6-v2` model from Hugging Face (internet required once). After caching, embeddings can run offline. The default local vector database lives in `.chroma/` and is gitignored.

Try your own files:

```bash
python -m scripts.ingest "/path/to/lecture.pdf" "/path/to/notes.md"
```

Scanned/image-only PDFs are not supported yet. TXT and MD files use page 1 as their citation placeholder. Do not index confidential documents on a shared computer. The CLI stores text and vectors locally.

## Phase 3: test the LLM providers

The **offline demo** needs no key, model download, or running Ollama service:

```bash
python -m scripts.llm_smoke
python -m pytest -v
python -m ruff check .
```

To use **Ollama** locally, install and start Ollama separately, then pull the selected model:

```bash
ollama pull qwen2.5:3b
python -m scripts.llm_smoke --provider ollama
```

For **Gemini** or **Groq**, create your own free-tier key. Set it in your shell without committing it:

```bash
export GEMINI_API_KEY="your-key"
python -m scripts.llm_smoke --provider gemini
```

Or use `GROQ_API_KEY` with `--provider groq`. Free-tier limits, model names, and eligibility can change. Never paste keys in chat or commit them. In the later Streamlit UI, BYOK keys will live only in session memory.

The demo validates the common interface; it does **not** make a live cloud or Ollama request. Real provider calls are optional and should be tested separately.

## Architecture

```text
app.py                       Streamlit setup placeholder
src/config.py                Environment settings
src/ingestion/loaders.py     PDF/TXT/MD -> page-aware records
src/ingestion/chunker.py     Page-aware overlapping chunks
src/ingestion/service.py     Indexing orchestration
src/embeddings/embedder.py   Local sentence-transformer embeddings
src/vectorstore/chroma_store.py  Persistent Chroma search\nsrc/llm/base.py             Common abstract provider interface\nsrc/llm/factory.py          Provider selection and BYOK\nsrc/llm/*_provider.py       Ollama, Gemini and Groq adapters\nscripts/llm_smoke.py        Provider smoke test
scripts/ingest.py            Local indexing and search CLI
tests/                       Unit and integration tests
.github/workflows/ci.yml     Ruff + pytest
```

LangChain text splitters are used for chunking, with business logic kept out of the Streamlit UI.

## Roadmap
- [x] Phase 1: setup
- [x] Phase 2: ingestion, embeddings, Chroma
- [x] Phase 3: Ollama/Gemini/Groq/BYOK providers
- [ ] Phase 4: RAG answer generation with citations and not-found handling
- [ ] Phase 5: Streamlit upload/chat UI
- [ ] Phase 6: expanded tests and CI
- [ ] Phase 7: documentation and free deployment
- [ ] Phase 8: stretch goals

## Security and limits
Never commit API keys, `.env`, `.chroma`, or private study notes. API-based providers will use user-owned keys in later phases. PDF extraction requires a text layer; OCR is a future enhancement.

MIT licensed.


## Phase 4: retrieval-augmented answers

Phase 4 connects the local embedding model, ChromaDB search, and the provider-neutral LLM interface. Responses include retrieved file and page citations. When no sufficiently close passage is found, the pipeline returns a no-answer message. This is a retrieval threshold, not a guarantee that generated claims are correct.

First index the example file (if not already indexed), then run the offline smoke test:

```bash
python -m scripts.ingest examples/sample_notes.txt
python -m scripts.rag_smoke
python -m ruff check .
python -m pytest -v
```

The default `demo` provider proves retrieval and citation wiring without generating a factual answer. To generate a real answer with a running local Ollama model, use `python -m scripts.rag_smoke --provider ollama`. Alternatively choose `gemini` or `groq` after setting your own free-tier API key. The chat UI is planned for Phase 5.
