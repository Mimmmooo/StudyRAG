# StudyRAG

Free, open-source study assistant for asking questions about your notes with document and page citations.

**Current status: Phase 1 scaffold.** Document upload, retrieval and chat will be implemented in later phases.

## Planned features
- PDF, TXT and MD ingestion
- Configurable chunks, local sentence-transformer embeddings and ChromaDB
- Ollama locally or bring your own Gemini/Groq free-tier API key
- Streamlit chat with page-aware citations
- Tests and GitHub Actions CI

## Setup (Python 3.11 or 3.12 recommended)

```bash
git clone https://github.com/Mimmmooo/StudyRAG.git
cd StudyRAG
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
cp .env.example .env
python -m streamlit run app.py
```

## Checks

```bash
python -m ruff check .
python -m pytest -q
```

## Architecture

```text
app.py
src/config.py
src/ingestion/
src/embeddings/
src/vectorstore/
src/llm/
src/rag/
tests/
.github/workflows/ci.yml
```

LangChain's focused packages provide text splitting and interfaces, while the project keeps its own readable orchestration.

## Roadmap
- [x] Phase 1: setup and configuration
- [ ] Phase 2: ingestion, chunking, embeddings, ChromaDB
- [ ] Phase 3: LLM providers
- [ ] Phase 4: RAG with citations
- [ ] Phase 5: Streamlit chat UI
- [ ] Phase 6: expanded tests and CI
- [ ] Phase 7: documentation and free deployment
- [ ] Phase 8: stretch goals

## Security
Never commit keys, .env files, or uploaded documents. Free-tier quotas and model availability may change.

## License
MIT
