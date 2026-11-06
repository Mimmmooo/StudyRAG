# StudyRAG architecture

The pipeline is divided into small Python modules. The Streamlit page only renders inputs and outputs; parsing, embedding, storage and generation live in `src/`.

## Data flow

```text
Browser upload (PDF / TXT / MD)
             |
             v
src/ui_service.py           StudySession, upload validation, temporary storage
             |
             v
src/ingestion/loaders.py    Parse file into text pages + filename + page number
             |
             v
src/ingestion/chunker.py    Split each page into overlapping chunks
             |
             v
src/embeddings/embedder.py  sentence-transformers local embeddings
             |
             v
src/vectorstore/chroma_store.py    Chroma collection + page metadata


Browser question + up to six previous turns\n       |
       v
src/ui_service.py
       |
       v
src/rag/pipeline.py
       |  embed current question with recent user context -> Chroma similarity search -> filter distance
       |  prepare source-labelled context -> provider.generate(prompt)
       v
src/llm/factory.py -> Ollama / Gemini / Groq (src/llm/*_provider.py)
       |
       v
src/rag/models.py           RAGAnswer(answer, tuple[Citation, ...])
       |
       v
app.py                      Chat answer + expandable file/page passages
```

## Module responsibilities

| Module | Responsibility |
| --- | --- |
| `app.py` | Streamlit widgets and session-state chat display |
| `src/ui_service.py` | Session-based document ingestion and RAG orchestration |
| `src/ingestion/` | Load text-based documents, retain page metadata, chunk text |
| `src/embeddings/` | Local sentence-transformer adapter |
| `src/vectorstore/` | ChromaDB persistence and similarity queries |
| `src/llm/` | Provider-neutral interface and Ollama/Gemini/Groq implementations |
| `src/rag/` | Retrieval thresholds, grounding prompts, citations, answers |
| `scripts/` | CLI smoke tests and local diagnostics |
| `tests/` | Offline unit/integration/regression checks |

## Citation behavior

PDF page numbers are 1-based, derived from the text layer. TXT/MD are treated as one page. Each chunk keeps `source`, `page` and `chunk_index` metadata. Retrieval returns the source/page alongside its passage; the final answer exposes those retrieved source locations. These citations are **candidate evidence**; the application does not yet automatically verify every claim against the cited passage.

## Model separation

`BaseLLMProvider.generate(prompt)` defines the shared provider interface. The factory selects local Ollama or optional Gemini/Groq APIs. The RAG pipeline does not need to know which provider was selected. Recent chat turns (up to six, with each text clipped to 500 characters) help resolve follow-up references, but only newly retrieved file passages count as factual evidence. API keys are supplied by the user/environment and should never be committed.

## Storage and trust boundaries

- The CLI uses the configured `.chroma/` directory locally (gitignored).
- The Streamlit UI uses a temporary per-session directory, including a Chroma index. Original uploads are removed after ingestion; extracted content is still held in the vector index while active.
- A public hosted deployment transfers documents to the server, and third-party providers receive retrieved passages. No authentication or full production isolation guarantees are implemented.
- Document text is untrusted input and is passed to the LLM as reference context. The system prompt requests grounded answers, but prompt injection and unsupported model claims are not eliminated.

## Testing

`python -m ruff check .` checks lint; `python -m pytest -q` runs offline tests. GitHub Actions runs both on Python 3.11 and 3.12. The local Ollama live test is a separate manual check. No hosted deployment is required for this portfolio project.
