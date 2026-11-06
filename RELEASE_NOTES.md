# StudyRAG v1.0.0

First public open-source portfolio release of StudyRAG. **Run locally, no deployment required.**

## Highlights

- Upload text-based PDF, TXT, or Markdown notes.
- Page-aware text chunking with local sentence-transformer embeddings and ChromaDB similarity search.
- Answer questions with the free local Ollama model, or optional Gemini/Groq free-tier bring-your-own-key providers.
- Follow-up-aware chat that uses bounded recent conversation while grounding answers in retrieved passages.
- Show numbered citations with source filename, page number, and expandable passages.
- Streamlit interface with retrieval count, indexed-file overview, and New chat / Clear notes controls.
- Ruff and 53 offline tests verified on Python 3.11 and 3.12 through GitHub Actions.
- README, architecture, contributor notes, MIT license, and interface preview.

## Quick start

```bash
git clone https://github.com/Mimmmooo/StudyRAG.git
cd StudyRAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m streamlit run app.py
```

For fully local responses, install [Ollama](https://ollama.com/download), run `ollama pull qwen2.5:3b`, and select `ollama` in the app.

## Known limitations

- Image-only PDFs need OCR.
- The retrieval-distance cutoff is not calibrated for every type of notes.
- Cited passages are supporting evidence candidates, not automated fact verification.
- Temporary per-browser-session index; no long-term document storage or account system.
- No cloud deployment was created for this release.

MIT licensed. See the [README](https://github.com/Mimmmooo/StudyRAG#readme) for setup and usage.
