# Contributing to StudyRAG

Thanks for helping improve a free study assistant. Keep every feature usable without a paid service.

## Setup

Use Python 3.11 or 3.12:

```bash
git clone https://github.com/Mimmmooo/StudyRAG.git
cd StudyRAG
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
```

## Project conventions

- Keep Streamlit code in `app.py` limited to presentation. Implement logic in `src/`.
- Use small readable functions, type hints and short docstrings.
- Keep provider-specific network code inside `src/llm/`; share the `BaseLLMProvider` contract.
- Preserve source filename and page information during ingestion/retrieval.
- Add offline tests with mocks or temporary folders. Do not require Ollama, Gemini, Groq or actual API keys for CI.
- Run Ruff and pytest before opening a pull request. GitHub Actions checks both Python 3.11 and 3.12.

## Reporting bugs

Include the reproduction steps, operating system, Python version, commands, expected result and actual result. **Do not attach private notes, API keys or full prompts containing personal material.** Use the provided `examples/sample_notes.txt` to produce safe, shareable examples.

## Data and secrets

Never commit `.env`, `.streamlit/secrets.toml`, API keys, indexed Chroma databases, virtual environments or uploaded personal documents. Prefer documented free alternatives. If you add a new provider, explain its free/local mode and the data it transmits.

## Deployment notes

Local Ollama works only where the Ollama server is reachable. A remotely hosted Streamlit app cannot reach a laptop's `localhost`. Hosted builds are experimental and limited by free-tier resources; see [docs/DEPLOYMENT.md](docs/DEPLOYMENT.md).

## License

Contributions are made under the repository's MIT license.
