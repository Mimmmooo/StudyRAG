# StudyRAG deployment guide

StudyRAG is free and open source. There are two supported *paths to try*: running on your own computer, or experimenting with Streamlit Community Cloud. The cloud option is not yet validated for StudyRAG's heavy ML dependencies, so a successful build and stable performance are **not guaranteed**.

## Option A: Local development (recommended and tested)

### Requirements

- Python 3.11 or 3.12
- Free disk space for Python packages, the embedding model and optional Ollama model
- Ollama installed locally, or a user-owned Gemini/Groq free-tier API key

On macOS or Linux (Ubuntu/Debian):

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

For Ollama, install it from https://ollama.com/download and run:

```bash
ollama pull qwen2.5:3b
ollama list
curl http://localhost:11434/api/tags
```

Start the Ollama application; run `ollama serve` only if it is not already running. If it says `address already in use` and `curl` responds, the server is already running. Select **ollama** in StudyRAG.

Alternatively, select **gemini** or **groq** in the sidebar and enter your own API key in the password field. Free-tier availability, limits and model names can change. Never paste keys into issue reports or commit them.

Open http://localhost:8501, upload `examples/sample_notes.txt`, click **Index selected files**, and ask:

> What does ChromaDB store?

Expected content: ChromaDB stores vectors and lets us search for similar passages, with source `sample_notes.txt, page 1`.

### Update an existing local checkout

```bash
cd ~/StudyRAG
git pull origin main
source .venv/bin/activate
python -m pip install -r requirements-dev.txt
python -m ruff check .
python -m pytest -q
python -m streamlit run app.py
```

### Windows local setup (PowerShell)

Install Git and Python 3.11 or 3.12, then use the following without activating PowerShell scripts:

```powershell
git clone https://github.com/Mimmmooo/StudyRAG.git
cd StudyRAG
py -3.12 -m venv .venv
.\\.venv\\Scripts\\python.exe -m pip install -r requirements-dev.txt
.\\.venv\\Scripts\\python.exe -m ruff check .
.\\.venv\\Scripts\\python.exe -m pytest -q
.\\.venv\\Scripts\\python.exe -m streamlit run app.py
```

If you have Python 3.11, change the first `py -3.12` command to `py -3.11`. Install Ollama for Windows for fully local inference, or select Gemini/Groq with your own free-tier key.

## Option B: Streamlit Community Cloud (experimental)

Streamlit Community Cloud offers a free GitHub-connected hosting path: https://share.streamlit.io/. **StudyRAG has not been deployed or load-tested there yet.** Its dependencies include PyTorch, sentence-transformers and ChromaDB, so cold starts, dependency installation and RAM usage may exceed free-tier allowances. Do not consider the deployment complete until upload, embeddings, retrieval and model answers all work on the hosted URL.

1. Visit https://share.streamlit.io/ and sign in with GitHub.
2. Choose **Create app**, link the public repository `Mimmmooo/StudyRAG`, branch `main`, entrypoint `app.py`.
3. Select **Python 3.12** in advanced settings when available (CI covers 3.11 and 3.12).
4. Deploy. The existing root `requirements.txt` is the install manifest. No custom `packages.txt` is currently needed.
5. For generated answers, choose **gemini** or **groq** and provide a free-tier key through the sidebar password field. Do not publish shared API keys.
6. Test with `examples/sample_notes.txt`, ask `What does ChromaDB store?`, and inspect the `sample_notes.txt, page 1` citation.

**Ollama on a laptop is NOT accessible from the hosted app at `localhost:11434`.** That URL refers to the cloud container, not your MacBook. The currently implemented Ollama option is for local use. Do not expose your laptop's Ollama port to the public internet as a workaround.

### Hosting, security and data limitations

- Uploading notes to a hosted app sends their contents to that hosting service. Do not use personal, confidential, copyrighted-with-restrictions or institution-protected notes without checking permissions and privacy needs.
- Each browser session uses a temporary local ChromaDB index on the server. Originals are removed after indexing; vectors and extracted passages persist while that session is alive, and local temporary storage is not durable across restarts. This is **not a formal guarantee of immediate secure deletion**.
- Sessions are isolated in the application state but this project has no authentication, persistent user accounts, quotas, malware scanning or audited multi-tenant security. **Do not represent it as production-ready for arbitrary public uploads.**
- The current 15 MB limit is per file in application validation; Streamlit's own uploader may allow selecting larger files before the app rejects them.
- Requests to Gemini/Groq transmit selected note passages and prompts to third-party providers. Their retention policies, quotas and rates are outside StudyRAG's control.
- The free cloud may sleep or restart, and model download/build time may be substantial. Local hosting remains the reliable no-cost path.
- Up to six recent conversation turns are sent as reference context for follow-up questions. For Gemini/Groq, this text is transmitted to the external provider alongside retrieved passages.

### Diagnosing a deployment

- **Build fails / out of memory:** review Community Cloud build logs. Large ML dependencies may exceed the free hosting budget; run locally instead.
- **Ollama not reachable:** expected on Community Cloud; select Gemini or Groq with your own key, or run locally.
- **No answer found:** test the exact sample question and inspect the no-answer diagnostics expander; document a reproducible test without sharing credentials or private notes.
- **PDF has no extractable text:** scanned PDFs need OCR, which this project does not yet support.
- **App loses indexed notes:** temporary per-session storage is not persistent; re-upload after a session/server reset.
- **Missing API key:** use the in-app password field, not GitHub source files or public issue comments.

### Official references

- Streamlit Community Cloud overview: https://docs.streamlit.io/deploy/streamlit-community-cloud
- Deploy an app: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy
- Dependencies: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies
- Resource limits: https://docs.streamlit.io/deploy/streamlit-community-cloud/manage-your-app
- Secrets: https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management
