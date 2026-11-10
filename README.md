# 📚 StudyRAG

**Free, open-source, local-first RAG study assistant.** Upload PDF/TXT/Markdown notes and receive AI-written answers from your documents. The default AI uses **Ollama on your computer**, with **no API key** and no cloud hosting. Optional Gemini/Groq BYOK integrations remain available.

![StudyRAG interface preview](docs/assets/studyrag-ui-preview.svg)

## What works

- PDF/TXT/MD upload (200 MiB per file; practical limit depends on your computer).
- Bengali and English **scanned PDF OCR** using Tesseract, with a local OCR cache to speed repeat indexing. Image-dominant pages are OCR'd even when a selectable page number or watermark is present.
- Multilingual ONNX embeddings via FastEmbed and a temporary ChromaDB index per session.
- Bengali keyword/title retrieval alongside vector similarity, without citation markers in chat.
- Local Ollama-generated answers and follow-up questions with a simple chat interface.
- Optional Gemini/Groq providers, keyless source-excerpt mode, and session reset.
- Python 3.11/3.12, Streamlit, pytest, Ruff and CI on Ubuntu, macOS and Windows.

## One command to run (Ubuntu / Linux)

You need **Git and an internet connection the first time**. On Ubuntu, install Git once if it is missing: `sudo apt install git`.

The **same command** works for first launch and every later launch. It updates the current feature branch (fast-forward only), then checks for and installs **only missing dependencies**:

```bash
bash -lc 'if [ ! -d "$HOME/StudyRAG/.git" ]; then git clone -b fix/cross-platform-reliability https://github.com/Mimmmooo/StudyRAG.git "$HOME/StudyRAG"; fi; cd "$HOME/StudyRAG" && git pull --ff-only && bash scripts/start.sh'
```

A browser opens at **http://localhost:8501** (or open it manually). Stop Streamlit using `Ctrl+C`. Next time run the **same command**: it skips installed packages and the already-downloaded model.

The first run automatically checks/sets up:
1. Bengali and English Tesseract OCR (Ubuntu uses apt and may ask for your sudo password).
2. Python **3.11 or 3.12**; if Ubuntu defaults to Python 3.14, it downloads its own managed Python 3.12 using uv instead of modifying system Python.
3. A project-specific `.venv`. This is intentionally **not a global pip install**, to avoid breaking other Python projects. It persists across launches.
4. Python packages *only* when `requirements.txt` changes or a required package is missing.
5. The official Ollama app and **`qwen2.5:3b`** (recommended for 8 GB RAM) *only* when missing. Ollama models stay in Ollama's normal model cache. The Ollama server starts automatically if needed.

**Requirements:** internet for the initial downloads; available disk space for OCR, Python packages and a ~2 GB LLM model; adequate RAM. GPU is optional, CPU inference may be slower. No paid service is required.

## macOS

With Git and [Homebrew](https://brew.sh/) available, use the same repository setup then start the app:

```bash
git clone -b fix/cross-platform-reliability https://github.com/Mimmmooo/StudyRAG.git ~/StudyRAG
cd ~/StudyRAG && bash scripts/start.sh
```

Later runs: `cd ~/StudyRAG && git pull --ff-only && bash scripts/start.sh`. The launcher uses Homebrew for OCR and Ollama (only when absent). It reuses the Python environment and downloaded model.

## Windows (PowerShell)

Install Git and Windows App Installer (`winget`) if not already present. Then:

```powershell
git clone -b fix/cross-platform-reliability https://github.com/Mimmmooo/StudyRAG.git "$HOME\StudyRAG"
Set-Location "$HOME\StudyRAG"
powershell -ExecutionPolicy Bypass -File scripts\start.ps1
```

Later runs: `Set-Location "$HOME\StudyRAG"; git pull --ff-only; powershell -ExecutionPolicy Bypass -File scripts\start.ps1`.

The Windows launcher checks for Python, packages, Ollama, and Bengali/English OCR language data. Missing binaries can require a one-time Windows elevation/installer prompt. The user-level trained-data cache is not repeatedly downloaded.

## Test with a real Bengali book

1. Run the launcher. It ensures Ollama is reachable and `qwen2.5:3b` has been pulled.
2. The sidebar should default to **Ollama (local AI)**.
3. Upload a scanned Bengali PDF, including a textbook such as `Secondary (BV)-2026_Class 8_Anondopath_compressed.pdf`, then click **Index selected files**.
4. Check the indexed chunk count, then ask **কাকতাড়ুয়া গল্পের লেখক কে?**
5. The expected author is **সত্যজিৎ রায়**. Check generated answers against your original book, especially when OCR quality is poor.

The 36.9 MB Bengali scanned book has been indexed successfully (167 chunks). This demonstrates one real-book success, **not** a guarantee that all 200 MB PDFs will index or that every OCR word will be correct. The current local bootstrap still requires a first-run smoke test on each OS.

## How the components work

```text
Upload PDF / TXT / MD
     |     (PyMuPDF + Bengali/English Tesseract OCR for scans)
     v
Page-aware chunks -> multilingual ONNX embeddings -> ChromaDB
                                                    |
Question -> vector + Bengali keyword retrieval ------+
                         |
                 Grounded context
                         v
              Local Ollama (default)
                         v
          AI-written answer (no citation markers)
```

The Streamlit UI (`app.py`) contains no ingestion or RAG business logic. The modules are separated under `src/ingestion`, `src/embeddings`, `src/vectorstore`, `src/llm`, and `src/rag`.

## Why Qwen2.5 3B?

Qwen3 4B repeatedly timed out or produced analysis instead of final Bengali answers on an 8 GB Ubuntu machine. The default `qwen2.5:3b` is a smaller non-thinking baseline; it is more practical for local CPU use, although Bengali answer accuracy is not guaranteed. Avoid forcing `OLLAMA_MODEL=qwen3:4b` unless you deliberately want to compare models. OCR uses one CPU thread per Tesseract invocation and a modest DPI reduction; first-time processing still depends on page count and hardware. Repeat uploads reuse locally cached OCR results when the file has not changed. The image-dominant scan detection update uses a versioned OCR cache, so existing PDFs may need one reprocessing pass after updating.

## Other answer providers

**Ollama is the local default, with no API key.** Gemini/Groq free-tier BYOK are optional. Never commit keys or notes to GitHub. When you select an external provider, retrieved passages and recent conversation context are sent to that provider. The **Source passages (no AI)** mode performs citation-based retrieval without a text generation model.

To switch the locally downloaded Ollama model, set `OLLAMA_MODEL` before running the launcher. The model is pulled only when missing. `requirements-local.txt` is optional for experimenting with PyTorch-backed sentence-transformers models; the default multilingual ONNX model needs no PyTorch/CUDA download.

## Testing and development

Inside the repository, after the first launcher setup:

```bash
.venv/bin/python -m ruff check .
.venv/bin/python -m pytest -q
.venv/bin/python -m scripts.ingest examples/sample_notes.txt
```

On Windows use `.venv\Scripts\python.exe` instead. Ruff/pytest are developer tools: install once with `.venv/bin/python -m pip install -r requirements-dev.txt` (or the Windows equivalent) before running these checks. CI runs offline tests on three operating systems.

## Notes and limitations

- Scanned OCR is best-effort and can misread characters, names, punctuation or columns. Image-heavy pages with only a page number or short overlay are now checked for OCR; tiny inset images on text pages do not trigger it. The app no longer displays citation markers or verifies each factual claim: check important answers against the original PDF.
- 200 MiB is the upload ceiling, not a universal indexing performance guarantee. Processing speed depends on scan DPI, page count, RAM and CPU.
- The local Streamlit server binds to `127.0.0.1`; it is **not** a public website. Data stays on your computer when using local Ollama.
- PDF page numbers refer to physical pages of the uploaded PDF, which can differ from printed page numbers.
- Index data is temporary for the browser session; original upload bytes are removed after indexing. OCR results are cached in `~/.cache/studyrag/ocr/` (or `STUDYRAG_OCR_CACHE_DIR`) to speed repeated uploads. Remove that folder if you want to erase cached extracted text. CLI ingestion uses the local `.chroma/` folder.
- Automated CI and a manual local PDF/Ollama smoke test should pass before this branch is merged. Do not claim production-grade verification from an offline test alone.

## Project

This is a student open-source portfolio project under the [MIT License](LICENSE). Technical structure: [Architecture](docs/ARCHITECTURE.md). Contributions: [CONTRIBUTING](CONTRIBUTING.md).

## OCR engine evaluation (2026)

[Baidu Unlimited-OCR-2](https://huggingface.co/baidu/Unlimited-OCR-2) was evaluated as a potential replacement. Its model is MIT licensed and targets long document parsing, but the documented inference path is GPU-heavy (tested on NVIDIA A800 80 GB with CUDA). StudyRAG keeps Tesseract as the default free CPU OCR for 8 GB local machines. We adopted the practical lesson of checking for image-dominant scanned pages even when a sparse text layer exists, rather than adding an untested large model download. See [OCR review](docs/OCR_REVIEW.md).
