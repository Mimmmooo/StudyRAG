# Local installation and running

StudyRAG is a **locally hosted Streamlit web interface**. There is no Streamlit Community Cloud deployment, Codespaces requirement, or GitHub Actions inference backend. Only GitHub is used to store and test code.

## Ubuntu / Linux

Run the same command every time:

```bash
bash -lc 'if [ ! -d "$HOME/StudyRAG/.git" ]; then git clone -b fix/cross-platform-reliability https://github.com/Mimmmooo/StudyRAG.git "$HOME/StudyRAG"; fi; cd "$HOME/StudyRAG" && git pull --ff-only && bash scripts/start.sh'
```

The launcher skips Python packages and Ollama model downloads that are already present. It will install Ubuntu OCR dependencies with apt only if missing and requests a sudo password as needed. If system Python is not 3.11/3.12, it creates a managed 3.12 environment via uv. The Python dependencies live in the project's persistent `.venv/`, not globally. The Ollama binary and model are available system-wide through their normal installation/cache directories.

## macOS

Clone repository and run `bash scripts/start.sh`. For re-runs, `cd ~/StudyRAG && git pull --ff-only && bash scripts/start.sh`. Homebrew is required for first-time automatic OCR installation.

## Windows

From a PowerShell terminal, clone the repository and run `powershell -ExecutionPolicy Bypass -File scripts\start.ps1`. The launcher uses winget for missing Tesseract/Ollama, installs Bengali/English OCR data in your user profile, and reuses `.venv/` and model cache.

## Troubleshooting

- **Ollama is not reachable:** Check that the startup script finished successfully; the model is pulled only once. Inspect `ollama list` and `http://127.0.0.1:11434/api/tags`.
- **Python 3.14 on Ubuntu:** It is not a supported StudyRAG interpreter. The launcher uses managed Python 3.12 rather than uninstalling system Python.
- **Disk quota exceeded:** You need disk space for model weights and dependencies. The default FastEmbed ONNX embeddings avoid the many-gigabyte PyTorch/CUDA dependency chain.
- **Image-based Bengali PDF:** Tesseract must have `ben` and `eng` language data. The launcher installs them once.
- **200 MB PDF fails:** Large or scanned files may require substantial free RAM and processing time; test smaller pages or a text-based PDF before diagnosing.
- **RAG cites the wrong page:** OCR is approximate; original PDF pages are always preserved in metadata. The multilingual semantic search and exact Bengali title search help but don't ensure perfect ranking.

## Security and privacy

With Ollama, indexing and inference happen on your computer. The Streamlit server binds only to `127.0.0.1`. Do not forward it to the public internet without adding authentication and user isolation. Optional Groq and Gemini require BYOK and send retrieved text to those external providers. No API keys are committed or logged.
