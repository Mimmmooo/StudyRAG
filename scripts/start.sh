#!/usr/bin/env bash
# Idempotent local launcher for Ubuntu/Linux and macOS.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
MODEL="${OLLAMA_MODEL:-qwen2.5:3b}"

info() { printf '\n[StudyRAG] %s\n' "$*"; }
fail() { printf '[StudyRAG] ERROR: %s\n' "$*" >&2; exit 1; }

if [ "$(uname -s)" = Linux ]; then
  if ! command -v tesseract >/dev/null 2>&1 ||
     ! tesseract --list-langs 2>&1 | grep -qx ben ||
     ! tesseract --list-langs 2>&1 | grep -qx eng; then
    command -v apt-get >/dev/null 2>&1 ||
      fail "Install Tesseract Bengali/English language packs for your Linux distribution."
    info "Installing missing OCR tools (one-time; may ask for sudo password)"
    sudo apt-get update
    sudo apt-get install -y tesseract-ocr tesseract-ocr-ben tesseract-ocr-eng curl
  fi
elif [ "$(uname -s)" = Darwin ]; then
  if ! command -v tesseract >/dev/null 2>&1 ||
     ! tesseract --list-langs 2>&1 | grep -qx ben ||
     ! tesseract --list-langs 2>&1 | grep -qx eng; then
    command -v brew >/dev/null 2>&1 ||
      fail "Install Homebrew (brew.sh) once to auto-install Bengali OCR on macOS."
    info "Installing missing Tesseract OCR language packs"
    brew install tesseract tesseract-lang
  fi
else
  fail "This script supports Linux/macOS. On Windows run scripts/start.ps1."
fi

valid_python() {
  "$1" -c 'import sys; assert sys.version_info[:2] in ((3, 11), (3, 12))' >/dev/null 2>&1
}
PYTHON=""
for candidate in python3.12 python3.11; do
  if command -v "$candidate" >/dev/null 2>&1 && valid_python "$candidate"; then
    PYTHON="$(command -v "$candidate")"
    break
  fi
done
if ! command -v curl >/dev/null 2>&1; then
  fail "curl is required. Install it once (Ubuntu: sudo apt install curl)."
fi
if [ -z "$PYTHON" ]; then
  info "Python 3.11/3.12 not found; installing managed Python 3.12 (one-time)"
  if ! command -v uv >/dev/null 2>&1; then
    command -v curl >/dev/null 2>&1 || fail "curl is required to install Python."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  fi
  uv python install 3.12
  PYTHON="$(uv python find 3.12)"
fi

if [ -f .venv/pyvenv.cfg ] &&
   ! .venv/bin/python -c 'import sys; assert sys.version_info[:2] in ((3, 11), (3, 12))' >/dev/null 2>&1; then
  info "Replacing incompatible StudyRAG virtual environment"
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  info "Creating isolated Python environment (one-time)"
  if "$PYTHON" -m venv .venv; then :; else
    if ! command -v uv >/dev/null 2>&1; then
      curl -LsSf https://astral.sh/uv/install.sh | sh
      export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
    fi
    uv venv --python "$PYTHON" --seed .venv
  fi
fi

# Check a requirements fingerprint, not the app's source files.
REQUIREMENTS_SHA="$(cat requirements.txt | cksum | awk '{print $1 ":" $2}')"
if [ ! -f .venv/.studyrag-deps ] ||
   [ "$(cat .venv/.studyrag-deps)" != "$REQUIREMENTS_SHA" ] ||
   ! .venv/bin/python -c 'import streamlit, fitz, fastembed, chromadb' >/dev/null 2>&1; then
  info "Installing or repairing Python dependencies (only when required)"
  .venv/bin/python -m pip --version >/dev/null 2>&1 ||
    "$PYTHON" -m ensurepip --upgrade >/dev/null 2>&1 ||
    fail "pip is unavailable. Install python3-venv or Python with ensurepip."
  .venv/bin/python -m pip install -r requirements.txt
  printf '%s\n' "$REQUIREMENTS_SHA" > .venv/.studyrag-deps
else
  info "Python dependencies already installed, skipping pip."
fi

if ! command -v ollama >/dev/null 2>&1; then
  info "Installing Ollama (one-time)"
  if [ "$(uname -s)" = Linux ]; then
    curl -fsSL https://ollama.com/install.sh | sh
  else
    command -v brew >/dev/null 2>&1 ||
      fail "Install Ollama from ollama.com/download (or install Homebrew)."
    brew install --cask ollama
  fi
fi

if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
  info "Starting local Ollama server"
  if [ "$(uname -s)" = Linux ] && command -v systemctl >/dev/null 2>&1; then
    sudo systemctl start ollama 2>/dev/null || true
  fi
  if ! curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1; then
    nohup ollama serve > "$ROOT/.venv/ollama.log" 2>&1 < /dev/null &
  fi
  for attempt in $(seq 1 30); do
    curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null 2>&1 && break
    sleep 1
  done
fi
curl -fsS --max-time 2 http://127.0.0.1:11434/api/tags >/dev/null ||
  fail "Ollama did not start. Check .venv/ollama.log or systemctl status ollama."

if ! ollama list | awk 'NR > 1 { print $1 }' | grep -Fxq "$MODEL"; then
  info "Downloading Ollama model $MODEL (one-time)"
  ollama pull "$MODEL"
else
  info "Ollama model $MODEL is already installed."
fi
info "Launching StudyRAG at http://localhost:8501"
exec .venv/bin/python -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501
