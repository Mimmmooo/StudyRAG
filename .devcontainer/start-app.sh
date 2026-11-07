#!/usr/bin/env bash
# Start the Streamlit preview inside GitHub Codespaces, not on the user's PC.
set -euo pipefail
if curl --silent --fail http://127.0.0.1:8501/_stcore/health >/dev/null; then
  echo "StudyRAG is already running at port 8501."
  exit 0
fi
nohup python -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501 --server.headless true > /tmp/studyrag-streamlit.log 2>&1 < /dev/null &
echo "StudyRAG started on port 8501. View /tmp/studyrag-streamlit.log for logs."
