#!/usr/bin/env bash
# AskDocs + Ollama — one-click launcher
set -e
cd "$(dirname "$0")"

OLLAMA="$HOME/.local/bin/ollama"

# Start Ollama in the background if it's not already running
if ! curl -s --max-time 2 http://localhost:11434/api/tags >/dev/null 2>&1; then
  echo "Starting Ollama..."
  "$OLLAMA" serve >/tmp/ollama.log 2>&1 &
  sleep 2
fi

# Create virtualenv and install deps on first run
if [ ! -d ".venv" ]; then
  echo "Creating virtual environment and installing dependencies..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

echo "Starting AskDocs at http://localhost:8000 ..."
exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
