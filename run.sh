#!/usr/bin/env bash
# AskDocs — one-click launcher
set -e
cd "$(dirname "$0")"

if [ ! -d ".venv" ]; then
  echo "Creating virtual environment and installing dependencies..."
  python3 -m venv .venv
  .venv/bin/pip install -r requirements.txt
fi

echo "Starting AskDocs at http://localhost:8000 ..."
exec .venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
