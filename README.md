# AskDocs — Minimal RAG Assistant 📄🤖

A tiny Retrieval-Augmented Generation (RAG) assistant: drop your documents into
`./data`, start the server, and ask questions about them in your browser.

Works with either:

- a **local** model via [Ollama](https://ollama.com) (private, no API key), or
- the **OpenAI** API.

## Features

- Ingest `.txt`, `.md`, and `.pdf` documents
- Chunk + embed documents into a lightweight local index (`data/index.json`)
- Retrieve the most relevant passages and answer from them only
- Simple chat web UI with source attribution

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Configure a provider
cp .env.example .env

# 3. Drop your documents into ./data

# 4. Run
python app.py
# open http://localhost:8000
```

Then click **Re-index documents** and start asking questions.

## Providers

### Local with Ollama (default)

```bash
ollama pull nomic-embed-text
ollama pull llama3.2
```

```env
RAG_PROVIDER=ollama
```

### OpenAI

```env
RAG_PROVIDER=openai
OPENAI_API_KEY=sk-...
```

## Project structure

```
ask-docs/
├── app.py            # FastAPI app + RAG engine
├── static/index.html # chat UI
├── data/             # put your documents here
├── requirements.txt
└── .env.example
```

## License

MIT
