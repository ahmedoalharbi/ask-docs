"""AskDocs — a minimal RAG (Retrieval-Augmented Generation) assistant.

Drop documents (.txt / .md / .pdf) into ./data, start the server and chat with
them in your browser. Works with a local Ollama model or the OpenAI API.

If no model is reachable, it automatically falls back to keyword-based
retrieval so the system still works out of the box.

Quick start:
    pip install -r requirements.txt
    cp .env.example .env   # then edit the provider/model
    python app.py          # open http://localhost:8000
"""
import hashlib
import html as _html
import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

PROVIDER = os.getenv("RAG_PROVIDER", "ollama").lower()  # "ollama" | "openai"

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
OLLAMA_EMBED_MODEL = os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text")
OLLAMA_CHAT_MODEL = os.getenv("OLLAMA_CHAT_MODEL", "llama3.2")

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
OPENAI_CHAT_MODEL = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")

DATA_DIR = Path(os.getenv("RAG_DATA_DIR", str(BASE_DIR / "data")))
INDEX_FILE = DATA_DIR / "index.json"
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "80"))
TOP_K = int(os.getenv("RAG_TOP_K", "4"))
HASH_DIM = 512


# --------------------------------------------------------------------------- #
# Text chunking
# --------------------------------------------------------------------------- #
def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap
    return chunks


# --------------------------------------------------------------------------- #
# Embeddings + generation (Ollama or OpenAI, with a no-model fallback)
# --------------------------------------------------------------------------- #
def hash_embed(text: str) -> List[float]:
    """Bag-of-words hashing embedding — requires no external model."""
    vec = [0.0] * HASH_DIM
    for word in re.findall(r"\w+", text.lower()):
        h = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16)
        vec[h % HASH_DIM] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    return [x / norm for x in vec] if norm else vec


def _embed(text: str) -> List[float]:
    try:
        if PROVIDER == "openai":
            if not OPENAI_API_KEY:
                raise RuntimeError("OPENAI_API_KEY is not set")
            resp = httpx.post(
                f"{OPENAI_BASE_URL}/embeddings",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": OPENAI_EMBED_MODEL, "input": text},
                timeout=60,
            )
            resp.raise_for_status()
            return resp.json()["data"][0]["embedding"]

        resp = httpx.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={"model": OLLAMA_EMBED_MODEL, "prompt": text},
            timeout=60,
        )
        resp.raise_for_status()
        return resp.json()["embedding"]
    except Exception:
        return hash_embed(text)


def _generate(prompt: str, context: str) -> Optional[str]:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that answers questions ONLY from the "
                "provided context. If the answer is not in the context, say so clearly. "
                "Answer concisely in the same language as the question."
            ),
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion: {prompt}",
        },
    ]
    try:
        if PROVIDER == "openai":
            resp = httpx.post(
                f"{OPENAI_BASE_URL}/chat/completions",
                headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
                json={"model": OPENAI_CHAT_MODEL, "messages": messages},
                timeout=120,
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

        resp = httpx.post(
            f"{OLLAMA_URL}/api/chat",
            json={"model": OLLAMA_CHAT_MODEL, "messages": messages, "stream": False},
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["message"]["content"]
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Documents + index
# --------------------------------------------------------------------------- #
def load_documents(directory: Path) -> List[Dict[str, str]]:
    docs: List[Dict[str, str]] = []
    if not directory.exists():
        return docs
    for path in sorted(directory.iterdir()):
        if path.is_dir():
            continue
        suffix = path.suffix.lower()
        if suffix in {".txt", ".md"}:
            docs.append(
                {"source": path.name, "text": path.read_text(encoding="utf-8", errors="ignore")}
            )
        elif suffix == ".pdf":
            try:
                from pypdf import PdfReader
            except ImportError:
                continue
            reader = PdfReader(str(path))
            text = "\n".join((page.extract_text() or "") for page in reader.pages)
            docs.append({"source": path.name, "text": text})
    return docs


def build_index(directory: Path = DATA_DIR) -> Tuple[int, List[Dict[str, Any]]]:
    directory.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, Any]] = []
    for doc in load_documents(directory):
        for chunk in chunk_text(doc["text"]):
            if not chunk.strip():
                continue
            records.append(
                {
                    "source": doc["source"],
                    "text": chunk,
                    "embedding": _embed(chunk),
                }
            )
    return len(records), records


def load_index() -> List[Dict[str, Any]]:
    if not INDEX_FILE.exists():
        return []
    return json.loads(INDEX_FILE.read_text(encoding="utf-8"))


def save_index(records: List[Dict[str, Any]]) -> None:
    INDEX_FILE.write_text(json.dumps(records, ensure_ascii=False), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Retrieval + RAG
# --------------------------------------------------------------------------- #
def cosine(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


def retrieve(query: str, records: List[Dict[str, Any]], top_k: int = TOP_K) -> List[Dict[str, Any]]:
    qvec = _embed(query)
    scored = [(cosine(qvec, r["embedding"]), r) for r in records]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


STOPWORDS = {
    "ما", "هي", "هو", "في", "من", "عن", "على", "أن", "إلى", "كان", "هذا", "هذه",
    "ذلك", "كيف", "لماذا", "أين", "اين", "ماذا", "الذي", "التي", "الذين", "مع",
    "لا", "لم", "لن", "هل", "او", "أو", "و",
    "what", "is", "are", "the", "a", "an", "of", "in", "on", "for", "to",
    "how", "why", "where", "when", "which", "who", "do", "does", "did",
}


def highlight_terms(text: str, query: str) -> str:
    """HTML-escape text and wrap matched query terms in <mark> tags."""
    escaped = _html.escape(text)
    terms = {t.lower() for t in re.findall(r"\w+", query) if len(t) >= 2}
    terms -= {t for t in terms if t in STOPWORDS}
    for term in sorted(terms, key=len, reverse=True):
        safe = _html.escape(term)
        escaped = re.sub(
            re.escape(safe),
            lambda m: f"<mark>{m.group(0)}</mark>",
            escaped,
            flags=re.IGNORECASE,
        )
    return escaped


def answer(
    question: str, records: List[Dict[str, Any]], top_k: int = TOP_K
) -> Tuple[str, List[str]]:
    if not records:
        return (
            "لا توجد مستندات مفهرسة بعد. أضف ملفات إلى مجلد data ثم اضغط «فهرسة المستندات».",
            [],
        )
    hits = retrieve(question, records, top_k)
    context = "\n\n".join(h["text"] for h in hits)
    generated = _generate(question, context)
    if generated:
        answer_text = _html.escape(generated)
    else:
        parts = [
            f"【{_html.escape(h['source'])}】\n{highlight_terms(h['text'], question)}"
            for h in hits
        ]
        answer_text = (
            "⚠️ لا يوجد نموذج لغوي متاح — هذا وضع الاسترجاع النصي. "
            "المقاطع الأكثر صلة (المطابق مظلّل):\n\n"
            + "\n\n———\n\n".join(parts)
        )
    sources = sorted({h["source"] for h in hits})
    return answer_text, sources


def get_stats(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    sources = sorted({r["source"] for r in records})
    return {"documents": len(sources), "chunks": len(records), "sources": sources}


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app = FastAPI(title="AskDocs — RAG Assistant")

_records: List[Dict[str, Any]] = load_index()


class Query(BaseModel):
    question: str
    top_k: int = TOP_K


@app.get("/")
def index() -> FileResponse:
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.post("/api/chat")
def chat(q: Query):
    answer_text, sources = answer(q.question, _records, q.top_k)
    return {"answer": answer_text, "sources": sources}


@app.post("/api/ingest")
def ingest():
    global _records
    try:
        count, _records = build_index()
        save_index(_records)
        return {"indexed": count, "chunks": len(_records)}
    except Exception as exc:  # noqa: BLE001
        return {"error": str(exc)}


@app.get("/api/stats")
def stats():
    return get_stats(_records)


@app.post("/api/clear")
def clear():
    global _records
    _records = []
    if INDEX_FILE.exists():
        INDEX_FILE.unlink()
    return {"ok": True, "documents": 0, "chunks": 0}


app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
