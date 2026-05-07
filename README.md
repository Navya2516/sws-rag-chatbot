# SWS AI RAG Chatbot

A production-ready RAG (Retrieval Augmented Generation) chatbot that lets employees ask natural-language questions about SWS AI company policies and receive accurate, grounded answers — with source citations.

---

## Architecture

```
PDF Documents (10 files)
        │
        ▼
pdfplumber — Text Extraction
        │
        ▼
RecursiveCharacterTextSplitter — Chunking (chunk_size=500, overlap=50)
        │
        ▼
sentence-transformers/all-MiniLM-L6-v2 — Embeddings (local, free, 384-dim)
        │
        ▼
ChromaDB (local persistent) — Vector Storage
        │
     User Question
        │
        ▼
Embed Question → Retrieve Top-4 Chunks (cosine similarity)
        │
        ▼
Anthropic Claude (claude-3-haiku) + Context Window
        │
        ▼
Grounded Answer + Source Documents
```

### Design Decisions

| Component | Choice | Reason |
|---|---|---|
| **Embedding Model** | `all-MiniLM-L6-v2` | Local, free, fast, good quality for document retrieval. No API cost per query. |
| **Vector DB** | ChromaDB (local persistent) | Zero setup — runs in-process, persists to disk. Perfect for this use case. |
| **Chunk Size** | 500 tokens, 50 overlap | Balances context richness vs. retrieval precision. Overlap prevents splitting mid-sentence. |
| **Retrieval k** | Top-4 chunks | Enough context for multi-part answers without exceeding Claude's context budget. |
| **LLM** | `claude-3-haiku-20240307` | Fast (< 2s), cheap ($0.00025/1K input), excellent instruction-following for RAG prompts. Swap to `claude-3-5-sonnet` for higher quality. |
| **Framework** | FastAPI | Async, fast, automatic OpenAPI docs at `/docs`. |

---

## Setup

### Prerequisites
- Python 3.9+
- Node.js not required (frontend is plain HTML)
- An Anthropic API key → https://console.anthropic.com

### 1. Clone / unzip the project

```
sws-rag-chatbot/
├── backend/
│   ├── main.py
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   └── index.html
├── documents/
│   ├── SWS-AI-hr-policy.pdf
│   ├── SWS-AI-leave-policy.pdf
│   └── ... (all 10 PDFs)
└── README.md
```

### 2. Set up the backend

```bash
cd backend

# Create and activate virtual environment
python -m venv venv

# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Configure environment

```bash
# Copy the example env file
cp .env.example .env

# Edit .env and add your Anthropic API key:
# ANTHROPIC_API_KEY=sk-ant-...
```

### 4. Ensure documents are in place

Make sure all 10 PDF files are in the `documents/` folder (already there if you downloaded the project as-is):

```
documents/
  SWS-AI-hr-policy.pdf
  SWS-AI-leave-policy.pdf
  SWS-AI-resignation-policy.pdf
  SWS-AI-it-security-policy.pdf
  SWS-AI-code-of-conduct.pdf
  SWS-AI-wfh-policy.pdf
  SWS-AI-performance-review.pdf
  SWS-AI-benefits-compensation.pdf
  SWS-AI-onboarding-guide.pdf
  SWS-AI-company-overview.pdf
```

### 5. Start the backend

```bash
cd backend
python main.py
```

Or with uvicorn directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

On first start, documents are **automatically ingested** — you'll see:
```
INFO: Auto-ingesting documents from ../documents...
INFO: Ingesting: SWS-AI-hr-policy.pdf
INFO:   ✓ HR Policy: 12 chunks
...
INFO: Startup ingestion: 10 new docs, 128 new chunks
```

### 6. Open the frontend

Open `frontend/index.html` directly in your browser:
- **Windows:** Double-click `frontend/index.html` or drag it to Chrome
- **macOS:** `open frontend/index.html`
- Or serve it: `cd frontend && python -m http.server 3000` then go to `http://localhost:3000`

---

## API Reference

Auto-generated docs at: **http://localhost:8000/docs**

| Method | Endpoint | Description |
|---|---|---|
| GET | `/api/status` | Check indexed docs and chunk count |
| POST | `/api/chat` | Main RAG endpoint — ask a question |
| POST | `/api/ingest` | Re-ingest documents from the `documents/` folder |
| POST | `/api/upload` | Upload PDF files via multipart form |
| DELETE | `/api/reset` | Clear the vector store |

### POST /api/chat

**Request:**
```json
{
  "question": "How many days of sick leave do I get?",
  "conversation_history": []
}
```

**Response:**
```json
{
  "answer": "Employees at SWS AI are entitled to 10 days of sick leave per year...",
  "sources": ["Leave Policy"],
  "chunks_used": 4
}
```

---

## Sample Queries

Test your chatbot with these:

- "What is the annual leave policy at SWS AI?"
- "How many days of sick leave do employees get?"
- "What is the notice period for resignation?"
- "What tools does SWS AI use for communication?"
- "What is the password policy for company systems?"
- "How are performance reviews conducted?"
- "What are the WFH guidelines?"
- "Does SWS AI offer health insurance?"
- "What happens if I fail a PIP?"
- "Can I carry forward unused casual leave?"

---

## Upgrading the LLM

In `backend/main.py`, change `LLM_MODEL`:

```python
LLM_MODEL = "claude-3-5-sonnet-20241022"  # Higher quality, slower, more expensive
LLM_MODEL = "claude-3-haiku-20240307"     # Default — fast and cheap
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| `ANTHROPIC_API_KEY not configured` | Create `backend/.env` and add your key |
| `No documents indexed` | Ensure PDFs are in `documents/` and restart backend |
| `Backend offline` in UI | Make sure `python main.py` is running on port 8000 |
| Slow first startup | `sentence-transformers` downloads the model on first run (~90MB) |
| Port conflict | Change port: `uvicorn main:app --port 8001` and update `API` in `frontend/index.html` |
