"""
SWS AI RAG Chatbot — FastAPI Backend
=====================================
Architecture:
  - PDF ingestion: pdfplumber
  - Chunking: LangChain RecursiveCharacterTextSplitter (chunk_size=500, overlap=50)
  - Embeddings: sentence-transformers (all-MiniLM-L6-v2, local, free)
  - Vector DB: ChromaDB (local, persistent)
  - LLM: Anthropic Claude (claude-3-haiku-20240307 — fast & cost-effective)
  - API: FastAPI with CORS for the frontend
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Optional

import pdfplumber
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv

# ─── Config ──────────────────────────────────────────────────────────────────
load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
DOCUMENTS_PATH = Path(os.getenv("DOCUMENTS_PATH", "../documents"))
CHROMA_DB_PATH = os.getenv("CHROMA_DB_PATH", "./chroma_db")
COLLECTION_NAME = "sws_ai_documents"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"   # Fast, local, 384-dim
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 4                               # Retrieve top-4 chunks per query
LLM_MODEL = "claude-3-haiku-20240307"  # Fast + cheap; swap to claude-3-5-sonnet for higher quality

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

# ─── Initialize models (loaded once at startup) ───────────────────────────────
log.info("Loading embedding model...")
embedder = SentenceTransformer(EMBEDDING_MODEL)
log.info("Embedding model loaded.")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

client = OpenAI(
    api_key=GROQ_API_KEY,
    base_url="https://api.groq.com/openai/v1"
)

# ChromaDB — persistent local vector store
chroma_client = chromadb.PersistentClient(
    path=CHROMA_DB_PATH,
    settings=Settings(anonymized_telemetry=False)
)
collection = chroma_client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"}
)

# Text splitter
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# ─── FastAPI app ──────────────────────────────────────────────────────────────
app = FastAPI(title="SWS AI RAG Chatbot", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # Restrict in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
FRONTEND_PATH = Path(__file__).parent.parent / "frontend"
if FRONTEND_PATH.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_PATH)), name="static")


# ─── Pydantic models ──────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    conversation_history: Optional[List[dict]] = []

class ChatResponse(BaseModel):
    answer: str
    sources: List[str]
    chunks_used: int

class IngestResponse(BaseModel):
    message: str
    documents_processed: int
    total_chunks: int

class StatusResponse(BaseModel):
    indexed_documents: List[str]
    total_chunks: int
    ready: bool


# ─── PDF Ingestion ────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_path: Path) -> str:
    """Extract all text from a PDF using pdfplumber."""
    text_parts = []
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(f"[Page {page_num}]\n{page_text.strip()}")
    except Exception as e:
        log.error(f"Error extracting {pdf_path.name}: {e}")
        return ""
    return "\n\n".join(text_parts)


def ingest_documents(documents_dir: Path) -> dict:
    """
    Load PDFs → extract text → chunk → embed → store in ChromaDB.
    Skips documents already in the collection (idempotent).
    """
    pdf_files = list(documents_dir.glob("*.pdf"))
    if not pdf_files:
        return {"documents_processed": 0, "total_chunks": 0}

    # Get already-indexed doc names
    existing = set()
    try:
        all_meta = collection.get(include=["metadatas"])
        for m in all_meta["metadatas"]:
            existing.add(m.get("source", ""))
    except Exception:
        pass

    docs_processed = 0
    total_chunks = 0

    for pdf_path in pdf_files:
        doc_name = pdf_path.stem  # e.g. "SWS-AI-leave-policy"
        friendly_name = get_friendly_name(pdf_path.name)

        if friendly_name in existing:
            log.info(f"Skipping (already indexed): {friendly_name}")
            continue

        log.info(f"Ingesting: {pdf_path.name}")
        raw_text = extract_text_from_pdf(pdf_path)
        if not raw_text.strip():
            log.warning(f"No text extracted from {pdf_path.name}")
            continue

        # Chunk
        chunks = splitter.split_text(raw_text)
        if not chunks:
            continue

        # Embed
        embeddings = embedder.encode(chunks, show_progress_bar=False).tolist()

        # Store in ChromaDB
        ids = [f"{doc_name}_chunk_{i}" for i in range(len(chunks))]
        metadatas = [
            {
                "source": friendly_name,
                "filename": pdf_path.name,
                "chunk_index": i,
                "total_chunks": len(chunks),
            }
            for i in range(len(chunks))
        ]

        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=chunks,
            metadatas=metadatas,
        )

        docs_processed += 1
        total_chunks += len(chunks)
        log.info(f"  ✓ {friendly_name}: {len(chunks)} chunks")

    return {"documents_processed": docs_processed, "total_chunks": total_chunks}


def get_friendly_name(filename: str) -> str:
    """Convert filename to human-readable doc name."""
    name_map = {
        "SWS-AI-hr-policy.pdf": "HR Policy",
        "SWS-AI-leave-policy.pdf": "Leave Policy",
        "SWS-AI-resignation-policy.pdf": "Resignation & Exit Policy",
        "SWS-AI-it-security-policy.pdf": "IT & Security Policy",
        "SWS-AI-code-of-conduct.pdf": "Code of Conduct",
        "SWS-AI-wfh-policy.pdf": "Work From Home Policy",
        "SWS-AI-performance-review.pdf": "Performance Review Policy",
        "SWS-AI-benefits-compensation.pdf": "Benefits & Compensation",
        "SWS-AI-onboarding-guide.pdf": "Employee Onboarding Guide",
        "SWS-AI-company-overview.pdf": "Company Overview & Mission",
    }
    return name_map.get(filename, filename.replace(".pdf", "").replace("-", " ").title())


# ─── RAG Query ────────────────────────────────────────────────────────────────
def retrieve_chunks(question: str, k: int = TOP_K) -> List[dict]:
    """Embed question → retrieve top-k chunks from ChromaDB."""
    q_embedding = embedder.encode([question], show_progress_bar=False).tolist()[0]
    results = collection.query(
        query_embeddings=[q_embedding],
        n_results=min(k, collection.count() or 1),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    if results["documents"] and results["documents"][0]:
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chunks.append({
                "text": doc,
                "source": meta.get("source", "Unknown"),
                "relevance_score": round(1 - dist, 3),
            })
    return chunks


def build_system_prompt(chunks: List[dict]) -> str:
    """Build the RAG system prompt with retrieved context."""
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(f"[Source: {chunk['source']}]\n{chunk['text']}")

    context = "\n\n---\n\n".join(context_parts)

    return f"""You are the SWS AI internal knowledge assistant. Your job is to answer employee questions accurately and helpfully using ONLY the company document context provided below.

STRICT RULES:
1. Answer ONLY from the provided context. Do not use any outside knowledge.
2. If the answer is not in the context, say exactly: "I don't have that information in the company documents."
3. Be concise and direct. Use bullet points for lists.
4. Always be professional and helpful in tone.
5. Do not mention "the context" or "the documents provided" — just answer naturally as if you know the company policies.
6. If multiple documents are relevant, synthesize the information clearly.

COMPANY DOCUMENT CONTEXT:
{context}"""


def generate_answer(question: str, chunks: List[dict], history: List[dict]) -> str:
    system_prompt = build_system_prompt(chunks)

    messages = [
        {
            "role": "system",
            "content": system_prompt
        }
    ]

    for turn in history[-6:]:
        messages.append({
            "role": turn["role"],
            "content": turn["content"]
        })

    messages.append({
        "role": "user",
        "content": question
    })

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        temperature=0.3
    )

    return response.choices[0].message.content


# ─── API Routes ───────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Auto-ingest documents from documents folder on startup."""
    if DOCUMENTS_PATH.exists():
        log.info(f"Auto-ingesting documents from {DOCUMENTS_PATH}...")
        result = ingest_documents(DOCUMENTS_PATH)
        log.info(f"Startup ingestion: {result['documents_processed']} new docs, {result['total_chunks']} new chunks")
    else:
        log.warning(f"Documents folder not found: {DOCUMENTS_PATH}")


@app.get("/")
async def serve_frontend():
    """Serve the frontend HTML."""
    index_path = FRONTEND_PATH / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "SWS AI RAG Chatbot API. Frontend not found — serve frontend/index.html separately."}


@app.get("/api/status", response_model=StatusResponse)
async def get_status():
    """Check which documents are indexed and how many chunks exist."""
    try:
        count = collection.count()
        if count == 0:
            return StatusResponse(indexed_documents=[], total_chunks=0, ready=False)

        all_meta = collection.get(include=["metadatas"])
        doc_names = sorted(set(m.get("source", "Unknown") for m in all_meta["metadatas"]))
        return StatusResponse(
            indexed_documents=doc_names,
            total_chunks=count,
            ready=count > 0,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/ingest", response_model=IngestResponse)
async def ingest_endpoint():
    """Manually trigger document ingestion from the documents folder."""
    if not DOCUMENTS_PATH.exists():
        raise HTTPException(status_code=404, detail=f"Documents folder not found: {DOCUMENTS_PATH}")
    try:
        result = ingest_documents(DOCUMENTS_PATH)
        total = collection.count()
        return IngestResponse(
            message="Ingestion complete.",
            documents_processed=result["documents_processed"],
            total_chunks=total,
        )
    except Exception as e:
        log.error(f"Ingestion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
    """Upload PDF files directly via the API and ingest them."""
    import tempfile, shutil

    if not GROQ_API_KEY:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY not set.")

    upload_dir = DOCUMENTS_PATH
    upload_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for file in files:
        if not file.filename.endswith(".pdf"):
            continue
        dest = upload_dir / file.filename
        with open(dest, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)

    if not saved:
        raise HTTPException(status_code=400, detail="No valid PDF files provided.")

    result = ingest_documents(upload_dir)
    return {
        "uploaded": saved,
        "documents_processed": result["documents_processed"],
        "total_chunks": result["total_chunks"],
    }


@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Main RAG chat endpoint.
    1. Embed the question
    2. Retrieve top-K relevant chunks from ChromaDB
    3. Pass chunks + question to Claude as context
    4. Return answer + source document names
    """
    if not GROQ_API_KEY:
        raise HTTPException(status_code=400, detail="GROQ_API_KEY not configured. Set it in backend/.env")

    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    if collection.count() == 0:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed yet. Add PDFs to the /documents folder and restart, or call POST /api/ingest."
        )

    try:
        # Step 1: Retrieve relevant chunks
        chunks = retrieve_chunks(request.question)
        if not chunks:
            return ChatResponse(
                answer="I couldn't find relevant information in the company documents.",
                sources=[],
                chunks_used=0,
            )

        # Step 2: Generate answer
        answer = generate_answer(
            question=request.question,
            chunks=chunks,
            history=request.conversation_history or [],
        )

        # Step 3: Deduplicate sources
        sources = list(dict.fromkeys(c["source"] for c in chunks))

        return ChatResponse(answer=answer, sources=sources, chunks_used=len(chunks))

    except Exception as e:
        log.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")


@app.delete("/api/reset")
async def reset_collection():
    """Reset the ChromaDB collection (re-ingestion needed after this)."""
    global collection
    chroma_client.delete_collection(COLLECTION_NAME)
    collection = chroma_client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}
    )
    return {"message": "Collection reset. Call POST /api/ingest to re-index."}


# ─── Run ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
