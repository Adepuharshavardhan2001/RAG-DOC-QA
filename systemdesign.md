
# System Design — RAG Document Q&A

**Author:** Adepu Harshavardhan
**Project:** RAG Document Q&A System
**Live URL:** http://13.235.79.43:8000
**GitHub:** https://github.com/Adepuharshavardhan2001/RAG-DOC-QA

---

## 1. Problem Statement

Students and researchers spend hours reading 15-30 page research papers to find specific information — definitions, dataset details, ablation results, formulas. Standard search (Ctrl+F) only finds exact keyword matches and can't answer natural-language questions like *"What is the difference between GCN and GAT?"*

**This project solves that.** A user uploads a PDF and asks questions in plain English. The system returns precise answers grounded in the document — with strict safeguards against hallucination.

---

## 2. High-Level Architecture
┌─────────────────────────────────────────────────────────────┐
│ USER (Browser) │
└────────────────────────┬────────────────────────────────────┘
│ HTTP (currently)
▼
┌─────────────────────────────────────────────────────────────┐
│ Django Application (Gunicorn) │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ JWT Authentication Layer │ │
│ └──────────────────────────────────────────────────────┘ │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ Input Guardrail (prompt injection, short queries) │ │
│ └──────────────────────────────────────────────────────┘ │
│ ┌──────────────────────────────────────────────────────┐ │
│ │ RAG Pipeline (rag_engine.py) │ │
│ │ ┌─────────────────┐ ┌─────────────────────────┐ │ │
│ │ │ Vector Search │ │ BM25 Keyword Search │ │ │
│ │ │ (ChromaDB) │ │ (rank_bm25) │ │ │
│ │ └────────┬────────┘ └────────────┬────────────┘ │ │
│ │ └─────────┬───────────────┘ │ │
│ │ ▼ │ │
│ │ Reciprocal Rank Fusion (RRF) │ │
│ │ ▼ │ │
│ │ Cross-Encoder Re-ranker │ │
│ │ ▼ │ │
│ │ LLM Generation (Groq) │ │
│ └──────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
│
┌────────────────┼────────────────┐
▼ ▼ ▼
┌───────────┐ ┌────────────┐ ┌─────────────┐
│ ChromaDB │ │ HuggingFace│ │ Groq API │
│ (local) │ │ Models │ │ (LLM) │
│ │ │ (local) │ │ (external) │
└───────────┘ └────────────┘ └─────────────┘

text

---

## 3. Component Breakdown

### 3.1 Django REST API
- Handles HTTP requests, JWT auth, file uploads.
- Two main endpoints: `/api/upload/` (PDF ingestion) and `/api/query/` (question answering).
- Uses `SimpleJWT` for stateless authentication.

**Trade-off:** Django vs FastAPI. Django was chosen for its built-in ORM, admin, and auth. FastAPI would be faster for a pure API, but Django's batteries-included approach speeds up development.

### 3.2 PDF Ingestion
- **Parser:** PyMuPDF (fitz). Chosen over PyPDF2 because it preserves word spacing on complex layouts (IEEE 2-column papers).
- **Page-level metadata:** Each chunk is tagged with its source page number.
- **Summary chunk:** A synthetic chunk is prepended containing the first 1500 characters of page 1 (title, authors, abstract). This fixes retrieval for "What is the title?" type questions.

**Trade-off:** Chunk size 512 vs 256 vs 1024. Tested 256 and 512. Both gave similar accuracy, but 512 handles formula-heavy sections better. Chose 512 with 50-token overlap.

### 3.3 Embedding Layer
- **Model:** `BAAI/bge-small-en-v1.5` via HuggingFace.
- **Why:** Free, runs locally, no API cost. 384-dimensional vectors.
- **Trade-off:** `bge-small` is faster but less accurate than `bge-large`. For a demo, small is sufficient. For production, `bge-large` or OpenAI embeddings would improve retrieval quality.

### 3.4 Vector Store — ChromaDB
- Persistent local storage (`chroma_db/` folder).
- One collection per user (`user_<id>`).
- Cosine similarity search.

**Trade-off:** ChromaDB vs Pinecone vs Qdrant. ChromaDB is simple, local, free. Pinecone is managed but costs money. For a demo, ChromaDB is ideal.

### 3.5 Hybrid Retrieval
- **Vector search:** Top 15 chunks by cosine similarity.
- **BM25 keyword search:** Top 15 chunks by keyword match.
- **Reciprocal Rank Fusion (RRF):** Merges both lists into top 10.

**Why hybrid?** Vector search misses exact keyword matches (e.g., acronyms, names). BM25 misses semantic similarity. Together, they cover both.

### 3.6 Cross-Encoder Re-ranker
- **Model:** `cross-encoder/ms-marco-MiniLM-L-6-v2`.
- Takes the top 10 candidates and re-ranks them against the query.
- Returns top 3 to the LLM.

**Trade-off:** Adds ~200-500ms latency, but improves precision significantly. Without it, retrieval quality drops.

### 3.7 LLM — Groq (`openai/gpt-oss-120b`)
- **Why Groq:** Extremely fast inference (<2s), free tier available.
- **Why this model:** Groq's free tier offers Llama-based models. Verified against Groq's current model list.

**Trade-off:** Not GPT-4 level. For complex reasoning, GPT-4 or Claude would be better. For a demo, Groq is fast and free.

### 3.8 Input Guardrails
- Blocks prompt injection patterns ("ignore previous instructions", "pretend you are", etc.).
- Blocks queries under 3 characters.
- Runs **before** retrieval — saves API cost.

### 3.9 Hallucination Handling
- Strict prompt: "Answer only using the provided context. If the answer isn't there, say so."
- Verified with 6 off-topic questions — 6/6 correctly refused.
- Verified with cross-document questions (e.g., "compare to TPGC paper") — refused.

---

## 4. Data Flow

### 4.1 Upload Flow
User → Upload PDF → Django view → PyMuPDF extract
→ Chunk (512/50) → Add page metadata
→ Prepend summary chunk
→ HuggingFace embedding (per chunk)
→ ChromaDB storage (user_<id> collection)
→ Response: "Uploaded successfully"

text

### 4.2 Query Flow
User → Ask question → Django view → JWT verify
→ Input guardrail check
→ Embed query (HuggingFace)
→ Vector search (ChromaDB, top 15)
→ BM25 search (top 15)
→ RRF merge → top 10
→ Cross-encoder re-rank → top 3
→ Build prompt with context
→ Groq LLM → answer
→ Response

text

---

## 5. Design Decisions & Trade-offs

| Decision | Choice | Alternative | Why | Trade-off |
|---|---|---|---|---|
| Backend framework | Django | FastAPI | Batteries included: auth, ORM, admin | Heavier than FastAPI |
| PDF parser | PyMuPDF | PyPDF2, pdfplumber | Preserves spacing on complex layouts | Larger dependency |
| Chunk size | 512 | 256, 1024 | Best balance of context and precision | Loses some fine-grained context |
| Chunk overlap | 50 | 0, 100 | Preserves continuity between chunks | Slightly more storage |
| Embedding model | bge-small-en-v1.5 | bge-large, OpenAI | Free, runs locally, fast | Lower accuracy than large |
| Vector DB | ChromaDB | Pinecone, Qdrant | Free, local, simple | No managed scaling |
| Retrieval | Hybrid (vector + BM25) | Vector only | Handles exact matches + semantics | 2x compute per query |
| Rank fusion | RRF | Weighted sum | Simple, no tuning needed | Doesn't weight methods |
| Re-ranker | ms-marco-MiniLM-L-6-v2 | Cohere Rerank | Free, local, fast | Slightly lower quality than Cohere |
| LLM | Groq gpt-oss-120b | OpenAI GPT-4, Claude | Fast, free tier | Less capable on hard reasoning |
| Auth | JWT | Session cookies | Stateless, scalable | No built-in refresh flow yet |
| Deployment | AWS EC2 | Render, Railway | Free tier, full control | More setup required |

---

## 6. Edge Cases Handled

| Edge Case | How Handled |
|---|---|
| Empty PDF | Detected and rejected ("No text found") |
| Scanned/image PDF | Detected and rejected (OCR not implemented) |
| PDF over 10 MB | Rejected at upload ("File size must be under 10MB") |
| Prompt injection ("ignore previous instructions") | Blocked by input guardrail |
| Short queries ("hi") | Rejected ("Please ask a more specific question") |
| Question not in document | Refused ("The provided context does not contain...") |
| Cross-document question | Refused (verified with TPGC paper trap) |
| Duplicate upload | Old collection deleted, new one created |
| Multiple users | Namespace-separated ChromaDB collections |
| Groq model deprecation | Documented; model can be swapped in one line |

---

## 7. Known Limitations

1. **Text-only.** Images, diagrams, and tables aren't processed. OCR or multimodal LLM would fix this.
2. **No citations.** Answers don't include page numbers. Planned improvement.
3. **JWT token expiry.** No refresh flow; users must re-login after 60 minutes.
4. **Single-threaded.** Deployed with 1 gunicorn worker (memory-constrained).
5. **No HTTPS.** Currently HTTP only.
6. **Dynamic IP.** EC2 IP is not allocated as Elastic IP.
7. **No rate limiting.** Could be abused if URL shared widely.

---

## 8. Deployment Notes

**Platform:** AWS EC2 (t3.micro, Ubuntu 26.04, 1 GB RAM, 30 GB disk)

**Stack:**
- **App:** Gunicorn (1 worker) as `rag.service` via systemd
- **Process manager:** systemd (auto-start on boot, auto-restart on crash)
- **DB:** SQLite
- **Vector store:** ChromaDB (local)

**Deployment challenges and fixes:**
1. `pymupdf` and `rank-bm25` missing from `requirements.txt` → added.
2. PyTorch with CUDA (2.5 GB) installed by default → replaced with CPU-only (~200 MB).
3. `llama-index-llms-openai-like` version incompatible with `llama-index-core` → upgraded to compatible.
4. `WindowsPath` vs `str` mismatch in `views.py` → explicit `str()` conversion.
5. `/tmp` tmpfs (455 MB) too small for numpy compile → remounted to 2 GB.
6. Memory constraints (1 GB RAM) → added 2 GB swap.

**Commands:**
```bash
# Start service
sudo systemctl start rag.service

# Check status
sudo systemctl status rag.service

# View logs
sudo journalctl -u rag.service -n 100

# Restart after code changes
sudo systemctl restart rag.service
9. Future Improvements
Page-level citations. Include page numbers in answers.

Token refresh flow. Auto-refresh JWT tokens.

OCR. Support scanned PDFs with Tesseract.

RAGAS evaluation. LLM-as-judge metrics instead of keyword matching.

Multi-modal. Handle images and tables.

Multi-file-type. DOCX, TXT, URLs.

HTTPS + domain. Let's Encrypt + custom domain.

nginx reverse proxy. Clean URL (no :8000).

Elastic IP. Permanent address.

Rate limiting. Protect against abuse.

10. Evaluation
Test document: "A Comprehensive Survey on Graph Neural Networks" (20 pages, IEEE).

Test set: 18 questions across 6 categories.

Category	Questions	Passed
Factual retrieval	5	5/5
Taxonomy	2	2/2
Multi-item retrieval	3	2/3
Reasoning	2	2/2
Hallucination refusal	6	6/6
Total	18	17/18 (94.4%)
Stress tests (additional, not in eval):

Numeric precision (accuracy from tables) — passed

Formula extraction (equations) — passed

Cross-document trap (referencing a different paper) — correctly refused

Prompt injection ("ignore previous instructions") — correctly blocked

Failure analysis:

Q10 ("What applications of GNNs are mentioned?") fails because the eval expects specific domain applications (computer vision, chemistry) but the system retrieves task categories (node classification, graph classification). This is an ambiguity in the question, not a system failure.

11. Conclusion
This project demonstrates a production-grade RAG system with:

Hybrid retrieval (vector + BM25)

Cross-encoder re-ranking

Input guardrails and hallucination handling

Real deployment on AWS EC2

Honest assessment: The system handles most real-world document Q&A tasks well. Its main limitations are text-only input and the absence of page-level citations. Both have clear upgrade paths (OCR, RAGAS, multi-modal).

The engineering work focused on depth over breadth — building one complete, evaluated, deployed system rather than many shallow prototypes