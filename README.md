# RAG Document Q&A System

A production-grade Retrieval-Augmented Generation (RAG) system that lets users upload PDF documents and ask natural-language questions about their content. Built with Django, LlamaIndex, ChromaDB, HuggingFace embeddings, and Groq.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Django](https://img.shields.io/badge/Django-4.2-green)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-purple)

---

##  Live Demo

**URL:** http://13.235.79.43:8000

**Login credentials:**
- **Username:** `user`
- **Password:** `user@1234`

**What to try:** Upload any PDF (research paper, contract, manual) and ask questions about it.

> **Note:** Deployed on AWS EC2 (t3.micro, 1 GB RAM). Cold starts may take a few seconds. JWT tokens expire after 60 minutes — log in again if you get an error.

![Live Demo](docs/screenshots/live-demo.png)

---

##  Features

### Core RAG Pipeline
- **PDF Ingestion** — Upload PDFs via a web interface. Text is extracted with PyMuPDF, chunked with page-level metadata, and embedded.
- **Hybrid Retrieval** — Combines vector search (HuggingFace embeddings) with BM25 keyword search using Reciprocal Rank Fusion (RRF).
- **Cross-Encoder Re-ranking** — Re-ranks the top candidates with `ms-marco-MiniLM-L-6-v2` before sending to the LLM.
- **Summary Chunk Augmentation** — Automatically generates a metadata-rich summary chunk during ingestion to fix retrieval of title/authors/abstract questions.

### Safety & Guardrails
- **Input Guardrails** — Blocks prompt injection attacks ("ignore previous instructions...") before they reach the LLM.
- **Hallucination Handling** — Strict grounding: the LLM answers only from retrieved context. If the answer isn't there, it says so.
- **User Isolation** — Each user has a private ChromaDB collection. Users can only query their own documents.

### Application
- **JWT Authentication** — Secure login with access/refresh tokens.
- **Clean UI** — Simple login, upload, and query interface (HTML/CSS/JS).
- **Local Embeddings** — Uses `BAAI/bge-small-en-v1.5` from HuggingFace (no API cost).

---

##  Evaluation Results

The system is evaluated on an **18-question test set** covering six categories, using the paper *"A Comprehensive Survey on Graph Neural Networks"* (20 pages, IEEE) as the test document.

| Category | Questions | Passed |
|---|---|---|
| Factual retrieval (title, authors, acronyms) | 5 | 5/5  |
| Taxonomy (GNN categories, GAE definition) | 2 | 2/2  |
| Multi-item retrieval (datasets, directions, applications) | 3 | 2/3  |
| Reasoning (GNN vs network embedding, spectral vs spatial) | 2 | 2/2  |
| Hallucination refusal (off-topic questions) | 6 | 6/6  |
| **Total** | **18** | **17/18 (94.4%)** |

**How to reproduce:**
```bash
python run_eval.py

Deployment

The app is deployed on AWS EC2 (t3.micro, Ubuntu 26.04).

Stack:

App server: Gunicorn (1 worker)

Process manager: systemd (rag.service)

Reverse proxy: nginx (planned)

Database: SQLite

Vector store: ChromaDB (local disk)

WSGI: core.wsgi:application

Deployment process:

Clone repo to EC2

Create virtual environment

Install dependencies

Run migrations + collect static files

Start systemd service

Deployment challenges and fixes:

pymupdf and rank-bm25 were missing from requirements.txt → added.

torch installed with CUDA (2.5 GB) → replaced with CPU-only version (~200 MB).

llama-index-llms-openai-like version incompatibility with llama-index-core → upgraded to compatible versions.

WindowsPath vs str mismatch in views.py → fixed with explicit str() conversion.

/tmp tmpfs (455 MB) too small for numpy compilation → remounted to 2 GB.

Known Limitations

Text-only: Only text is extracted from PDFs. Embedded images, diagrams, and scanned pages are not supported. A natural extension would be OCR (for scanned PDFs) or a multimodal LLM.

No tables: Table content is extracted as flat text but structure isn't preserved.

JWT token expiry: Currently no refresh-token flow, so users must log in again after 60 minutes.

ChromaDB persistence: On the free-tier EC2 deployment, vector data doesn't persist across server restarts.

Single-threaded: The deployed app uses gunicorn with 1 worker (memory-constrained on t3.micro).

No HTTPS: Currently HTTP only. In production, nginx + Let's Encrypt would add SSL.

Dynamic IP: The EC2 public IP is not allocated as an Elastic IP, so it may change on instance restart.

🛠️ Tech Stack
Layer	Technology
Backend	Django, Django REST Framework
Auth	JWT (SimpleJWT)
RAG Framework	LlamaIndex
Vector Database	ChromaDB
LLM	Groq (openai/gpt-oss-120b)
Embeddings	HuggingFace BAAI/bge-small-en-v1.5
Re-ranker	Cross-Encoder ms-marco-MiniLM-L-6-v2
Keyword Search	BM25 (rank_bm25)
PDF Parsing	PyMuPDF
Frontend	HTML, CSS, Vanilla JavaScript
Deployment	AWS EC2, Gunicorn, systemd
🧠 Architecture
text
User Question
     ↓
[Input Guardrail] → block prompt injection
     ↓
Vector Search (ChromaDB, top 15)  +  BM25 Keyword Search (top 15)
     ↓
Reciprocal Rank Fusion → merge into top 10
     ↓
Cross-Encoder Re-ranker → select top 3
     ↓
LLM (Groq) → generate answer with strict grounding
     ↓
Answer or "Not in document"
For a deep dive into architecture, trade-offs, and known limitations, see SYSTEM_DESIGN.md.

 Project Structure
text
rag-doc-qa/
├── api/
│   ├── models.py            # Document model
│   ├── views.py             # Upload & Query API endpoints
│   ├── urls.py              # API routes
│   ├── rag_engine.py        # RAG pipeline (chunk, embed, retrieve, rerank, generate)
│   └── evaluation.py        # 18-question eval set + scoring harness
├── core/
│   ├── settings.py          # Django configuration
│   └── urls.py              # Main URL routing
├── templates/
│   └── index.html           # Frontend UI
├── media/pdfs/              # Uploaded PDFs (gitignored)
├── chroma_db/               # ChromaDB persistent storage (gitignored)
├── docs/
│   └── screenshots/         # Screenshots for README
├── run_eval.py              # Eval runner script
├── manage.py
├── requirements.txt
├── README.md
└── SYSTEM_DESIGN.md

Installation
1. Clone
bash
git clone https://github.com/Adepuharshavardhan2001/RAG-DOC-QA.git
cd RAG-DOC-QA
2. Create Virtual Environment
bash
python -m venv venv
venv\Scripts\activate       # Windows
source venv/bin/activate    # Mac/Linux
3. Install Dependencies
bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
4. Set Up Environment
Create a .env file in the project root:

text
GROQ_API_KEY=gsk_your_groq_api_key_here
DJANGO_SECRET_KEY=your-secret-key-here
DEBUG=True
ALLOWED_HOSTS=*
Get a free Groq API key at console.groq.com.

5. Run Migrations
bash
python manage.py migrate
6. Create Superuser
bash
python manage.py createsuperuser
7. Run Server
bash
python manage.py runserver
8. Open
Visit http://127.0.0.1:8000/ and log in.

