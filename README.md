# RAG Document Q&A System

A production-grade Retrieval-Augmented Generation (RAG) system that lets users upload PDF documents and ask natural-language questions about their content. Built with Django, LlamaIndex, ChromaDB, HuggingFace embeddings, and Groq.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Django](https://img.shields.io/badge/Django-4.2-green)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-RAG-orange)
![ChromaDB](https://img.shields.io/badge/ChromaDB-VectorDB-purple)

---

## 🚀 Features

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

## 📊 Evaluation Results

The system is evaluated on an **18-question test set** covering six categories, using the paper *"A Comprehensive Survey on Graph Neural Networks"* (20 pages, IEEE) as the test document.

| Category | Questions | Passed |
|---|---|---|
| Factual retrieval (title, authors, acronyms) | 5 | 5/5 ✅ |
| Taxonomy (GNN categories, GAE definition) | 2 | 2/2 ✅ |
| Multi-item retrieval (datasets, directions, applications) | 3 | 2/3 ⚠️ |
| Reasoning (GNN vs network embedding, spectral vs spatial) | 2 | 2/2 ✅ |
| Hallucination refusal (off-topic questions) | 6 | 6/6 ✅ |
| **Total** | **18** | **17/18 (94.4%)** |

**How to reproduce:**
```bash
python run_eval.py
