# 📄 RAG Document Q&A System

A full-stack web application that allows users to upload PDF documents and ask questions about their content using Retrieval-Augmented Generation (RAG) with Groq LLM and ChromaDB.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![Django](https://img.shields.io/badge/Django-4.2-green)
![LlamaIndex](https://img.shields.io/badge/LlamaIndex-0.9-orange)
![JWT](https://img.shields.io/badge/Auth-JWT-red)

---

## 🚀 Features

- **PDF Ingestion** — Upload PDFs via web interface. Text is extracted, chunked, and embedded
- **Intelligent Q&A** — Ask questions about your documents. System retrieves relevant chunks and generates answers
- **User Isolation** — Each user has a private ChromaDB collection. Users can only query their own documents
- **JWT Authentication** — Secure login with access and refresh tokens
- **Local Embeddings** — Uses BAAI/bge-small-en-v1.5 from HuggingFace (free, no API cost)
- **Groq LLM** — High-speed inference with Llama 3.3 70B
- **Clean UI** — Simple login, upload, and query interface

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | Django, Django REST Framework |
| **Auth** | Simple JWT |
| **RAG Framework** | LlamaIndex |
| **Vector Database** | ChromaDB |
| **LLM** | Groq (Llama 3.3 70B) |
| **Embeddings** | HuggingFace (BAAI/bge-small-en-v1.5) |
| **PDF Parsing** | PyPDF |
| **Frontend** | HTML, CSS, Vanilla JavaScript |

---

## 📂 Project Structure

rag_doc-qa/

├── api/

│ ├── models.py # Document model with validation

│ ├── views.py # Upload & Query API endpoints

│ ├── urls.py # API routes

│ └── rag_engine.py # RAG pipeline (chunk, embed, query)


├── core/

│ ├── settings.py # Django configuration

│ └── urls.py # Main URL routing

├── templates/

│ └── index.html # Frontend UI

├── manage.py

├── requirements.txt

├── Dockerfile

├── docker-compose.yml

├── .env.example

└── README.md


## 🔧 Installation

### 1. Clone

git clone https://github.com/Adepuharshavardhan2001/rag-doc-qa.git
cd rag-doc-qa

2. Create Virtual Environment

python -m venv venv
venv\Scripts\activate

3. Install Dependencies
bash
pip install -r requirements.txt

5. Set Up Environment
Create .env file:


DJANGO_SECRET_KEY=your-secret-key
GROQ_API_KEY=gsk_your_key_here
DEBUG=True
ALLOWED_HOSTS=*

5. Run Migrations
bash
python manage.py migrate

7. Create Superuser

python manage.py createsuperuser

8. Run Server
python manage.py runserver

10. Open

http://127.0.0.1:8000/
