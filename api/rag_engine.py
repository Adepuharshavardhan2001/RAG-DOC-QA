import logging
import chromadb
import pymupdf  # PyMuPDF
from django.conf import settings
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi
from llama_index.core import (
    Document,
    Settings as LlamaSettings,
    VectorStoreIndex,
    StorageContext,
)
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

logger = logging.getLogger(__name__)

embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
LlamaSettings.embed_model = embed_model

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# Keywords that usually appear in "metadata" questions about a paper
PAGE_1_KEYWORDS = [
    "title", "author", "authors", "abstract", "published",
    "citation", "cite", "paper name", "written by", "who wrote"
]


def _get_chroma_client():
    return chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)


def _is_metadata_query(query: str) -> bool:
    """Return True if the query is likely asking about title/authors/abstract."""
    query_lower = query.lower()
    return any(kw in query_lower for kw in PAGE_1_KEYWORDS)


def process_and_store_document(file_path: str, user_id: int) -> dict:
    try:
        doc = pymupdf.open(file_path)
        all_text = ""
        pages = []
        for page_num, page in enumerate(doc, start=1):
            page_text = page.get_text()
            pages.append((page_num, page_text))
            all_text += page_text + "\n"
        doc.close()

        logger.info(f"Extracted {len(all_text)} characters from PDF")

        if len(all_text.strip()) == 0:
            raise ValueError("No text found in PDF. PDF may be scanned/image-based.")

    except Exception as e:
        logger.error(f"PDF read error: {e}")
        raise Exception(f"PDF Read Error: {str(e)}")

    # --- SUMMARY CHUNK ---
    # Always include a synthetic summary chunk with title/authors/abstract.
    # This fixes retrieval for metadata questions like "What is the title?"
    first_page_text = pages[0][1] if pages else ""
    summary_text = f"""Document Summary:
Type: Research Paper / Survey
Title and authors and abstract (from first page):
{first_page_text[:1500]}
"""

    documents = [
        Document(
            text=summary_text,
            metadata={"page": 0, "type": "summary"}
        )
    ]

    # --- PAGE CHUNKS ---
    documents += [
        Document(text=page_text, metadata={"page": page_num})
        for page_num, page_text in pages
        if page_text.strip()
    ]

    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    logger.info(f"Created {len(nodes)} chunks")

    client = _get_chroma_client()
    collection_name = f"user_{user_id}"

    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    llm = Groq(model="openai/gpt-oss-120b", api_key=settings.GROQ_API_KEY)
    LlamaSettings.llm = llm

    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    logger.info(f"Indexed {collection.count()} chunks into {collection_name}")

    return {"status": "success", "chunks": len(nodes)}


def _reciprocal_rank_fusion(list_a, list_b, k=60):
    scores = {}
    for rank, item in enumerate(list_a):
        scores[item] = scores.get(item, 0) + 1.0 / (k + rank)
    for rank, item in enumerate(list_b):
        scores[item] = scores.get(item, 0) + 1.0 / (k + rank)
    return sorted(scores.keys(), key=lambda x: scores[x], reverse=True)


def query_documents(query_text: str, user_id: int) -> str:
    # --- INPUT GUARDRAIL ---
    BLOCKED_PATTERNS = [
        "ignore previous", "ignore instructions", "system prompt",
        "jailbreak", "pretend you are", "act as", "forget everything",
        "disregard", "override"
    ]
    query_lower = query_text.lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern in query_lower:
            return "I can only answer questions related to the uploaded document."

    if len(query_text.strip()) < 3:
        return "Please ask a more specific question."

    # --- RETRIEVAL ---
    collection_name = f"user_{user_id}"
    client = _get_chroma_client()

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        return "No documents found. Please upload a PDF first."

    if collection.count() == 0:
        return "No documents found. Please upload a PDF first."

    try:
        # --- VECTOR SEARCH ---
        query_embedding = embed_model.get_text_embedding(query_text)
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=15,
            include=["documents", "metadatas"]
        )
        vector_docs = vector_results.get("documents", [[]])[0]
        vector_metas = vector_results.get("metadatas", [[]])[0]

        # --- KEYWORD SEARCH (BM25) ---
        all_data = collection.get(include=["documents"])
        all_chunks = all_data.get("documents", [])

        if not all_chunks:
            return "No relevant information found in your documents."

        tokenized_corpus = [chunk.lower().split() for chunk in all_chunks]
        bm25 = BM25Okapi(tokenized_corpus)

        tokenized_query = query_text.lower().split()
        bm25_scores = bm25.get_scores(tokenized_query)

        top_bm25_indices = bm25_scores.argsort()[-15:][::-1]
        bm25_docs = [all_chunks[i] for i in top_bm25_indices]

        # --- HYBRID: Merge with RRF ---
        merged = _reciprocal_rank_fusion(vector_docs, bm25_docs)
        candidates = merged[:10]

        if not candidates:
            return "No relevant information found in your documents."

        # --- RE-RANK ---
        pairs = [(query_text, chunk) for chunk in candidates]
        scores = reranker.predict(pairs)
        top_indices = scores.argsort()[-3:][::-1]
        top_chunks = [candidates[i] for i in top_indices]

        context = "\n\n".join(top_chunks)

        # --- GENERATION ---
        llm = Groq(model="openai/gpt-oss-120b", api_key=settings.GROQ_API_KEY)

        prompt = f"""You are a helpful document assistant.
Answer the question using only the provided context.

CRITICAL RULES:
1. Use EXACT terminology from the document. Do not paraphrase or guess.
2. Do NOT expand acronyms unless the document explicitly defines them.
3. If you are unsure of an acronym's expansion, quote it as it appears.
4. When asked for a specific formula, return the exact equation as written.
5. Provide as much relevant detail as the context supports.
6. If the context doesn't contain the answer, say so honestly.
7. When asked to perform a calculation, show your work step by step.

Context:
{context}

Question: {query_text}

Answer:"""

        response = llm.complete(prompt)
        answer = str(response).strip()
        return answer

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return f"Error: {str(e)}"