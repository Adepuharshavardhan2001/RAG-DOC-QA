import logging
import chromadb
from pypdf import PdfReader
from django.conf import settings
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


def _get_chroma_client():
    return chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)


def process_and_store_document(file_path: str, user_id: int) -> dict:
    try:
        reader = PdfReader(file_path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"

        logger.info(f"Extracted {len(text)} characters from PDF")

        if len(text.strip()) == 0:
            raise ValueError("No text found in PDF. PDF may be scanned/image-based.")

    except Exception as e:
        logger.error(f"PDF read error: {e}")
        raise Exception(f"PDF Read Error: {str(e)}")

    documents = [Document(text=text)]
    parser = SimpleNodeParser.from_defaults(chunk_size=512, chunk_overlap=50)
    nodes = parser.get_nodes_from_documents(documents)
    logger.info(f"Created {len(nodes)} chunks")

    client = _get_chroma_client()
    collection_name = f"user_{user_id}"

    try:
        client.delete_collection(collection_name)
        logger.info(f"Deleted old collection: {collection_name}")
    except Exception:
        logger.info(f"No existing collection: {collection_name}")

    collection = client.create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    llm = Groq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)
    LlamaSettings.llm = llm

    VectorStoreIndex(nodes, storage_context=storage_context, embed_model=embed_model)
    logger.info(f"Indexed {collection.count()} chunks into {collection_name}")

    return {"status": "success", "chunks": len(nodes)}


def query_documents(query_text: str, user_id: int) -> str:
    collection_name = f"user_{user_id}"
    client = _get_chroma_client()

    try:
        collection = client.get_collection(collection_name)
    except Exception:
        logger.error(f"Collection not found: {collection_name}")
        return "No documents found. Please upload a PDF first."

    logger.info(f"Querying: {collection_name} ({collection.count()} docs)")

    if collection.count() == 0:
        return "No documents found. Please upload a PDF first."

    try:
        query_embedding = embed_model.get_text_embedding(query_text)
        results = collection.query(query_embeddings=[query_embedding], n_results=3)
        documents = results.get("documents", [])

        if not documents or len(documents[0]) == 0:
            return "No relevant information found in your documents."

        context = "\n\n".join(documents[0])

        llm = Groq(model="llama-3.3-70b-versatile", api_key=settings.GROQ_API_KEY)

        prompt = f"""You are a helpful document assistant.
Answer the question using only the provided context.
If the context doesn't contain the answer, say so honestly.
Keep your answer clear and concise. Use plain text only.

Context:
{context}

Question: {query_text}

Answer:"""

        response = llm.complete(prompt)
        answer = str(response).strip()
        return answer

    except Exception as e:
        logger.error(f"Query error: {e}", exc_info=True)
        return "Sorry, there was an error processing your question. Please try again."