import chromadb
from pypdf import PdfReader

from django.conf import settings

from llama_index.core import (
    Document,
    Settings,
    VectorStoreIndex,
    StorageContext,
)

from llama_index.core.node_parser import SimpleNodeParser
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore


# -----------------------------
# GLOBAL SETTINGS
# -----------------------------

embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

llm = Groq(
    model="llama-3.3-70b-versatile",
    api_key=settings.GROQ_API_KEY
)

Settings.embed_model = embed_model
Settings.llm = llm


# -----------------------------
# PDF PROCESSING
# -----------------------------

def process_and_store_document(file_path, user_id):

    try:
        reader = PdfReader(file_path)

        text = ""

        for page in reader.pages:
            page_text = page.extract_text()

            if page_text:
                text += page_text + "\n"

        print("=" * 50)
        print("Extracted Characters:", len(text))
        print("=" * 50)

        if len(text.strip()) == 0:
            raise Exception(
                "No text found in PDF. PDF may be scanned/image-based."
            )

    except Exception as e:
        raise Exception(f"PDF Read Error: {str(e)}")

    # Create document
    documents = [Document(text=text)]

    # Split into chunks
    parser = SimpleNodeParser.from_defaults(
        chunk_size=512,
        chunk_overlap=50
    )

    nodes = parser.get_nodes_from_documents(documents)

    print("Total Nodes Created:", len(nodes))

    # -----------------------------
    # CHROMA DB
    # -----------------------------

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection_name = f"user_{user_id}"

    try:
        client.delete_collection(collection_name)
        print("Old Collection Deleted")
    except:
        pass

    collection = client.create_collection(collection_name)

    print("Before Indexing:", collection.count())

    vector_store = ChromaVectorStore(
        chroma_collection=collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    # Store nodes
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model
    )

    print("After Indexing:", collection.count())

    return {
        "status": "success",
        "chunks": len(nodes)
    }


# -----------------------------
# QUERY DOCUMENTS
# -----------------------------

def query_documents(query_text, user_id):

    collection_name = f"user_{user_id}"

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    try:
        collection = client.get_collection(collection_name)
    except Exception as e:
        return f"Collection Error: {str(e)}"

    print("=" * 50)
    print("Collection:", collection_name)
    print("Collection Count:", collection.count())
    print("=" * 50)

    if collection.count() == 0:
        return "No documents found."

    try:
        # Create embedding for the question
        query_embedding = embed_model.get_text_embedding(
            query_text
        )

        # Query Chroma directly
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )

        documents = results.get("documents", [])

        if not documents or len(documents[0]) == 0:
            return "No relevant information found."

        context = "\n\n".join(documents[0])

        prompt = f"""
You are a helpful document assistant.

Answer the question only using the provided context.

Context:
{context}

Question:
{query_text}

Answer:
"""

        response = llm.complete(prompt)

        return str(response)

    except Exception as e:
        print("QUERY ERROR:", str(e))
        raise Exception(str(e))