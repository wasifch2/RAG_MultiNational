from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import os
import shutil

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_community.vectorstores import FAISS
# --- NEW IMPORTS ---
from langchain_community.embeddings import HuggingFaceEmbeddings
from backend.document_loader import load_all_multinational_documents
from backend.config import RAGConfig  # Import your configuration class

# --------------------


# --- Configuration ---
# Initialize RAGConfig once to get model names
CONFIG = RAGConfig()

# Define the directory for the single, combined vector store
# Path(__file__).parent.parent resolves to the RAG_4_Scratch-main root folder
VECTOR_STORE_DIR = os.path.join(Path(__file__).parent.parent, 'vector_store', 'multinational_rag_index')
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)  # Ensure directory exists
print(f"Vector Store Target Directory: {VECTOR_STORE_DIR}")

# This is the vector database layer: which builds and loads FAISS vector stores using LangChain-Documents.

# During offline step, called by the Vector DB Builder page to create FAISS DBs.
# During online step, called by RAG pipelines to load the correct DB and create retrievers.


# backend/vector_store.py
# Simple in-memory cache: {path -> FAISS vector store}
_VECTOR_STORE_CACHE: dict[str, FAISS] = {}


# --- NEW FUNCTION: Embedding Model Initialization ---
def get_embedding_model() -> Optional[Embeddings]:
    """Initializes the embedding model based on configuration."""
    if CONFIG.embedding_provider.lower() == "huggingface":
        print(f"Initializing HuggingFace Embeddings: {CONFIG.embedding_model_name}")
        try:
            # Uses the model name from config.py: sentence-transformers/all-MiniLM-L6-v2
            return HuggingFaceEmbeddings(model_name=CONFIG.embedding_model_name)
        except Exception as e:
            print(f"Error initializing HuggingFace model. Did you run 'pip install sentence-transformers'? Error: {e}")
            return None
    else:
        # Placeholder for other providers if needed
        print(f"Warning: Unsupported embedding provider: {CONFIG.embedding_provider}. Cannot initialize model.")
        return None


# ----------------------------------------------------


def build_vector_store(
        docs: List[Document],
        embedding_model,
        target_dir: str,
) -> None:
    os.makedirs(target_dir, exist_ok=True)

    vs = FAISS.from_documents(docs, embedding_model)
    vs.save_local(target_dir)

    _VECTOR_STORE_CACHE[target_dir] = vs


def load_vector_store(
        path: str,
        embedding_model,
) -> FAISS:
    cached = _VECTOR_STORE_CACHE.get(path)
    if cached is not None:
        return cached

    vs = FAISS.load_local(
        path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    _VECTOR_STORE_CACHE[path] = vs
    return vs


def clear_vector_store_cache(path: str) -> None:
    """Delete vector store from disk and from in-memory cache."""
    if path in _VECTOR_STORE_CACHE:
        del _VECTOR_STORE_CACHE[path]
    if os.path.isdir(path):
        shutil.rmtree(path)


# --- NEW FUNCTION: Combined Vector Store Builder ---
def build_multinational_vector_store():
    """
    Loads all multinational documents and creates a single vector index.
    Uses the existing build_vector_store utility.
    """
    print("--- Phase 1: Starting Multinational Vector Store Construction ---")

    # 1. Load Documents with Metadata
    documents = load_all_multinational_documents()
    if not documents:
        print("No documents loaded. Aborting index build.")
        return None

    # 2. Initialize Embedding Model
    embeddings = get_embedding_model()
    if embeddings is None:
        print("Embedding model failed to initialize. Cannot build index.")
        return None

    # 3. Create and Save the Vector Store
    print(f"Creating vector store using {len(documents)} documents...")
    try:
        # Use your existing, clean utility function to build and save the index
        build_vector_store(
            docs=documents,
            embedding_model=embeddings,
            target_dir=VECTOR_STORE_DIR
        )
        print(f"Vector store successfully built and saved to: {VECTOR_STORE_DIR}")

    except Exception as e:
        print(f"Error building or saving vector store: {e}")
        return None


# --- NEW FUNCTION: Combined Vector Store Loader ---
def get_multinational_vector_store() -> Optional[FAISS]:
    """Loads the pre-built multinational vector store from disk."""
    try:
        embeddings = get_embedding_model()
        if embeddings is None:
            print("Cannot load store: Embedding model failed to initialize.")
            return None

        vector_store = load_vector_store(VECTOR_STORE_DIR, embeddings)
        print(f"Multinational vector store loaded from: {VECTOR_STORE_DIR}")
        return vector_store
    except Exception as e:
        print(f"Error loading vector store: {e}")
        print("Attempting to rebuild the vector store...")
        # If the load fails, try to build it
        build_multinational_vector_store()

        # Try loading again after rebuilding
        try:
            embeddings = get_embedding_model()
            return load_vector_store(VECTOR_STORE_DIR, embeddings)
        except Exception as re:
            print(f"Rebuild failed. Final error: {re}")
            return None


# --- Main function to run the builder ---
if __name__ == '__main__':
    # This will execute the build process when you run the file directly
    build_multinational_vector_store()